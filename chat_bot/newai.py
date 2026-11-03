import json
import logging
import os
import sqlite3
import threading
import time
from typing import Optional
from uuid import uuid4

from flask import Flask, g, jsonify, make_response, render_template, request
from groq import Groq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("aura")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", GROQ_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DB_PATH = os.getenv("DB_PATH", "agent_memory.db")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "aura_session")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))

CONTEXT_MESSAGES = 4
MAX_INPUT_CHARS = 200_000
MAX_HISTORY_CHARS = 75_000
MAX_HISTORY_RECORDS = max(2, int(os.getenv("MAX_HISTORY_RECORDS", "2000")))
CHAT_RATE_LIMIT = max(1, int(os.getenv("CHAT_RATE_LIMIT", "12")))
CHAT_RATE_WINDOW_SECONDS = max(1, int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60")))
MAX_CONTENT_LENGTH_BYTES = 16 * 1024 * 1024


class ModelUnavailable(RuntimeError):
    """Raised when no configured language-model provider can answer."""


class SimpleMemory:
    """SQLite-backed memory with serialized access and bounded chat history."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            parent = os.path.dirname(self.db_path) or "."
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(
            self.db_path, timeout=30, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 30000")
        if self.db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.commit()

    def set_fact(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO facts (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=CURRENT_TIMESTAMP",
                (key, value),
            )
            self.conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT key, value FROM facts ORDER BY key"
            ).fetchall()
            return {row["key"]: row["value"] for row in rows}

    def delete_fact(self, key: str) -> bool:
        with self._lock:
            cursor = self.conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            self.conn.commit()
            return cursor.rowcount > 0

    def _prune_history(self, keep: int) -> None:
        self.conn.execute(
            "DELETE FROM history WHERE id NOT IN "
            "(SELECT id FROM history ORDER BY id DESC LIMIT ?)",
            (keep,),
        )

    def add_history(self, role: str, content: str, meta: Optional[dict] = None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
                (role, content, json.dumps(meta) if meta is not None else None),
            )
            self._prune_history(MAX_HISTORY_RECORDS)

    def record_exchange(self, user_text: str, assistant_text: str, meta: Optional[dict] = None) -> None:
        with self._lock, self.conn:
            self.conn.executemany(
                "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
                [
                    ("user", user_text, json.dumps(meta) if meta is not None else None),
                    ("assistant", assistant_text, json.dumps(meta) if meta is not None else None),
                ],
            )
            self._prune_history(MAX_HISTORY_RECORDS)

    def recent_history(self, limit: int = 10) -> list[dict[str, str]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {"role": row["role"], "content": row["content"]}
                for row in reversed(rows)
            ]

    def clear_history(self) -> int:
        with self._lock, self.conn:
            cursor = self.conn.execute("DELETE FROM history")
            return cursor.rowcount

    def stats(self) -> dict[str, int]:
        with self._lock:
            fact_count = self.conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
            history_count = self.conn.execute(
                "SELECT COUNT(*) FROM history"
            ).fetchone()[0]
            return {"facts": fact_count, "history": history_count}

    def close(self) -> None:
        with self._lock:
            self.conn.close()


class Aura:
    """Assistant wrapper that keeps a small memory and recent conversation context."""

    COMMAND_HELP = "Commands: /help, /memory, /clear, /stats, /remember key=value"

    def __init__(self, memory: Optional[SimpleMemory] = None):
        self.memory = memory or SimpleMemory(DB_PATH)

    def _facts_block(self) -> str:
        facts = self.memory.all_facts()
        if not facts:
            return "No remembered facts."
        return "\n".join(f"- {key}: {value}" for key, value in sorted(facts.items()))

    def _context_messages(self) -> list[dict[str, str]]:
        history = self.memory.recent_history(limit=CONTEXT_MESSAGES * 2)
        selected = []
        chars = 0
        for item in reversed(history):
            if item["role"] not in {"user", "assistant"}:
                continue
            if selected and chars + len(item["content"]) > MAX_HISTORY_CHARS:
                break
            selected.append(item)
            chars += len(item["content"])
        return list(reversed(selected))

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": "You are AURA, a helpful assistant."}]
        messages.extend(self._context_messages())
        messages.append({"role": "user", "content": user_input})
        return messages

    def _complete_with_groq(self, messages: list[dict[str, str]], model: str) -> str:
        client = Groq(api_key=GROQ_API_KEY, timeout=60, max_retries=0)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
            top_p=1,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("The provider returned an empty completion.")
        return content

    def _complete_with_ollama(self, messages: list[dict[str, str]], model: str) -> str:
        import urllib.request

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 800,
        }
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not body.get("choices"):
            raise RuntimeError("The Ollama API returned no completion choices.")
        content = str(body["choices"][0].get("message", {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("The provider returned an empty completion.")
        return content

    def _complete(self, messages: list[dict[str, str]]) -> str:
        failures = []
        if GROQ_API_KEY:
            try:
                return self._complete_with_groq(messages, GROQ_MODEL)
            except Exception as exc:
                failures.append(("Groq", type(exc).__name__))
        try:
            return self._complete_with_ollama(messages, OLLAMA_MODEL)
        except Exception as exc:
            failures.append(("Ollama", type(exc).__name__))

        LOGGER.warning(
            "All configured model providers failed: %s",
            ", ".join(f"{name} ({kind})" for name, kind in failures),
        )
        raise ModelUnavailable("The model service is temporarily unavailable.") from None

    def answer_directly(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return "Please provide a message."
        if len(text) > MAX_INPUT_CHARS:
            return f"Messages must be {MAX_INPUT_CHARS:,} characters or fewer."

        lowered = text.lower()
        if lowered in {"/help", "help"}:
            return self.COMMAND_HELP
        if lowered in {"/memory", "memory"}:
            return self._facts_block()
        if lowered in {"/stats", "stats"}:
            stats = self.memory.stats()
            return f"Facts: {stats['facts']} | History: {stats['history']}"
        if lowered in {"/clear", "clear"}:
            self.memory.clear_history()
            return "Conversation history cleared."
        if lowered.startswith("/remember "):
            _, _, payload = text.partition(" ")
            if "=" not in payload:
                return "Usage: /remember key=value"
            key, value = payload.split("=", 1)
            if not key.strip() or not value.strip():
                return "Usage: /remember key=value"
            self.memory.set_fact(key.strip(), value.strip())
            return f"Remembered: {key.strip()} = {value.strip()}"

        return self._complete(self._build_messages(text))

    def handle(self, user_input: str) -> str:
        return self.answer_directly(user_input)

    def answer_and_record(self, user_input: str, session_id: Optional[str] = None) -> str:
        answer = self.answer_directly(user_input)
        if user_input.strip().lower() not in {"/clear", "clear"}:
            self.memory.record_exchange(
                user_input.strip(),
                answer,
                meta={"session_id": session_id} if session_id else None,
            )
        return answer

    def run_command(self, text: str) -> bool:
        text = text.strip()
        if not text or not text.startswith("/"):
            return False
        self.answer_and_record(text)
        return True


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_BYTES
app.config["JSON_SORT_KEYS"] = False
aura = Aura(memory=SimpleMemory(DB_PATH))
_rate_lock = threading.Lock()
_rate_events: dict[str, list[float]] = {}


def get_or_create_session_id() -> str:
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        return existing
    session_id = uuid4().hex
    response = make_response()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
    )
    g.session_cookie_response = response
    return session_id


@app.before_request
def bind_request_context():
    g.request_id = uuid4().hex
    g.session_id = get_or_create_session_id()
    if hasattr(g, "session_cookie_response"):
        response = g.session_cookie_response
        g.session_cookie_response = None
        return response
    return None


@app.after_request
def add_request_headers(response):
    response.headers["X-Request-ID"] = g.request_id
    return response


@app.before_request
def limit_chat_requests():
    if request.endpoint != "chat" or request.method != "POST":
        return None

    now = time.monotonic()
    client = request.remote_addr or "unknown"
    cutoff = now - CHAT_RATE_WINDOW_SECONDS
    with _rate_lock:
        for address, stamps in list(_rate_events.items()):
            active = [stamp for stamp in stamps if stamp > cutoff]
            if active:
                _rate_events[address] = active
            else:
                _rate_events.pop(address, None)
        events = _rate_events.setdefault(client, [])
        if len(events) >= CHAT_RATE_LIMIT:
            return jsonify({"error": "Too many requests. Please wait and try again."}), 429
        events.append(now)
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def stats_api():
    stats = aura.memory.stats()
    return jsonify({
        "status": "ok",
        "facts": stats["facts"],
        "history": stats["history"],
        "session_id": g.session_id,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400
    user_input = payload.get("message")
    if not isinstance(user_input, str) or not user_input.strip():
        return jsonify({"error": "Message is required and must be text."}), 400
    user_input = user_input.strip()
    if len(user_input) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Message must be {MAX_INPUT_CHARS:,} characters or fewer."}), 413

    try:
        response = aura.answer_and_record(user_input, session_id=g.session_id)
    except ModelUnavailable:
        return jsonify({"error": "The model service is temporarily unavailable."}), 503
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify({"error": "The chat service could not complete this request."}), 502
    return jsonify({"response": response})


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large."}), 413


def main(db_path: str = DB_PATH) -> None:
    global aura
    aura = Aura(memory=SimpleMemory(db_path))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)


if __name__ == "__main__":
    main()
