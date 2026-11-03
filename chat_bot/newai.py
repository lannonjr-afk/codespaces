import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from flask import Flask, g, jsonify, render_template, request
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
SESSION_TTL_SECONDS = max(60, int(os.getenv("SESSION_TTL_SECONDS", "86400")))

CONTEXT_MESSAGES = 4
MAX_INPUT_CHARS = 200_000
MAX_HISTORY_CHARS = 75_000
MAX_HISTORY_RECORDS = max(2, int(os.getenv("MAX_HISTORY_RECORDS", "2000")))
MAX_HISTORY_RECORDS -= MAX_HISTORY_RECORDS % 2
CHAT_RATE_LIMIT = max(1, int(os.getenv("CHAT_RATE_LIMIT", "12")))
CHAT_RATE_WINDOW_SECONDS = max(1, int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60")))
MAX_CONTENT_LENGTH_BYTES = 16 * 1024 * 1024


class ModelUnavailable(RuntimeError):
    """Raised when no configured language-model provider can answer."""


def current_utc_time() -> datetime:
    """Return the current timezone-aware UTC time for factual date/time answers."""
    return datetime.now(timezone.utc)


def format_utc_date(value: datetime) -> str:
    """Format a date without platform-specific strftime directives."""
    return f"{value:%A, %B} {value.day}, {value:%Y} UTC"


class SimpleMemory:
    """SQLite-backed memory with serialized access and bounded chat history."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
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
                """CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
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

    def _prune_history(self) -> None:
        self.conn.execute(
            "DELETE FROM history WHERE id NOT IN "
            "(SELECT id FROM history ORDER BY id DESC LIMIT ?)",
            (MAX_HISTORY_RECORDS,),
        )

    def add_history(self, role: str, content: str, meta: Optional[dict] = None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
                (role, content, json.dumps(meta) if meta is not None else None),
            )
            self._prune_history()

    def record_exchange(
        self, user_text: str, assistant_text: str, meta: Optional[dict] = None
    ) -> None:
        with self._lock, self.conn:
            encoded_meta = json.dumps(meta) if meta is not None else None
            self.conn.executemany(
                "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
                [
                    ("user", user_text, encoded_meta),
                    ("assistant", assistant_text, encoded_meta),
                ],
            )
            self._prune_history()

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
            facts = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            history = self.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            return {"facts": facts, "history": history}

    def close(self) -> None:
        with self._lock:
            self.conn.close()


class Aura:
    """Assistant wrapper that keeps a small amount of memory and recent context."""

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
        char_count = 0
        for item in reversed(history):
            if item["role"] not in {"user", "assistant"}:
                continue
            size = len(item["content"])
            if selected and char_count + size > MAX_HISTORY_CHARS:
                break
            selected.append(item)
            char_count += size
        return list(reversed(selected))

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        now = current_utc_time()
        system_prompt = (
            "You are AURA, a helpful assistant. Answer the user's actual question "
            "directly and concisely. Do not repeat previous greetings or unrelated "
            "answers. Do not claim to have feelings or a real-world identity. "
            "Never invent the user's name, identity, location, or personal details; "
            "if you do not know, say so plainly. The current date and time are "
            f"{now.isoformat(timespec='seconds')} (UTC). Use this as the reliable "
            "current date/time when asked, and state that it is UTC unless the user "
            "has specified a different timezone. Do not guess the user's local time."
        )
        return [
            {"role": "system", "content": system_prompt},
            *self._context_messages(),
            {"role": "user", "content": user_input},
        ]

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
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("The provider returned an empty completion.")
        return answer

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
        answer = str(body["choices"][0].get("message", {}).get("content") or "").strip()
        if not answer:
            raise RuntimeError("The provider returned an empty completion.")
        return answer

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

    def _remember_name_if_mentioned(self, text: str) -> Optional[str]:
        patterns = [
            r"\bmy name is\s+([A-Za-z][A-Za-z' -]{1,40})",
            r"\bi\s+am\s+([A-Za-z][A-Za-z' -]{1,40})",
            r"\bi'm\s+([A-Za-z][A-Za-z' -]{1,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate and candidate.lower() not in {"aura", "assistant", "ai"}:
                    self.memory.set_fact("name", candidate)
                    return candidate
        return None

    def answer_directly(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return "Please provide a message."
        if len(text) > MAX_INPUT_CHARS:
            return f"Messages must be {MAX_INPUT_CHARS:,} characters or fewer."

        remembered_name = self.memory.get_fact("name")
        if self._remember_name_if_mentioned(text):
            name = self.memory.get_fact("name")
            return f"Nice to meet you, {name}. I’ll remember that."

        normalized = re.sub(r"[^a-z0-9\s]", " ", text.casefold()).strip()
        question = " ".join(normalized.split())
        if question in {"hi", "hey", "hello", "good morning", "good afternoon", "good evening"}:
            return "Hi! What can I help you with?"
        if question in {"who are you", "what are you", "what is your name", "whats your name"}:
            return "I’m AURA, an AI assistant. I can answer questions and help with tasks."
        if question in {"who am i", "what is my name", "whats my name"}:
            if remembered_name:
                return f"You told me your name is {remembered_name}."
            return "I don’t know your name yet. You can tell me what to call you."
        if question in {"what day is it", "what date is it", "what is todays date", "whats todays date", "what is the date today", "what date is it today"}:
            return f"Today is {format_utc_date(current_utc_time())}."
        if question in {"what time is it", "what is the time", "whats the time", "current time"}:
            return f"The current time is {current_utc_time():%H:%M} UTC."

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
            meta = {"session_id": session_id} if session_id else None
            self.memory.record_exchange(user_input.strip(), answer, meta=meta)
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


@app.before_request
def bind_request_context():
    g.request_id = uuid4().hex
    supplied_id = request.cookies.get(SESSION_COOKIE_NAME, "")
    g.session_id = supplied_id if re.fullmatch(r"[a-f0-9]{32}", supplied_id) else uuid4().hex
    g.set_session_cookie = g.session_id != supplied_id


@app.after_request
def add_response_headers(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", uuid4().hex)
    if getattr(g, "set_session_cookie", False):
        response.set_cookie(
            SESSION_COOKIE_NAME,
            g.session_id,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            samesite="Lax",
            secure=request.is_secure,
        )
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def stats_api():
    stats = aura.memory.stats()
    return jsonify({"status": "ok", **stats})


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
        answer = aura.answer_and_record(user_input, session_id=g.session_id)
    except ModelUnavailable:
        return jsonify({"error": "The model service is temporarily unavailable."}), 503
    except Exception:
        app.logger.exception("Chat request failed; request_id=%s", g.request_id)
        return jsonify({"error": "The chat service could not complete this request."}), 502
    return jsonify({"response": answer})


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large."}), 413


def main(db_path: str = DB_PATH) -> None:
    global aura
    aura = Aura(memory=SimpleMemory(db_path))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)


if __name__ == "__main__":
    main()
