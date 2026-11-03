import json
import os
import sqlite3
import threading
from typing import Optional

from flask import Flask, jsonify, render_template, request
from groq import Groq


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DB_PATH = os.getenv("DB_PATH", "agent_memory.db")
CONTEXT_MESSAGES = 4
MAX_INPUT_CHARS = 200_000
MAX_HISTORY_RECORDS = max(100, int(os.getenv("MAX_HISTORY_RECORDS", "2000")))
CHAT_RATE_LIMIT = max(1, int(os.getenv("CHAT_RATE_LIMIT", "12")))
CHAT_RATE_WINDOW_SECONDS = max(1, int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60")))
MAX_CONTENT_LENGTH_BYTES = 16 * 1024 * 1024


class SimpleMemory:
    """SQLite-backed memory with serialized access and bounded chat history."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
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

    def add_history(self, role: str, content: str, meta: Optional[dict] = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
                (role, content, json.dumps(meta) if meta is not None else None),
            )
            self.conn.execute(
                "DELETE FROM history WHERE id NOT IN "
                "(SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                (MAX_HISTORY_RECORDS,),
            )
            self.conn.commit()

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
        with self._lock:
            cursor = self.conn.execute("DELETE FROM history")
            self.conn.commit()
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
    """Assistant wrapper that keeps a small amount of memory and chat history."""

    COMMAND_HELP = "Commands: /help, /memory, /clear, /stats, /remember key=value"

    def __init__(self, memory: Optional[SimpleMemory] = None):
        self.memory = memory or SimpleMemory(DB_PATH)

    def _facts_block(self) -> str:
        facts = self.memory.all_facts()
        if not facts:
            return "No remembered facts."
        return "\n".join(f"- {key}: {value}" for key, value in sorted(facts.items()))

    def _context_messages(self, user_input: str) -> list[dict[str, str]]:
        history = self.memory.recent_history(limit=CONTEXT_MESSAGES * 2)
        user_messages = [item for item in history if item["role"] == "user"]
        if user_input and user_input.strip():
            user_messages = [
                item for item in user_messages
                if item["content"].strip() != user_input.strip()
            ]
        return user_messages[-CONTEXT_MESSAGES:]

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": "You are AURA, a helpful assistant."}]
        messages.extend(self._context_messages(user_input))
        messages.append({"role": "user", "content": user_input})
        return messages

    def _complete_with_groq(self, messages: list[dict[str, str]], model: str) -> str:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
            top_p=1,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()

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
        content = body["choices"][0].get("message", {}).get("content") or ""
        return str(content).strip()

    def _complete(self, messages: list[dict[str, str]]) -> str:
        failures = []
        if GROQ_API_KEY:
            try:
                return self._complete_with_groq(messages, GROQ_MODEL)
            except Exception as exc:
                failures.append(f"Groq: {exc}")
        try:
            return self._complete_with_ollama(messages, OLLAMA_MODEL)
        except Exception as exc:
            failures.append(f"Ollama: {exc}")
        return "Model request failed. Check the configured provider and model. " + "; ".join(failures)

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

    def answer_and_record(self, user_input: str) -> str:
        answer = self.answer_directly(user_input)
        self._record(user_input, answer)
        return answer

    def _record(self, user_input: str, answer: str, meta: Optional[dict] = None) -> None:
        self.memory.add_history("user", user_input.strip(), meta)
        self.memory.add_history("assistant", answer, meta)

    def run_command(self, text: str) -> bool:
        text = text.strip()
        if not text or not text.startswith("/"):
            return False
        self.answer_and_record(text)
        return True


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_BYTES
aura = Aura(memory=SimpleMemory(DB_PATH))
_rate_lock = threading.Lock()
_rate_events: dict[str, list[float]] = {}


@app.before_request
def limit_chat_requests():
    if request.endpoint != "chat":
        return None
    import time

    now = time.monotonic()
    client = request.remote_addr or "unknown"
    cutoff = now - CHAT_RATE_WINDOW_SECONDS
    with _rate_lock:
        events = [stamp for stamp in _rate_events.get(client, []) if stamp > cutoff]
        if len(events) >= CHAT_RATE_LIMIT:
            return jsonify({"error": "Too many requests. Please wait and try again."}), 429
        events.append(now)
        _rate_events[client] = events
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400
    user_input = str(payload.get("message", "")).strip()
    if not user_input:
        return jsonify({"error": "Message is required."}), 400
    if len(user_input) > MAX_INPUT_CHARS:
        return jsonify({"error": f"Message must be {MAX_INPUT_CHARS:,} characters or fewer."}), 413

    try:
        response = aura.answer_and_record(user_input)
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
