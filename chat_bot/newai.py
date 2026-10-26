import json
import os
import sqlite3
from typing import Optional

from flask import Flask, jsonify, render_template, request


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_PATH = os.getenv("DB_PATH", "agent_memory.db")
ENABLE_ATTACHMENTS = os.getenv("ENABLE_ATTACHMENTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CONTEXT_MESSAGES = 4
MAX_INPUT_CHARS = 200_000
MAX_HISTORY_CHARS = 75_000
MAX_CONTENT_LENGTH_BYTES = 16 * 1024 * 1024


class SimpleMemory:
    """Very small SQLite-backed memory store for user facts and recent chat history."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
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
        self.conn.execute(
            "INSERT INTO facts (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            (key, value),
        )
        self.conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete_fact(self, key: str) -> bool:
        cursor = self.conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        self.conn.commit()
        return cursor.rowcount > 0

    def add_history(self, role: str, content: str, meta: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
            (role, content, json.dumps(meta) if meta is not None else None),
        )
        self.conn.commit()

    def recent_records(self, limit: int = 10) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT role, content, meta, created_at FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_history(self, limit: int = 10) -> list[dict[str, str]]:
        rows = self.conn.execute(
            "SELECT role, content FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            results.append({"role": row["role"], "content": row["content"]})
        return list(reversed(results))

    def clear_history(self) -> int:
        cursor = self.conn.execute("DELETE FROM history")
        self.conn.commit()
        return cursor.rowcount

    def stats(self) -> dict[str, int]:
        fact_count = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        history_count = self.conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        return {"facts": fact_count, "history": history_count}

    def close(self) -> None:
        self.conn.close()


class Aura:
    """Simple assistant wrapper that keeps a memory and returns helpful replies."""

    COMMAND_HELP = (
        "Commands: /help, /memory, /clear, /stats"
    )

    def __init__(self, memory: Optional[SimpleMemory] = None):
        self.memory = memory or SimpleMemory(DB_PATH)

    def _facts_block(self) -> str:
        facts = self.memory.all_facts()
        if not facts:
            return "No remembered facts."
        return "\n".join(f"- {key}: {value}" for key, value in sorted(facts.items()))

    def _context_messages(self, user_input: str) -> list[dict[str, str]]:
        history = self.memory.recent_history(limit=CONTEXT_MESSAGES)
        trimmed = user_input.strip()
        if not trimmed:
            return history
        return [*history, {"role": "user", "content": trimmed}]

    def answer_directly(self, user_input: str) -> str:
        text = user_input.strip()
        if not text:
            return "Please provide a message."

        if text.lower() in {"/help", "help"}:
            return self.COMMAND_HELP

        if text.lower() in {"/memory", "memory"}:
            return self._facts_block()

        if text.lower() in {"/stats", "stats"}:
            stats = self.memory.stats()
            return f"Facts: {stats['facts']} | History: {stats['history']}"

        if text.lower() in {"/clear", "clear"}:
            self.memory.clear_history()
            return "Conversation history cleared."

        if text.lower().startswith("/remember "):
            _, _, payload = text.partition(" ")
            if not payload:
                return "Usage: /remember key=value"
            if "=" not in payload:
                return "Usage: /remember key=value"
            key, value = payload.split("=", 1)
            self.memory.set_fact(key.strip(), value.strip())
            return f"Remembered: {key.strip()} = {value.strip()}"

        user_context = self._context_messages(text)
        if not user_context:
            return "I’m ready to help."

        summary = "\n".join(f"{msg['role']}: {msg['content']}" for msg in user_context)
        return (
            "I can help with that. "
            f"Context so far:\n{summary}\n\n"
            "If you want, ask me for a summary, a direct answer, or a rewrite."
        )

    def handle(self, user_input: str) -> str:
        return self.answer_directly(user_input)

    def answer_and_record(self, user_input: str) -> str:
        answer = self.answer_directly(user_input)
        self.memory.add_history("user", user_input.strip())
        self.memory.add_history("assistant", answer)
        return answer

    def _record(self, user_input: str, answer: str, meta: Optional[dict] = None) -> None:
        self.memory.add_history("user", user_input.strip(), meta)
        self.memory.add_history("assistant", answer, meta)

    def run_command(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        if text.startswith("/"):
            self.answer_and_record(text)
            return True
        return False


app = Flask(__name__)
aura = Aura(memory=SimpleMemory(DB_PATH))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = str(payload.get("message", "")).strip()
    if not user_input:
        return jsonify({"error": "Message is required."}), 400

    response = aura.answer_and_record(user_input)
    return jsonify({"response": response})


def main(db_path: str = DB_PATH) -> None:
    global aura
    aura = Aura(memory=SimpleMemory(db_path))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)


if __name__ == "__main__":
    main()
