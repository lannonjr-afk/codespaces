"""AURA chat assistant with a customizable system prompt and SQLite memory.

The CLI uses local Ollama by default. Set ``GROQ_API_KEY`` to use Groq's
OpenAI-compatible API instead. The Flask app exposes the same assistant over
HTTP for deployment.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request
from openai import OpenAI, OpenAIError

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
else:
    MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DB_PATH = os.getenv("DB_PATH", "agent_memory.db")
ENABLE_ATTACHMENTS = os.getenv("ENABLE_ATTACHMENTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CONTEXT_MESSAGES = 12
MAX_INPUT_CHARS = 200_000
MAX_HISTORY_CHARS = 75_000
MAX_CONTENT_LENGTH_BYTES = 16 * 1024 * 1024
YOUTUBE_BASE_URL = "https://www.youtube.com"
YOUTUBE_PAGE_TIMEOUT_SECONDS = 15
YOUTUBE_PAGE_MAX_BYTES = 5_000_000
YOUTUBE_SUBSCRIBER_QUESTION = re.compile(
    r"\bhow many\s+(?:youtube\s+)?(?:subscribers|subs)\s+(?:does|do)\s+"
    r"(?P<channel>.+?)\s+(?:currently\s+)?have"
    r"(?:\s+currently|\s+right now|\s+today)?"
    r"(?:\s+on\s+youtube)?[?.!]*$",
    re.IGNORECASE,
)
YOUTUBE_SUBSCRIBER_COUNT_QUESTION = re.compile(
    r"\b(?:subscriber|subscribers|sub|subs)\s+count\s+(?:for|of)\s+"
    r"(?P<channel>.+?)[?.!]*$",
    re.IGNORECASE,
)
YOUTUBE_POSSESSIVE_SUBSCRIBER_QUESTION = re.compile(
    r"\bwhat(?:'s| is)\s+(?P<channel>.+?)(?:'s|’s)\s+"
    r"(?:youtube\s+)?(?:subscriber|subscribers|sub|subs)\s+count[?.!]*$",
    re.IGNORECASE,
)


def trim_for_model(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Keep the start and end of long text while trimming the middle.

    This preserves enough context to remain useful without letting a single giant
    message blow past the model's context window.
    """
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    if max_chars <= 32:
        return clean[:max_chars]

    suffix = "\n...[text trimmed for model context]...\n"
    keep = max_chars - len(suffix)
    head = keep // 2
    tail = keep - head
    return f"{clean[:head]}{suffix}{clean[-tail:]}"


class YouTubeLookupError(Exception):
    """An error that can be shown to the user when a live lookup fails."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _fetch_youtube_page(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; AURA/1.0; +https://www.youtube.com)"
            ),
            "Accept-Language": "en",
        },
    )
    try:
        with urlopen(request, timeout=YOUTUBE_PAGE_TIMEOUT_SECONDS) as response:
            page = response.read(YOUTUBE_PAGE_MAX_BYTES + 1)
    except HTTPError as exc:
        raise YouTubeLookupError(
            f"YouTube blocked or rejected the lookup (HTTP {exc.code}). "
            "Try again later or provide the channel's exact handle."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise YouTubeLookupError(
            "Could not reach YouTube to check the current subscriber count. "
            "Check the network and try again."
        ) from exc
    if len(page) > YOUTUBE_PAGE_MAX_BYTES:
        raise YouTubeLookupError("YouTube's page was too large to process.")
    return page.decode("utf-8", errors="replace")


def _youtube_channel_url(reference: str) -> str | None:
    reference = reference.strip()
    if reference.startswith("@"):
        handle = quote(reference[1:], safe="._-")
        return f"{YOUTUBE_BASE_URL}/@{handle}"
    if re.fullmatch(r"UC[\w-]{22}", reference):
        return f"{YOUTUBE_BASE_URL}/channel/{reference}"

    candidate_url = reference if "://" in reference else ""
    if not candidate_url and reference.lower().startswith("youtube.com/"):
        candidate_url = f"{YOUTUBE_BASE_URL}/{reference[len('youtube.com/') :]}"
    if not candidate_url:
        return None

    parsed = urlsplit(candidate_url)
    if parsed.hostname not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:
        return None
    if not parsed.path or parsed.path == "/":
        return None
    return f"{YOUTUBE_BASE_URL}{parsed.path}"


def _youtube_initial_data(page: str) -> dict[str, Any]:
    marker = re.search(r"ytInitialData\s*=\s*", page)
    if not marker:
        raise YouTubeLookupError(
            "YouTube did not provide page data for this lookup. "
            "It may be blocking automated requests."
        )
    try:
        data, _ = json.JSONDecoder().raw_decode(page[marker.end() :])
    except json.JSONDecodeError as exc:
        raise YouTubeLookupError("YouTube returned unreadable page data.") from exc
    if not isinstance(data, dict):
        raise YouTubeLookupError("YouTube returned invalid page data.")
    return data


def _youtube_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    simple_text = value.get("simpleText")
    if isinstance(simple_text, str):
        return simple_text
    runs = value.get("runs")
    if isinstance(runs, list):
        text = "".join(
            run["text"]
            for run in runs
            if isinstance(run, dict) and isinstance(run.get("text"), str)
        )
        if text:
            return text
    return None


def _youtube_channel_candidates(data: Any) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            renderer = value.get("channelRenderer")
            if isinstance(renderer, dict):
                channel_id = renderer.get("channelId")
                title = _youtube_text(renderer.get("title"))
                if (
                    isinstance(channel_id, str)
                    and title
                    and channel_id not in seen_ids
                ):
                    candidates.append((title, channel_id))
                    seen_ids.add(channel_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    return candidates


def _find_channel_title(data: dict[str, Any]) -> str | None:
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        channel_metadata = metadata.get("channelMetadataRenderer")
        if isinstance(channel_metadata, dict):
            title = channel_metadata.get("title")
            if isinstance(title, str):
                return title
    return None


def _find_channel_subscriber_count(data: dict[str, Any]) -> str | None:
    try:
        header = data["header"]["pageHeaderRenderer"]["content"]
        view_model = header["pageHeaderViewModel"]
        metadata = view_model["metadata"]["contentMetadataViewModel"]
        rows = metadata["metadataRows"]
    except (KeyError, TypeError):
        return None
    if not isinstance(rows, list):
        return None

    for row in rows:
        parts = row.get("metadataParts", []) if isinstance(row, dict) else []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            count_text = text.get("content") if isinstance(text, dict) else None
            accessibility = part.get("accessibilityLabel")
            if (
                isinstance(count_text, str)
                and isinstance(accessibility, str)
                and "subscriber" in accessibility.casefold()
            ):
                match = re.search(
                    r"([\d,.]+\s*[KMB]?)\s+subscribers?",
                    count_text,
                    re.IGNORECASE,
                )
                if match:
                    return match.group(1).strip()
    return None


def _resolve_youtube_channel(reference: str) -> tuple[str, str] | str:
    search_url = (
        f"{YOUTUBE_BASE_URL}/results?"
        f"{urlencode({'search_query': reference, 'sp': 'EgIQAg=='})}"
    )
    data = _youtube_initial_data(_fetch_youtube_page(search_url))
    candidates = _youtube_channel_candidates(data)
    exact_name_matches = [
        (title, channel_id)
        for title, channel_id in candidates
        if title.strip().casefold() == reference.strip().casefold()
    ]
    if len(exact_name_matches) == 1:
        return exact_name_matches[0]
    normalized_reference = re.sub(r"\W+", "", reference).casefold()
    normalized_matches = [
        (title, channel_id)
        for title, channel_id in candidates
        if re.sub(r"\W+", "", title).casefold() == normalized_reference
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if not candidates:
        raise YouTubeLookupError(
            f"I couldn't find a YouTube channel matching {reference!r}. "
            "Try its exact @handle or channel URL."
        )

    options = "\n".join(
        f"- {title} — {YOUTUBE_BASE_URL}/channel/{channel_id}"
        for title, channel_id in candidates[:5]
    )
    return (
        f"I found possible YouTube channels for {reference!r}, but couldn't "
        "reliably identify the right one. Reply with the exact @handle or "
        f"channel URL:\n{options}"
    )


def lookup_youtube_subscribers(reference: str) -> str:
    """Read the public subscriber count from a YouTube channel page."""
    channel_url = _youtube_channel_url(reference)
    if channel_url:
        source = channel_url
        title = reference
    else:
        resolved = _resolve_youtube_channel(reference)
        if isinstance(resolved, str):
            return resolved
        title, channel_id = resolved
        channel_url = f"{YOUTUBE_BASE_URL}/channel/{channel_id}"
        source = channel_url

    data = _youtube_initial_data(_fetch_youtube_page(channel_url))
    title = _find_channel_title(data) or title
    metadata = data.get("metadata")
    channel_metadata = (
        metadata.get("channelMetadataRenderer", {})
        if isinstance(metadata, dict)
        else {}
    )
    if isinstance(channel_metadata, dict):
        channel_id = channel_metadata.get("externalId")
        if isinstance(channel_id, str):
            source = f"{YOUTUBE_BASE_URL}/channel/{channel_id}"
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subscriber_count = _find_channel_subscriber_count(data)
    if subscriber_count is None:
        return (
            f"YouTube's public page doesn't show a subscriber count for "
            f"{title}; it may be hidden. Source: {source} "
            f"(checked {checked_at})."
        )
    return (
        f"YouTube currently displays about {subscriber_count} subscribers for "
        f"{title}. "
        f"Source: {source} (checked {checked_at})."
    )

# =========================================================
# 1. PERSISTENT MEMORY (SQLite)
# =========================================================


class SimpleMemory:
    """Key/value facts plus a durable conversation log.

    Facts are the durable "settings" (name, project, preferences);
    history is the raw turn-by-turn session log that doubles as the
    model's conversation context.
    """

    def __init__(self, db_path: str = "agent_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                " key TEXT PRIMARY KEY,"
                " value TEXT"
                ")"
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS history ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " role TEXT NOT NULL,"
                " content TEXT,"
                " meta TEXT,"
                " created_at TEXT DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
        # Keep databases created by the earlier, schema-less version working.
        self._migrate()

    def _migrate(self) -> None:
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(history)")}
        with self.conn:
            if "meta" not in cols:
                self.conn.execute("ALTER TABLE history ADD COLUMN meta TEXT")
            if "created_at" not in cols:
                self.conn.execute("ALTER TABLE history ADD COLUMN created_at TEXT")

    # -----------------------------------------------------
    # Facts
    # -----------------------------------------------------

    def set_fact(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO facts (key, value) VALUES (?, ?)", (key, value)
            )

    def get_fact(self, key: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM facts WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        cursor = self.conn.execute("SELECT key, value FROM facts ORDER BY key")
        return {row["key"]: row["value"] for row in cursor}

    def delete_fact(self, key: str) -> bool:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        return cursor.rowcount > 0

    # -----------------------------------------------------
    # History
    # -----------------------------------------------------

    def add_history(self, role: str, content: str, meta: Optional[dict] = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO history (role, content, meta) VALUES (?, ?, ?)",
                (role, content, json.dumps(meta) if meta is not None else None),
            )

    def recent_records(self, limit: int = 10) -> list[dict[str, Any]]:
        """Newest `limit` turns, oldest first, including stored metadata."""
        cursor = self.conn.execute(
            "SELECT role, content, meta, created_at FROM history"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        records = []
        for row in reversed(cursor.fetchall()):
            meta = None
            if row["meta"]:
                try:
                    meta = json.loads(row["meta"])
                except (TypeError, ValueError):
                    meta = None
            records.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "meta": meta,
                    "created_at": row["created_at"],
                }
            )
        return records

    def recent_history(self, limit: int = 10) -> list[dict[str, str]]:
        """Context-ready `{"role", "content"}` pairs for the model."""
        return [
            {"role": r["role"], "content": r["content"]}
            for r in self.recent_records(limit)
            if r["content"]
        ]

    def clear_history(self) -> int:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM history")
        return cursor.rowcount

    def stats(self) -> dict[str, int]:
        turns = self.conn.execute("SELECT COUNT(*) AS n FROM history").fetchone()["n"]
        facts = self.conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        return {"turns": turns, "facts": facts}

    def close(self) -> None:
        self.conn.close()


# =========================================================
# 2. UNIVERSAL DOMAIN-AGNOSTIC JSON SCHEMAS
# =========================================================

PROCEDURAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "procedural_guide_response",
        "schema": {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "prerequisites_or_components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity_or_level": {"type": "string"},
                            "source_or_method": {"type": "string"},
                        },
                        "required": ["name", "quantity_or_level", "source_or_method"],
                        "additionalProperties": False,
                    },
                },
                "steps_or_modifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_name": {"type": "string"},
                            "details": {"type": "string"},
                        },
                        "required": ["step_name", "details"],
                        "additionalProperties": False,
                    },
                },
                "final_optimizations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "overview",
                "prerequisites_or_components",
                "steps_or_modifications",
                "final_optimizations",
            ],
            "additionalProperties": False,
        },
    },
}

TECHNICAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "technical_troubleshooting_response",
        "schema": {
            "type": "object",
            "properties": {
                "issue_summary": {"type": "string"},
                "root_cause": {"type": "string"},
                "solution": {"type": "string"},
                "prevention_best_practices": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "issue_summary",
                "root_cause",
                "solution",
                "prevention_best_practices",
            ],
            "additionalProperties": False,
        },
    },
}

COMPARISON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "comparative_analysis_response",
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "spec_matrix": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string"},
                            "option_a": {"type": "string"},
                            "option_b": {"type": "string"},
                        },
                        "required": ["feature", "option_a", "option_b"],
                        "additionalProperties": False,
                    },
                },
                "key_takeaways": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["verdict", "spec_matrix", "key_takeaways"],
            "additionalProperties": False,
        },
    },
}

INFORMATIONAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "informational_overview_response",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_details": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary", "key_details"],
            "additionalProperties": False,
        },
    },
}

SCHEMA_BY_INTENT = {
    "procedural": PROCEDURAL_SCHEMA,
    "technical": TECHNICAL_SCHEMA,
    "comparison": COMPARISON_SCHEMA,
    "informational": INFORMATIONAL_SCHEMA,
}

# ==================================
# REACHED-ANSWER PATH
# ==================================

RESEARCH_SYSTEM_PROMPT = """You are AURA, a capable research assistant with a confident voice, a little attitude, and an unusually stubborn commitment to accuracy. You are not a search-result parrot, [...]"""

# =========================================================
# 3. LOCAL OLLAMA CHAT
# =========================================================


class Aura:
    """Chat with a customizable system prompt and persistent memory."""

    COMMAND_HELP = (
        "Commands:\n"
        "  /facts                  list everything AURA has remembered\n"
        "  /remember <key> = <val> store a fact that persists across sessions\n"
        "  /forget <key>           delete a stored fact\n"
        "  /clear                  wipe the conversation history (facts are kept)\n"
        "  /help                   show this help\n"
        "  exit                    quit"
    )

    def __init__(self, memory: Optional[SimpleMemory] = None):
        if GROQ_API_KEY:
            self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
        else:
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.memory = memory or SimpleMemory()

    # -----------------------------------------------------
    # Model plumbing
    # -----------------------------------------------------

    def _complete(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model or MODEL, messages=messages, **kwargs
        )
        return (completion.choices[0].message.content or "").strip()

    def _facts_block(self) -> str:
        """Remembered facts, injected into every system prompt."""
        facts = self.memory.all_facts()
        if not facts:
            return ""
        lines = "\n".join(f"- {key}: {value}" for key, value in facts.items())
        return f"\n\nKNOWN FACTS (remembered from earlier sessions):\n{lines}"

    @staticmethod
    def _sized_history(
        history: list[dict[str, str]],
        max_chars: int = MAX_HISTORY_CHARS,
    ) -> list[dict[str, str]]:
        """Trim older history so the model gets the most recent relevant context."""
        compact: list[dict[str, str]] = []
        used = 0
        for item in reversed(history):
            payload = f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            if used + len(payload) > max_chars and compact:
                break
            compact.append(item)
            used += len(payload)
        return list(reversed(compact))

    def _prepare_user_input(self, user_input: str, max_chars: int = MAX_INPUT_CHARS) -> str:
        """Prune giant input without losing the useful beginning/end of it."""
        clean = user_input.strip()
        if len(clean) <= max_chars:
            return clean
        trimmed = trim_for_model(clean, max_chars=max_chars)
        return trimmed + "\n\n[Input was trimmed to fit the model context window.]"

    def _context_messages(
        self,
        system_prompt: str,
        user_input: str,
        history_limit: int = CONTEXT_MESSAGES,
    ) -> list[dict[str, Any]]:
        history = self.memory.recent_history(history_limit)
        history = self._sized_history(history)
        messages = [{"role": "system", "content": system_prompt + self._facts_block()}]
        messages.extend(history)
        messages.append({"role": "user", "content": self._prepare_user_input(user_input)})
        return messages

    @staticmethod
    def format_response(intent: str, parsed_data: dict) -> str:
        """Render structured data as the human-readable answer."""
        lines: list[str] = []

        if intent == "procedural":
            lines.append(f"Overview: {parsed_data.get('overview', '')}\n")
            lines.append("Prerequisites & Required Components:")
            for comp in parsed_data.get("prerequisites_or_components", []):
                lines.append(
                    f" - {comp.get('name')}: {comp.get('quantity_or_level')} "
                    f"(Source/Method: {comp.get('source_or_method')})"
                )
            lines.append("\nSteps & Key Modifications:")
            for step in parsed_data.get("steps_or_modifications", []):
                lines.append(f" - {step.get('step_name')}: {step.get('details')}")
            lines.append("\nFinal Optimizations:")
            for opt in parsed_data.get("final_optimizations", []):
                lines.append(f" - {opt}")

        elif intent == "technical":
            lines.append(f"Issue Summary: {parsed_data.get('issue_summary', '')}\n")
            lines.append(f"Root Cause:\n{parsed_data.get('root_cause', '')}\n")
            lines.append(f"Solution:\n{parsed_data.get('solution', '')}\n")
            lines.append("Best Practices & Prevention:")
            for bp in parsed_data.get("prevention_best_practices", []):
                lines.append(f" - {bp}")

        elif intent == "comparison":
            lines.append(f"Verdict: {parsed_data.get('verdict', '')}\n")
            lines.append("Specification Matrix:")
            for spec in parsed_data.get("spec_matrix", []):
                lines.append(
                    f" - {spec.get('feature')}: Option A = {spec.get('option_a')} "
                    f"| Option B = {spec.get('option_b')}"
                )
            lines.append("\nKey Takeaways:")
            for kt in parsed_data.get("key_takeaways", []):
                lines.append(f" - {kt}")

        else:
            lines.append(f"Summary: {parsed_data.get('summary', '')}\n")
            lines.append("Key Details:")
            for detail in parsed_data.get("key_details", []):
                lines.append(f" - {detail}")

        return "\n".join(lines)

    # -----------------------------------------------------
    # Turn handling
    # -----------------------------------------------------

    def answer_directly(self, user_input: str) -> str:
        """Answer directly, checking public YouTube pages for subscriber counts."""
        for pattern in (
            YOUTUBE_SUBSCRIBER_QUESTION,
            YOUTUBE_SUBSCRIBER_COUNT_QUESTION,
            YOUTUBE_POSSESSIVE_SUBSCRIBER_QUESTION,
        ):
            match = pattern.search(user_input)
            if match:
                return lookup_youtube_subscribers(match.group("channel").strip())

        messages = self._context_messages(SYSTEM_PROMPT, user_input)
        return self._complete(messages)

    def handle(self, user_input: str) -> None:
        """Send one turn to the configured model and persist the exchange."""
        answer = self.answer_directly(user_input)
        print("\n=== Response ===")
        print(f"{answer}\n")
        self._record(user_input, answer)

    def answer_and_record(self, user_input: str) -> str:
        answer = self.answer_directly(user_input)
        self._record(user_input, answer)
        return answer

    def _record(self, user_input: str, answer: str, meta: Optional[dict] = None) -> None:
        """Persist both sides of the turn; the readable answer is the context."""
        self.memory.add_history("user", user_input)
        self.memory.add_history("assistant", answer, meta=meta)

    # -----------------------------------------------------
    # Local commands
    # -----------------------------------------------------

    def run_command(self, text: str) -> bool:
        """Handle a local command. Returns True when the input was consumed."""
        if not text.startswith("/"):
            return False

        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command == "help":
            print("\n" + self.COMMAND_HELP + "\n")

        elif command == "facts":
            facts = self.memory.all_facts()
            print("\n=== Memory ===")
            if facts:
                for key, value in facts.items():
                    print(f" - {key}: {value}")
            else:
                print(" (nothing remembered yet)")
            print()

        elif command == "remember":
            if "=" not in argument:
                print("Usage: /remember <key> = <value>\n")
            else:
                key, value = (part.strip() for part in argument.split("=", 1))
                if not key or not value:
                    print("Usage: /remember <key> = <value>\n")
                else:
                    self.memory.set_fact(key, value)
                    print(f"Remembered: {key} = {value}\n")

        elif command == "forget":
            if not argument:
                print("Usage: /forget <key>\n")
            elif self.memory.delete_fact(argument):
                print(f"Forgot: {argument}\n")
            else:
                print(f"No fact stored under '{argument}'.\n")

        elif command == "clear":
            removed = self.memory.clear_history()
            print(f"Cleared {removed} stored messages.\n")

        else:
            print(f"Unknown command '/{command}'. Type /help for options.\n")

        return True


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_BYTES


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.post("/api/chat")
def chat():
    if request.files and not ENABLE_ATTACHMENTS:
        return (
            jsonify(
                error=(
                    "Attachment uploads are disabled in text-only mode. "
                    "Set ENABLE_ATTACHMENTS=true to re-enable them."
                )
            ),
            400,
        )

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        return jsonify(error="Send a JSON object with a string 'message' field."), 400

    user_input = payload["message"].strip()
    if not user_input:
        return jsonify(error="Message cannot be empty."), 400

    memory = SimpleMemory(DB_PATH)
    try:
        aura = Aura(memory)
        answer = aura.answer_and_record(user_input)
        return jsonify(response=answer)
    except YouTubeLookupError as exc:
        app.logger.warning("YouTube subscriber lookup failed: %s", exc)
        return jsonify(error=str(exc)), exc.status_code
    except OpenAIError:
        app.logger.exception("The configured language model request failed.")
        return (
            jsonify(
                error="The language model request failed. "
                "Check server configuration and try again."
            ),
            502,
        )
    finally:
        memory.close()


# =========================================================
# 4. MAIN INTERACTIVE LOOP
# =========================================================


def main(db_path: str = "agent_memory.db") -> None:
    memory = SimpleMemory(db_path)
    aura = Aura(memory)

    counts = memory.stats()
    print(
        f"AURA {'Groq' if GROQ_API_KEY else 'local Ollama'} assistant "
        f"initialized with {MODEL}. "
        "(Type 'help' for commands, 'exit' to quit)"
    )
    if counts["turns"] or counts["facts"]:
        print(
            f"Resumed memory: {counts['turns']} stored messages, "
            f"{counts['facts']} remembered facts.\n"
        )
    else:
        print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEnding session.")
                break

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Ending session.")
                break
            if aura.run_command(user_input):
                continue

            try:
                aura.handle(user_input)
            except Exception as e:
                print(f"Execution Error: {str(e)}\n")
    finally:
        memory.close()


if __name__ == "__main__":
    main()

# To commit changes: git push origin main
