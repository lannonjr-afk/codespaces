# AURA

## Privacy and access

The web app does not implement user accounts or authentication. Anyone who can
reach its service URL can access the chat API, so keep deployment access
restricted if the app is intended for personal use. Do not rely on a private
GitHub repository to make a deployed web service private.

Messages and assistant replies are stored in SQLite. Avoid entering sensitive
information. History is capped at 2,000 records by default and old records are
pruned as new turns are saved (`MAX_HISTORY_RECORDS` configures the cap). The
`/clear` command removes conversation history. The SQLite database is not
encrypted.

The chat API has a per-process limit of 12 requests per client IP per 60 seconds
by default (`CHAT_RATE_LIMIT` and `CHAT_RATE_WINDOW_SECONDS` configure this).
This is a lightweight safeguard for a small/private deployment, not a robust
shared limiter: it is not synchronized between Gunicorn workers or instances.
The app uses the direct peer IP and does not trust forwarded IP headers.

## Model providers

The app uses Groq when `GROQ_API_KEY` is configured and defaults to
`llama-3.3-70b-versatile` (`GROQ_MODEL` selects a different model). If Groq fails,
it attempts Ollama as a fallback. With no Groq key, it uses Ollama directly;
configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` for that setup. Provider details
are not returned to chat clients when both providers fail.

## Run locally

From the `chat_bot` directory, install requirements and run the web app:

```sh
python -m pip install -r requirements.txt
gunicorn newai:app
```

To use Ollama, ensure it is reachable from the app and configure its URL/model as
needed. To use Groq, configure `GROQ_API_KEY` in the environment. The SQLite
database defaults to `agent_memory.db`; set `DB_PATH` to choose another path.

The Render Blueprint uses a free web-service plan, so SQLite data is on
(ephemeral storage and can be lost when the service restarts or redeploys. For
durable storage, use persistent storage and set `DB_PATH` to a file on that disk.
