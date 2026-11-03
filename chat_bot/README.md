# AURA

## Deploy to Render

1. Push this repository to GitHub and create a Render Blueprint from it. Render
   uses the root `render.yaml` to build and start the web service.
2. Set `GROQ_API_KEY` to a Groq API key.
3. Open the service URL and start chatting.

The web app is public and does not require login. Anyone with the service URL can
send requests to the chat API. It applies a per-process limit of 12 chat requests
per client IP per 60 seconds by default (`CHAT_RATE_LIMIT` and
`CHAT_RATE_WINDOW_SECONDS` configure this). This in-process limit is not shared
between Gunicorn workers or instances; use a reverse-proxy or shared rate-limit
store for stronger production abuse controls. Configure trusted proxy handling
before relying on forwarded client IP headers.

Conversation messages and assistant replies are stored in SQLite. Avoid entering
sensitive information. History is capped at 2,000 records by default and older
records are pruned as new entries are added (`MAX_HISTORY_RECORDS` changes the
limit). The `/clear` command clears stored chat history. The SQLite database is
not encrypted.

The service uses Groq's OpenAI-compatible API and defaults to
`llama-3.3-70b-versatile`. Set `GROQ_MODEL` to choose another model. If Groq is
configured but unavailable, AURA attempts the configured Ollama endpoint as a
fallback. Without `GROQ_API_KEY`, it uses Ollama directly; configure
`OLLAMA_BASE_URL` and `OLLAMA_MODEL` as needed.

The service worker caches only the same-origin app shell and static files. It
does not cache chat API responses. The interface displays an offline notice;
sending chat messages still requires a network connection.

## Run locally

Install requirements and run the CLI with Ollama:

```sh
python -m pip install -r requirements.txt
python newai.py
```

To run the web app locally:

```sh
gunicorn newai:app
```

The Render Blueprint uses a free web-service plan, so SQLite data is on
(ephemeral storage and can be lost when the service restarts or redeploys. For
durable storage, use a persistent disk and set `DB_PATH` to a file on that disk.

