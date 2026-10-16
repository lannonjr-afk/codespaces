# AURA

## Deploy to Render

1. Push this repository to GitHub and create a Render Blueprint from it. Render
   will use the root `render.yaml` to build and start the web service.
2. When prompted, set `GROQ_API_KEY` to a Groq API key.
3. Open the service URL and start chatting.

The web app is public and does not require a login. Anyone with the service URL
can use it and send requests to the chat API.

The service uses Groq's OpenAI-compatible API by default. Set `GROQ_MODEL` in
Render if you want to select a different model available to your account. The
local command-line mode continues to use Ollama when `GROQ_API_KEY` is unset.
The `TAVILY_API_KEY` setting is not currently used by this application.

The Blueprint uses Render's free web-service plan, so the SQLite conversation
history is on ephemeral storage and can be lost when the service restarts or
redeploys. For durable memory, use a paid Render plan with a persistent disk,
mount it at `/var/data`, and set `DB_PATH` to
`/var/data/agent_memory.db`.

## Run locally

Install the requirements and run the existing CLI with Ollama:

```sh
python -m pip install -r requirements.txt
python newai.py
```

To run the web app locally, set `GROQ_API_KEY`, then run:

```sh
gunicorn newai:app
```

The web app can instead use a reachable Ollama server when `GROQ_API_KEY` is
unset. Set `OLLAMA_BASE_URL` and optionally `OLLAMA_MODEL` for that setup.
