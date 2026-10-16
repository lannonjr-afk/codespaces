# AURA

## Deploy to Render

1. Push this repository to GitHub and create a Render Blueprint from it. Render
   will use the root `render.yaml` to build and start the web service.
2. When prompted, set `GROQ_API_KEY` to a Groq API key.
3. Open the service URL and start chatting.

The web app is public and does not require a login. Anyone with the service URL
can use it and send requests to the chat API.

The service uses Groq's OpenAI-compatible API and defaults to
`openai/gpt-oss-120b`. Set `GROQ_MODEL` in Render to choose another model
available to your account. The local command-line mode continues to use Ollama
when `GROQ_API_KEY` is unset.
The `TAVILY_API_KEY` setting is not currently used by this application.

The chat page accepts multiple text and code files, CSV, XLSX, DOCX, text-based
PDF, and PNG/JPEG/WEBP images (up to 5 MB total per request). Files are read for
that request and are not saved. Scanned or password-protected PDFs are not
supported. Extracted text is limited to 100,000 characters per file. Text
attachments longer than 6,000 characters are analyzed in overlapping chunks
and combined into a final answer, using multiple model requests. Image questions
require setting `GROQ_VISION_MODEL` in the Render service environment to a
vision-capable model available to your account; text questions continue using
`GROQ_MODEL`.

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
