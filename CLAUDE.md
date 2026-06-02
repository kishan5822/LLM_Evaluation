# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (use the project venv)
.\venv\Scripts\pip.exe install -r requirements.txt

# Run the server (hot reload for development)
.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000

# Run without reload (production-style)
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000

# Check API health
curl http://localhost:8000/api/health

# Fetch the sample dataset
curl http://localhost:8000/api/sample
```

The app is served at **http://localhost:8000** — FastAPI serves `frontend/index.html` at `/` and all API routes under `/api/`.

## Architecture

This is a **single-service FastAPI app** — the Python backend serves both the REST API and the static frontend HTML file. There is no build step, no Node.js, no separate frontend server.

### Request flow

```
Browser (frontend/index.html)
  └─ POST /api/evaluate  ──►  api/routes.py::_run_sync()
                                  ├─ evaluators/ragas_eval.py::run_ragas_evaluation()
                                  └─ evaluators/deepeval_eval.py::run_deepeval_evaluation()
```

`_run_sync` is blocking (RAGAS/DeepEval calls can take 1–5 minutes). It runs in FastAPI's thread pool via `run_in_threadpool` so it doesn't block the event loop.

### Evaluator design

Both evaluators accept the same three optional API key parameters (`openai_api_key`, `gemini_api_key`, `grok_api_key`) and pick the first non-empty one as the judge.

**RAGAS** (`evaluators/ragas_eval.py`):
- Judge LLM: `ChatOpenAI` with `openai_api_base` overridden for Gemini/Grok (all three providers use the OpenAI SDK via their OpenAI-compatible endpoints)
- Embeddings: `OpenAIEmbeddings` when OpenAI key present, otherwise falls back to local `HuggingFaceEmbeddings(all-MiniLM-L6-v2)` via `sentence-transformers`
- RAGAS import path has a try/except (`ragas.metrics.collections` → `ragas.metrics`) for version compatibility

**DeepEval** (`evaluators/deepeval_eval.py`):
- OpenAI: sets `OPENAI_API_KEY` env var, passes `judge_model=None` (DeepEval uses it natively)
- Gemini/Grok: `_make_judge()` factory wraps the OpenAI SDK client into a `DeepEvalBaseLLM` subclass
- Metrics are instantiated fresh per sample (inside `_make_metrics()`) to avoid state leakage between samples

### Provider base URLs

```python
# Gemini OpenAI-compatible endpoint
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# xAI Grok OpenAI-compatible endpoint
_GROK_BASE = "https://api.x.ai/v1"
```

Both Gemini and Grok use `openai.OpenAI(base_url=...)` — no separate SDKs needed.

### Frontend

`frontend/index.html` is a self-contained SPA using CDN-loaded libraries only (Tailwind CSS, Alpine.js, Plotly.js). No build step. Charts are rendered client-side via Plotly.js after the `/api/evaluate` response arrives.

API key is **never stored** — it lives only in Alpine.js state for the duration of the browser session and is sent per-request to `/api/evaluate`.

### Dataset format

```json
[{
  "question":     "string",
  "answer":       "string  ← output from the RAG system under test",
  "contexts":     ["string", ...],
  "ground_truth": "string"
}]
```

The app evaluates pre-generated answers from external RAG systems. It does **not** generate answers internally — the API key is used only as a judge model for RAGAS/DeepEval scoring.

### Adding a new judge provider

1. Add `base_url` constant in both `evaluators/ragas_eval.py` and `evaluators/deepeval_eval.py`
2. Add an `elif` branch in `_build_ragas_llm()` using `ChatOpenAI(openai_api_key=..., openai_api_base=...)`
3. Add an `elif` branch in `run_deepeval_evaluation()` calling `_make_judge(...)`
4. Add provider entry to `JUDGE_MODELS` in `config.py`
5. Add provider card to `frontend/index.html` `providers` array in the Alpine.js `app()` function

### Deployment

Single Python process on Railway or Render (free tier). No environment variables required — all API keys come through the UI at runtime.
