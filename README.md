# RAGEval

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![RAGAS](https://img.shields.io/badge/RAGAS-VibrantLabs-orange)
![DeepEval](https://img.shields.io/badge/DeepEval-0.21-7c3aed)
![Railway](https://img.shields.io/badge/Deployed-Railway-blueviolet?logo=railway&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Evaluate any RAG pipeline in minutes.** Upload pre-generated answers, pick a judge model, and get structured scores across faithfulness, hallucination, relevancy, and more.

No answer generation happens inside this app — your API key is used only as a judge. It never gets stored.

---

## Try it live

**[https://llmevaluation.up.railway.app](https://llmevaluation.up.railway.app)**

No signup. No install. Bring a free API key from Gemini, Groq, or OpenRouter and run an evaluation in under a minute. A built-in sample dataset is included if you don't have data ready.

---

![Home](screenshots/Home.png)

---

## What it does

You bring the outputs from your RAG system (questions + answers + retrieved contexts + ground truths). RAGEval runs them through **RAGAS** and **DeepEval** side-by-side and gives you a per-metric, per-sample breakdown with charts, radar plots, and exportable CSV.

---

## Screenshots

| Evaluate setup | Dataset input |
|---|---|
| ![Evaluate](screenshots/Evalution.png) | ![Dataset](screenshots/Dataset.png) |

| Results overview | Metric breakdown |
|---|---|
| ![Results](screenshots/Result.png) | ![Metrics](screenshots/Result2.png) |

**Per-sample breakdown with expandable DeepEval reasons:**

![Per-sample](screenshots/Result3.png)

---

## Metrics

| Metric | Framework | What it measures |
|--------|-----------|-----------------|
| Faithfulness | RAGAS | Is the answer grounded in the retrieved context? |
| Answer Relevancy | RAGAS | Does the answer actually address the question? |
| Context Precision | RAGAS | Is the retrieved context relevant to the question? |
| Context Recall | RAGAS | Does the context cover what the ground truth needs? |
| Hallucination | DeepEval | Does the answer contain fabricated facts? |
| Contextual Relevancy | DeepEval | Is the context relevant to the query? |
| Bias | DeepEval | Does the answer carry demographic or political bias? |
| Toxicity | DeepEval | Is the answer toxic or harmful? |

---

## Supported judge providers

| Provider | Get a free key | Key format | Models |
|----------|---------------|-----------|--------|
| **Gemini** | [aistudio.google.com](https://aistudio.google.com) | `AIza…` | gemini-3.5-flash, gemini-3.1-pro, and more |
| **Groq** | [console.groq.com](https://console.groq.com) | `gsk_…` | Fetched live from Groq API |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai) | `sk-or-v1-…` | Free-tier models fetched live |

All three have a free tier — no credit card required to try.

---

## Quick start

```bash
git clone https://github.com/kishan5822/LLM_Evaluation.git
cd LLM_Evaluation

python -m venv venv
venv\Scripts\pip install -r requirements.txt        # Windows
# source venv/bin/activate && pip install -r requirements.txt  # Mac/Linux

venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — no `.env` file, no config. Enter your API key in the UI.

---

## Dataset format

Each row needs 4 fields:

```json
[
  {
    "question":     "What is retrieval-augmented generation?",
    "answer":       "Output from your RAG system...",
    "contexts":     ["Retrieved chunk 1", "Retrieved chunk 2"],
    "ground_truth": "The correct expected answer"
  }
]
```

Upload as JSON, enter manually, or click **Try sample dataset** in the UI to load a working example instantly.

`ground_truth` is **optional** — omit it and RAGEval runs in reference-free mode (faithfulness + answer relevancy only; context precision/recall are skipped since they need a reference).

---

## How to use

1. **Get a free API key** — Gemini (aistudio.google.com), Groq (console.groq.com), or OpenRouter (openrouter.ai)
2. **Prepare your dataset** — JSON array with `question`, `answer`, `contexts` per row, `ground_truth` optional (or use the built-in sample)
3. **Pick metrics** — select any combination of RAGAS and DeepEval metrics; optionally enable **Confidence mode** to run DeepEval 3× and get a ± stability band
4. **Run & read results** — aggregate scores, bar chart, radar chart, per-sample table with a failure-category badge per row (retriever vs LLM vs context) and expandable DeepEval reasoning

---

## Integrate with your RAG app

Score a single RAG output directly from your application via the REST API:

```python
import requests

r = requests.post("https://llmevaluation.up.railway.app/api/evaluate/single", json={
    "question": "What is RAG?",
    "answer": "RAG combines retrieval with generation...",
    "contexts": ["Retrieved chunk 1", "Retrieved chunk 2"],
    "ground_truth": "expected answer",   # optional
    "provider": "groq",                  # gemini | groq | openrouter
    "api_key": "gsk_...",
    "model": "llama-3.3-70b-versatile",
    "frameworks": ["ragas", "deepeval"],
})
print(r.json())   # { scores: {...}, diagnosis: {...}, reference_free: false }
```

---

## Features

- **8 metrics** across RAGAS + DeepEval in one run
- **Provider-agnostic judge** — Gemini, Groq, or OpenRouter (no OpenAI dependency)
- **Failure categorization** — each sample tagged with probable root cause (retriever failure, LLM hallucination, insufficient context, off-topic answer)
- **Confidence mode** — runs DeepEval 3× per metric, reports mean ± std, flags unstable scores
- **Reference-free mode** — works on production logs with no ground truth
- **Retry with backoff** on transient judge-LLM 429/5xx
- **CSV + JSON export**, plus a `/api/evaluate/single` REST endpoint for live integration

---

## Architecture

```
Your RAG system outputs (JSON)
         ↓
   RAGEval Dashboard
         ↓
  ┌──────────────┐   ┌──────────────┐
  │    RAGAS     │   │   DeepEval   │
  │  4 metrics   │   │  4 metrics   │
  └──────────────┘   └──────────────┘
         ↓
  FastAPI → Alpine.js SPA
  Radar · Bar chart · Per-sample breakdown · CSV export
```

Single FastAPI process — serves both the API and the frontend. No build step, no Node.js.

---

## Deploy

Deployed on Railway with auto-deploy from GitHub. Every push to `main` triggers a redeploy.

To self-host:

```bash
docker build -t rageval .
docker run -p 8000:8000 rageval
```

Or run directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Tech stack

- **Backend** — Python, FastAPI, RAGAS (VibrantLabs fork), DeepEval
- **Frontend** — Alpine.js, Tailwind CSS, Plotly.js (all CDN, no build step)
- **Judge models** — OpenAI-compatible endpoints (Gemini, Groq, OpenRouter)
- **Hosting** — Railway (auto-deploy from GitHub)

---

## Author

**Kishan Raj** — AI/GenAI Engineer  
[GitHub](https://github.com/kishan5822) · [LinkedIn](https://www.linkedin.com/in/kishan-raj-bb5b82220/)
