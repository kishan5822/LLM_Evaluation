# RAGEval — LLM Evaluation Dashboard

> Evaluate external RAG pipeline outputs using RAGAS + DeepEval. Bring your answers, we score them.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![RAGAS](https://img.shields.io/badge/RAGAS-0.1.9-blueviolet)
![DeepEval](https://img.shields.io/badge/DeepEval-0.21-orange)

## What it does

Upload answers from any external RAG system along with the questions, retrieved contexts, and ground truths. The dashboard evaluates them using RAGAS and DeepEval with your chosen judge model (OpenAI, Gemini, or Grok).

**No answer generation happens inside this app.** The API key is used only for the judge model.

## Metrics

| Metric | Framework | Description |
|--------|-----------|-------------|
| Faithfulness | RAGAS + DeepEval | Is the answer grounded in the retrieved context? |
| Answer Relevancy | RAGAS + DeepEval | Does the answer address the question? |
| Context Precision | RAGAS | Is the retrieved context relevant? |
| Context Recall | RAGAS | Does the context cover the ground truth? |
| Hallucination | DeepEval | Does the answer contain fabricated facts? |

## Setup

```bash
git clone https://github.com/kishansraj/llm-eval-dashboard
cd llm-eval-dashboard
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://localhost:8000** — no `.env` file needed. Enter your API key in the UI.

## Dataset Format

```json
[
  {
    "question":     "What is RAG?",
    "answer":       "Your RAG system's answer here...",
    "contexts":     ["Retrieved chunk 1", "Retrieved chunk 2"],
    "ground_truth": "The correct expected answer"
  }
]
```

## Architecture

```
Your External RAG System (answers)
        ↓
  Upload JSON / Manual Entry
        ↓
  Judge Model (OpenAI / Gemini / Grok)
        ↓
┌─────────────────────┐  ┌─────────────────────┐
│  RAGAS              │  │  DeepEval           │
│  • Faithfulness     │  │  • Hallucination    │
│  • Ans. Relevancy   │  │  • Ans. Relevancy   │
│  • Ctx. Precision   │  │  • Faithfulness     │
│  • Ctx. Recall      │  └─────────────────────┘
└─────────────────────┘
        ↓
  FastAPI + Modern SPA Dashboard
  (Radar · Heatmap · Table · CSV Export)
```

## Deploy

Single Python service — deploy on **Railway** or **Render** (free tier).

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Author

**Kishan Raj** — AI/GenAI Engineer  
[LinkedIn](https://linkedin.com/in/kishan) · [GitHub](https://github.com/kishansraj)
