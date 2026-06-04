from typing import Dict, Optional


def _val(scores: dict, *keys) -> Optional[float]:
    """First non-None numeric value among keys."""
    for k in keys:
        v = scores.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def categorize_failure(scores: Dict) -> Dict:
    """Map a per-sample metric pattern to a probable root cause.

    `scores` is a merged dict of RAGAS + DeepEval scores for one sample, e.g.
    {faithfulness, answer_relevancy, llm_context_precision_with_reference,
     context_recall, deepeval_hallucination, ...}

    Returns {category, label, color, explanation, recommendation}.
    Order of checks matters — most specific / upstream cause first.
    """
    faithfulness   = _val(scores, "faithfulness")
    ctx_precision  = _val(scores, "context_precision", "llm_context_precision_with_reference")
    ctx_recall     = _val(scores, "context_recall")
    answer_rel     = _val(scores, "answer_relevancy")
    hallucination  = _val(scores, "deepeval_hallucination")

    if ctx_precision is not None and ctx_precision < 0.4:
        return {
            "category": "retriever_failure",
            "label": "Retriever failure",
            "color": "#dc2626",
            "explanation": "Retrieved chunks are not relevant to the question.",
            "recommendation": "Improve the embedding model or retrieval strategy (BM25, hybrid search).",
        }

    if faithfulness is not None and faithfulness < 0.4 and (ctx_precision or 0) > 0.6:
        return {
            "category": "llm_hallucination",
            "label": "LLM hallucination",
            "color": "#ea580c",
            "explanation": "Context was relevant but the answer makes unsupported claims.",
            "recommendation": "Use a stronger judge model, tighten the system prompt, or lower temperature.",
        }

    if hallucination is not None and hallucination > 0.6:
        return {
            "category": "llm_hallucination",
            "label": "LLM hallucination",
            "color": "#ea580c",
            "explanation": "DeepEval flagged fabricated content not grounded in the context.",
            "recommendation": "Constrain generation to the retrieved context; reduce temperature.",
        }

    if ctx_recall is not None and ctx_recall < 0.4:
        return {
            "category": "context_insufficient",
            "label": "Context insufficient",
            "color": "#d97706",
            "explanation": "Retrieved context lacks the information needed to answer.",
            "recommendation": "Increase top-k retrieval, improve chunking, or expand the knowledge base.",
        }

    if answer_rel is not None and answer_rel < 0.4:
        return {
            "category": "question_ambiguity",
            "label": "Answer off-topic",
            "color": "#2563eb",
            "explanation": "The answer does not directly address the question.",
            "recommendation": "Review question phrasing or add a query-rewriting step.",
        }

    return {
        "category": "passing",
        "label": "Passing",
        "color": "#16a34a",
        "explanation": "All available metrics are within acceptable range.",
        "recommendation": "",
    }
