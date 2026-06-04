import math
import pandas as pd
from typing import List, Dict

_GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GROQ_BASE       = "https://api.groq.com/openai/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _build_llm(gemini_api_key="", groq_api_key="", openrouter_api_key="", openrouter_model=""):
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    if gemini_api_key:
        return LangchainLLMWrapper(ChatOpenAI(
            model="gemini-2.0-flash",
            openai_api_key=gemini_api_key,
            openai_api_base=_GEMINI_BASE,
            temperature=0,
            max_retries=3,
        ))
    if groq_api_key:
        return LangchainLLMWrapper(ChatOpenAI(
            model="llama-3.3-70b-versatile",
            openai_api_key=groq_api_key,
            openai_api_base=_GROQ_BASE,
            temperature=0,
            max_retries=3,
        ))
    if openrouter_api_key:
        return LangchainLLMWrapper(ChatOpenAI(
            model=openrouter_model or "openai/gpt-4o-mini",
            openai_api_key=openrouter_api_key,
            openai_api_base=_OPENROUTER_BASE,
            temperature=0,
            max_retries=3,
            default_headers={"HTTP-Referer": "https://rageval.app", "X-Title": "RAGEval"},
        ))
    raise ValueError("Provide at least one API key.")


def _build_embeddings(gemini_api_key=""):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    if gemini_api_key:
        from langchain_openai import OpenAIEmbeddings
        return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model="text-embedding-004",
            openai_api_key=gemini_api_key,
            openai_api_base=_GEMINI_BASE,
        ))
    # Groq/OpenRouter: fall back to local HuggingFace embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    except Exception:
        return None


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    gemini_api_key: str = "",
    groq_api_key: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "",
) -> pd.DataFrame:
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall

    llm = _build_llm(gemini_api_key, groq_api_key, openrouter_api_key, openrouter_model)
    embeddings = _build_embeddings(gemini_api_key)

    # Reference-free mode: if no sample has a ground truth, context_precision and
    # context_recall (both reference-based) can't be computed — run only the
    # reference-free metrics.
    has_reference = any((gt or "").strip() for gt in ground_truths)

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx,
            reference=gt,
        )
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
    ]
    if has_reference:
        metrics.append(LLMContextPrecisionWithReference())
        metrics.append(LLMContextRecall())

    kwargs = dict(dataset=dataset, metrics=metrics, llm=llm, raise_exceptions=False)
    if embeddings is not None:
        kwargs["embeddings"] = embeddings

    return evaluate(**kwargs).to_pandas()


def compute_ragas_summary(df: pd.DataFrame) -> Dict:
    # Map friendly keys to actual column names produced by this fork
    col_map = {
        "faithfulness":      "faithfulness",
        "answer_relevancy":  "answer_relevancy",
        "context_precision": "llm_context_precision_with_reference",
        "context_recall":    "context_recall",
    }
    out = {}
    for key, col in col_map.items():
        if col not in df.columns:
            continue
        v = df[col].mean()
        out[key] = round(float(v), 4) if not math.isnan(v) else None
    return out
