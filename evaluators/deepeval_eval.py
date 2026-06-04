import os
import pandas as pd

# DeepEval checks for OPENAI_API_KEY at import time even when a custom judge is used.
# Set a dummy value so the SDK doesn't raise a missing-credentials error.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-not-used")
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

_GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GROQ_BASE       = "https://api.groq.com/openai/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _make_judge(api_key: str, base_url: str, model_id: str, judge_name: str, extra_headers: dict = None):
    from openai import OpenAI as _OAI
    from deepeval.models.base_model import DeepEvalBaseLLM
    from evaluators._retry import retry_on_failure

    _client = _OAI(api_key=api_key, base_url=base_url, default_headers=extra_headers or {})
    _model_id = model_id

    @retry_on_failure(max_retries=3, delay=2)
    def _call(prompt: str) -> str:
        resp = _client.chat.completions.create(
            model=_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    class _Impl(DeepEvalBaseLLM):
        def __init__(self_):
            super().__init__(judge_name)

        def load_model(self_):
            return _client

        def generate(self_, prompt: str) -> str:
            return _call(prompt)

        async def a_generate(self_, prompt: str) -> str:
            return self_.generate(prompt)

        def get_model_name(self_) -> str:
            return _model_id

    return _Impl()


def run_deepeval_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    gemini_api_key: str = "",
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    openrouter_api_key: str = "",
    openrouter_model: str = "",
    confidence_runs: int = 1,
) -> pd.DataFrame:
    if gemini_api_key:
        judge_model = _make_judge(gemini_api_key, _GEMINI_BASE, "gemini-3.5-flash", "gemini-judge")
    elif groq_api_key:
        judge_model = _make_judge(groq_api_key, _GROQ_BASE, groq_model, "groq-judge")
    elif openrouter_api_key:
        judge_model = _make_judge(
            openrouter_api_key, _OPENROUTER_BASE,
            openrouter_model or "openai/gpt-4o-mini", "openrouter-judge",
            extra_headers={"HTTP-Referer": "https://rageval.app", "X-Title": "RAGEval"},
        )
    else:
        raise ValueError("Provide at least one API key.")

    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import HallucinationMetric, ContextualRelevancyMetric

    # Bias and Toxicity — graceful fallback if version doesn't have them
    try:
        from deepeval.metrics import BiasMetric
        _has_bias = True
    except ImportError:
        _has_bias = False

    try:
        from deepeval.metrics import ToxicityMetric
        _has_toxicity = True
    except ImportError:
        _has_toxicity = False

    test_cases = [
        LLMTestCase(input=q, actual_output=a, expected_output=gt, context=ctx, retrieval_context=ctx)
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
    ]

    def _make_metrics():
        m = [
            HallucinationMetric(threshold=0.3, model=judge_model, async_mode=False),
            ContextualRelevancyMetric(threshold=0.7, model=judge_model, async_mode=False),
        ]
        if _has_bias:
            m.append(BiasMetric(threshold=0.3, model=judge_model, async_mode=False))
        if _has_toxicity:
            m.append(ToxicityMetric(threshold=0.3, model=judge_model, async_mode=False))
        return m

    runs = max(1, confidence_runs)

    def _score_one(tc, metric):
        name = metric.__class__.__name__.replace("Metric", "").lower()
        if name == "hallucination" and not bool(tc.context):
            return name, None, "No context provided", None
        scores = []
        reason = None
        last_err = None
        for _ in range(runs):
            try:
                metric.measure(tc)
                scores.append(metric.score)
                if reason is None:
                    reason = metric.reason
            except Exception as e:
                last_err = str(e)
        if not scores:
            return name, None, last_err or "No score", None
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 if len(scores) > 1 else None
        return name, round(mean, 4), reason, (round(std, 4) if std is not None else None)

    results = []
    for i, tc in enumerate(test_cases):
        row = {"question": questions[i], "answer": answers[i]}
        metrics = _make_metrics()
        # Run all metrics for this sample in parallel
        with ThreadPoolExecutor(max_workers=len(metrics)) as ex:
            futures = {ex.submit(_score_one, tc, m): m for m in metrics}
            for fut in as_completed(futures):
                name, score, reason, std = fut.result()
                row[f"deepeval_{name}"] = score
                row[f"deepeval_{name}_reason"] = reason
                if std is not None:
                    row[f"deepeval_{name}_std"] = std
        results.append(row)

    return pd.DataFrame(results)


def compute_deepeval_summary(df: pd.DataFrame) -> Dict[str, float]:
    import math
    cols = [c for c in df.columns if c.startswith("deepeval_") and "reason" not in c and not c.endswith("_std")]
    out = {}
    for c in cols:
        if not df[c].notna().any():
            continue
        v = df[c].mean()
        out[c] = round(v, 4) if not math.isnan(v) else None
    return out
