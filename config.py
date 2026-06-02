JUDGE_MODELS = {
    "gemini":     ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gemini-3-flash-preview"],
    "groq":       [],  # fetched dynamically from https://api.groq.com/openai/v1/models
    "openrouter": [],  # fetched dynamically from https://openrouter.ai/api/v1/models
}

METRIC_THRESHOLDS = {
    # RAGAS
    "faithfulness":            0.7,
    "answer_relevancy":        0.7,
    "context_precision":       0.6,
    "context_recall":          0.6,
    # DeepEval (lower-is-better for hallucination/bias/toxicity)
    "deepeval_hallucination":  0.3,
    "deepeval_contextualrelevancy": 0.7,
    "deepeval_bias":           0.3,
    "deepeval_toxicity":       0.3,
}
