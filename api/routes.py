import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from evaluators.ragas_eval import run_ragas_evaluation, compute_ragas_summary
from evaluators.deepeval_eval import run_deepeval_evaluation, compute_deepeval_summary

router = APIRouter()

_GROQ_BASE       = "https://api.groq.com/openai/v1"
_GEMINI_BASE     = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class DatasetItem(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


class EvalRequest(BaseModel):
    items: List[DatasetItem]
    gemini_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    use_ragas: bool = True
    use_deepeval: bool = True


def _sanitize(records: list) -> list:
    import math
    out = []
    for row in records:
        clean = {}
        for k, v in row.items():
            try:
                if math.isnan(v):
                    clean[k] = None
                    continue
            except (TypeError, ValueError):
                pass
            clean[k] = v
        out.append(clean)
    return out


def _sanitize_dict(d: dict) -> dict:
    import math
    out = {}
    for k, v in d.items():
        try:
            if math.isnan(v):
                out[k] = None
                continue
        except (TypeError, ValueError):
            pass
        out[k] = v
    return out


def _run_sync(req: EvalRequest) -> dict:
    questions     = [it.question     for it in req.items]
    answers       = [it.answer       for it in req.items]
    contexts      = [it.contexts     for it in req.items]
    ground_truths = [it.ground_truth for it in req.items]

    result = {}

    if req.use_ragas:
        df = run_ragas_evaluation(
            questions, answers, contexts, ground_truths,
            gemini_api_key=req.gemini_api_key,
            groq_api_key=req.groq_api_key,
            openrouter_api_key=req.openrouter_api_key,
            openrouter_model=req.openrouter_model,
        )
        result["ragas"] = _sanitize(df.to_dict("records"))
        result["ragas_summary"] = _sanitize_dict(compute_ragas_summary(df))

    if req.use_deepeval:
        df = run_deepeval_evaluation(
            questions, answers, contexts, ground_truths,
            gemini_api_key=req.gemini_api_key,
            groq_api_key=req.groq_api_key,
            groq_model=req.groq_model,
            openrouter_api_key=req.openrouter_api_key,
            openrouter_model=req.openrouter_model,
        )
        result["deepeval"] = _sanitize(df.to_dict("records"))
        result["deepeval_summary"] = _sanitize_dict(compute_deepeval_summary(df))

    return result


@router.post("/evaluate")
async def evaluate(req: EvalRequest):
    if not req.items:
        raise HTTPException(400, "Dataset is empty.")
    if not (req.gemini_api_key or req.groq_api_key or req.openrouter_api_key):
        raise HTTPException(400, "Provide at least one API key.")
    if not (req.use_ragas or req.use_deepeval):
        raise HTTPException(400, "Select at least one evaluation framework.")
    try:
        return await run_in_threadpool(_run_sync, req)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/models/groq")
async def get_groq_models(api_key: str = Query(...)):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=_GROQ_BASE)
        models = client.models.list()
        ids = sorted([m.id for m in models.data])
        return {"models": ids}
    except Exception as e:
        raise HTTPException(400, f"Could not fetch Groq models: {e}")


@router.get("/models/openrouter")
async def get_openrouter_models(api_key: str = Query(...)):
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://openrouter.ai/api/v1/models/user",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
        # Keep only models where prompt pricing == "0" (free)
        free = [
            m["id"] for m in data
            if str(m.get("pricing", {}).get("prompt", "1")) == "0"
        ]
        ids = sorted(free)
        return {"models": ids}
    except Exception as e:
        raise HTTPException(400, f"Could not fetch OpenRouter models: {e}")


class ValidateRequest(BaseModel):
    provider: str
    api_key: str
    model: str = ""

@router.post("/validate")
async def validate_key(req: ValidateRequest):
    def _check():
        from openai import OpenAI
        if req.provider == "gemini":
            client = OpenAI(api_key=req.api_key, base_url=_GEMINI_BASE)
            model  = "gemini-3.5-flash"
        elif req.provider == "groq":
            client = OpenAI(api_key=req.api_key, base_url=_GROQ_BASE)
            model  = "llama-3.1-8b-instant"
        elif req.provider == "openrouter":
            client = OpenAI(api_key=req.api_key, base_url=_OPENROUTER_BASE)
            model  = req.model or "openai/gpt-4o-mini"
        else:
            raise ValueError("Unknown provider")
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=3
        )
    try:
        await run_in_threadpool(_check)
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)[:120]}


@router.get("/sample")
async def get_sample():
    path = Path("data/sample_dataset.json")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/health")
async def health():
    return {"status": "ok"}
