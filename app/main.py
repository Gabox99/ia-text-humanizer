"""FastAPI surface.

    POST /humanize   rewrite an article, returns {title, content, metrics, ...}
    POST /analyze    score text without rewriting it (free, no API call)
    GET  /health     liveness probe for EasyPanel
"""

from __future__ import annotations

import asyncio
import logging
import time

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.llm import MissingCredentials, Rewriter
from app.pipeline import Pipeline, to_report
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectRequest,
    DetectResponse,
    DetectSentence,
    HumanizeRequest,
    HumanizeResponse,
)
from app.scoring import metrics

logging.basicConfig(
    level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("humanizer")

app = FastAPI(
    title="IA Text Humanizer",
    version="1.0.0",
    description=(
        "Rewrites machine-generated articles so they read as human-written, "
        "preserving the heading and paragraph structure. Returns title and content "
        "separately, plus the stylometric measurements behind the decision."
    ),
)

# One Rewriter for the process lifetime: it holds the HTTP connection pool.
_rewriter: Rewriter | None = None


def get_pipeline(settings: Settings = Depends(get_settings)) -> Pipeline:
    global _rewriter
    if _rewriter is None:
        _rewriter = Rewriter(settings)
    return Pipeline(settings, _rewriter)


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.app_api_key:
        return  # auth disabled; only sane on a private network
    if x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "model": settings.model,
        "auth_required": bool(settings.app_api_key),
        "anthropic_key_configured": bool(settings.anthropic_api_key),
        "target_ai_score": settings.target_ai_score,
        "strength": settings.strength,
        "guidance_detector": settings.guidance_detector,
        "guidance_model": settings.guidance_model,
        "adversarial_enabled": settings.enable_adversarial,
        # Whether torch is actually importable in this image. If this is false
        # while guidance_detector is "local", the pass silently degrades to the
        # stylometric proxy and will not move trained classifiers.
        "detector_runtime_available": _detector_runtime_available(),
    }


def _detector_runtime_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@app.post("/detect", response_model=DetectResponse, dependencies=[Depends(require_api_key)])
async def detect(
    req: DetectRequest, settings: Settings = Depends(get_settings)
) -> DetectResponse:
    """Score text with the guidance detector — the same signal the adversarial
    pass optimises against.

    Use this to check whether the configured guidance detector agrees with the
    detector that actually gates your publishing. If it reports 0.02 on text
    your detector calls 100% AI, the two disagree and the adversarial pass has
    no useful gradient to follow; pick a different `GUIDANCE_MODEL`.
    """
    from app.detector import get_detector
    from app.html_bridge import html_to_markdown, looks_like_html
    from app.scoring.metrics import split_sentences, strip_markdown

    detector = get_detector(settings)
    # Same normalisation the pipeline applies, so /detect reports what the
    # adversarial pass would actually see. Without it the detector scores angle
    # brackets and the sentence split runs through `</p><p>`.
    text = req.text
    if looks_like_html(text):
        text = html_to_markdown(text).markdown
    plain = strip_markdown(text)
    sentences = split_sentences(plain) if req.sentences else []

    # Detector inference is blocking CPU work; keep the event loop free.
    reading = await asyncio.to_thread(detector.read, text, sentences)

    prob = reading.ai_probability
    verdict = "ai" if prob >= 0.75 else "human" if prob <= 0.25 else "borderline"
    return DetectResponse(
        detector=reading.model,
        neural=reading.neural,
        ai_probability=round(prob, 4),
        ai_percent=reading.percent,
        verdict=verdict,
        words=len(plain.split()),
        sentences=[
            DetectSentence(ai_probability=round(s.ai_probability, 4), text=s.text)
            for s in reading.worst_sentences(req.limit)
        ],
    )


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Score text without rewriting it. Useful as a before/after check."""
    m, s, _ = metrics.analyse(req.text, req.language)
    notes = metrics.suggestions(s, m)
    if m.words < 150:
        # Per-1k-word rates and sentence-length variance are both unstable on a
        # short sample; one flagged word in 30 reads as a density of 33/1k.
        notes.insert(
            0,
            f"Only {m.words} words: the per-1000-word rates and the burstiness figure are "
            "unreliable below ~150 words. Score the full article instead.",
        )
    return AnalyzeResponse(
        language=req.language, metrics=to_report(m, s), suggestions=notes
    )


@app.post("/humanize", response_model=HumanizeResponse, dependencies=[Depends(require_api_key)])
async def humanize(
    req: HumanizeRequest,
    settings: Settings = Depends(get_settings),
    pipeline: Pipeline = Depends(get_pipeline),
) -> HumanizeResponse:
    started = time.monotonic()
    try:
        result = await pipeline.run(req)
    # Error mapping rule: anything the operator must act on (missing key, bad key,
    # empty credit balance, malformed request) is surfaced as a 4xx, because
    # reverse proxies — EasyPanel/Traefik included — replace 5xx (especially
    # 502/503/504) with their own "service unreachable" page and swallow our
    # message. Only a genuine upstream outage or network failure stays 5xx.
    except MissingCredentials as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "ANTHROPIC_API_KEY is not configured on the server. "
                "Check GET /health -> anthropic_key_configured."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except anthropic.AuthenticationError as exc:
        raise HTTPException(
            status_code=400,
            detail="Anthropic rejected the API key. Check the ANTHROPIC_API_KEY env var.",
        ) from exc
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "60")
        raise HTTPException(
            status_code=429,
            detail=f"Anthropic rate limit reached; retry after {retry_after}s",
            headers={"Retry-After": retry_after},
        ) from exc
    except anthropic.BadRequestError as exc:
        # A 400 from Anthropic is a request/account problem, not a gateway fault.
        # The most common one in practice is an empty credit balance.
        message = getattr(exc, "message", None) or str(exc)
        low_balance = (
            "credit balance is too low" in message.lower() or "billing" in message.lower()
        )
        raise HTTPException(
            status_code=402 if low_balance else 400,
            detail=(
                "Anthropic account has no credit balance. Add credits at "
                "console.anthropic.com -> Plans & Billing."
                if low_balance
                else f"Anthropic rejected the request: {message}"
            ),
        ) from exc
    except anthropic.APIStatusError as exc:
        # Genuine upstream 5xx. Rare and transient; retry is the fix.
        raise HTTPException(
            status_code=502, detail=f"Anthropic API error ({exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=504, detail="could not reach the Anthropic API") from exc

    elapsed = time.monotonic() - started
    log.info(
        "humanized %d->%d words in %.1fs | score %.1f -> %.1f | %d calls | $%.4f",
        result.metrics_before.words,
        result.metrics.words,
        elapsed,
        result.metrics_before.ai_score,
        result.metrics.ai_score,
        result.usage.api_calls,
        result.usage.estimated_cost_usd,
    )
    return result


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
