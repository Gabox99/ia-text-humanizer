"""Guidance detector interface.

This is the component that was missing, and it is the reason the first version
of this service plateaued. The stylometric scorer in `app/scoring` measures
hand-picked surface features — burstiness, em-dash density, banned vocabulary.
Trained classifiers (GPTZero, Copyleaks, Originality) do not read those
features; they read a learned decision boundary over token distributions. So
optimising the surface proxy can drive it to zero while the classifier stays
pinned at 100%, which is exactly what happened.

A *guidance detector* is a real neural detector we run locally. Two properties
from the literature make it worth the RAM:

* Attribution. It tells us *which sentences* carry the machine signal, so the
  rewrite effort goes where it matters instead of being spread evenly.
* Transferability. Adversarial Paraphrasing (arXiv:2506.07001) shows that
  optimising against one guidance detector evades detectors never seen during
  the attack, because strong detectors converge on a shared notion of "human"
  in order to keep their false-positive rate low. SICO (arXiv:2305.10847)
  reports the same effect for greedy substitution guided by a proxy detector,
  dropping GPTZero AUC from 0.779 to 0.184.

The interface is deliberately tiny so the local transformer implementation, a
future hosted-API implementation, and the null fallback are interchangeable.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class SentenceScore:
    index: int
    text: str
    # Probability the sentence is machine-generated, 0.0-1.0.
    ai_probability: float

    @property
    def suspicious(self) -> bool:
        return self.ai_probability >= 0.5


@dataclass
class DetectorReading:
    """One detector verdict on a piece of text."""

    ai_probability: float
    sentences: list[SentenceScore] = field(default_factory=list)
    model: str = ""
    # True when this reading came from a real neural detector rather than the
    # stylometric fallback. Callers surface this so a caller never mistakes a
    # proxy number for a detector number.
    neural: bool = False

    @property
    def percent(self) -> float:
        return round(self.ai_probability * 100, 2)

    def worst_sentences(self, limit: int) -> list[SentenceScore]:
        """Highest-scoring sentences first — the ones worth rewriting."""
        return sorted(self.sentences, key=lambda s: s.ai_probability, reverse=True)[:limit]


class GuidanceDetector(abc.ABC):
    """A detector used as the optimisation target for the rewrite loop."""

    name: str = "abstract"
    neural: bool = False

    @abc.abstractmethod
    def score_document(self, text: str) -> float:
        """Probability the whole text is machine-generated, 0.0-1.0."""

    @abc.abstractmethod
    def score_sentences(self, sentences: list[str]) -> list[float]:
        """Per-sentence probabilities. Must accept a batch: the attack calls
        this with every candidate rewrite at once, and a per-sentence loop on
        CPU is the difference between seconds and minutes."""

    def read(self, text: str, sentences: list[str] | None = None) -> DetectorReading:
        from app.scoring.metrics import split_sentences, strip_markdown

        if sentences is None:
            sentences = split_sentences(strip_markdown(text))
        probs = self.score_sentences(sentences) if sentences else []
        return DetectorReading(
            ai_probability=self.score_document(text),
            sentences=[
                SentenceScore(index=i, text=s, ai_probability=p)
                for i, (s, p) in enumerate(zip(sentences, probs))
            ],
            model=self.name,
            neural=self.neural,
        )

    def warmup(self) -> None:
        """Optional: pay the model-load cost before the first request."""
        return None
