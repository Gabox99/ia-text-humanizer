"""Local neural guidance detector, via HuggingFace transformers on CPU.

Model choice matters more than anything else here. A detector trained only on
GPT-2 output (the classic `roberta-base-openai-detector`) barely fires on modern
model prose, so optimising against it teaches the pipeline nothing. The default
below is trained on the RAID benchmark, which covers current models.

Two structural details:

* Long documents are scored in overlapping windows and aggregated, because
  these encoders cap at 512 tokens and a 2000-word article is ~2700.
* Sentence scoring is batched. The attack scores dozens of candidate rewrites
  per round, and a Python loop over single forward passes would dominate the
  wall clock.
"""

from __future__ import annotations

import logging
import threading

from app.detector.base import GuidanceDetector

log = logging.getLogger(__name__)

# Known-good guidance detectors, best first. All permissively licensed.
#   desklib/ai-text-detector-v1.01     DeBERTa-v3-large, 0.4B, MIT, leads RAID.
#                                      Best signal, ~1.6GB RAM in fp32.
#   fakespot-ai/roberta-base-...       RoBERTa-base, 125M, ~500MB. Good tradeoff.
#   openai-community/roberta-base-...  GPT-2 era. Weak on modern prose; last resort.
DEFAULT_MODEL = "desklib/ai-text-detector-v1.01"

_MAX_TOKENS_CAP = 768   # desklib tokenises at 768; encoders cap lower
_WINDOW_STRIDE = 384  # overlap keeps a sentence from being cut at both edges


class LocalNeuralDetector(GuidanceDetector):
    neural = True

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        threads: int | None = None,
    ):
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.threads = threads
        self._model = None
        self._tokenizer = None
        self._head = ""
        self._max_length = 512
        # Model load is lazy but must happen once; concurrent chunk rewrites
        # would otherwise each start their own load.
        self._lock = threading.Lock()

    # --- loading ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoConfig, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - deployment guard
                raise RuntimeError(
                    "The local guidance detector needs torch and transformers. "
                    "Install requirements-detector.txt, or set "
                    "GUIDANCE_DETECTOR=none to run without it."
                ) from exc

            if self.threads:
                torch.set_num_threads(self.threads)

            log.info("Loading guidance detector %s (first call only)", self.model_id)
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            config = AutoConfig.from_pretrained(self.model_id)
            archs = getattr(config, "architectures", None) or []

            # Standard sequence-classification checkpoints load through the Auto
            # class. The strongest detectors on the RAID leaderboard do not: they
            # ship a backbone plus a mean-pooled linear head under a custom class
            # name, which the Auto class cannot instantiate.
            if any(a.endswith("ForSequenceClassification") for a in archs):
                self._model = self._load_sequence_classifier(config)
                self._head = "sequence_classification"
            else:
                self._model = self._load_mean_pool_head(config)
                self._head = "mean_pool_sigmoid"

            self._model.eval()
            self._model.to(self.device)
            self._max_length = min(
                _MAX_TOKENS_CAP, int(getattr(config, "max_position_embeddings", 512) or 512)
            )
            log.info(
                "Guidance detector ready: %s (head=%s, max_len=%d)",
                self.model_id, self._head, self._max_length,
            )

    def _load_sequence_classifier(self, config):
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(self.model_id)

    def _load_mean_pool_head(self, config):
        """Backbone + attention-masked mean pooling + Linear(hidden, 1) + sigmoid.

        Weight layout in these checkpoints: the encoder lives under a `model.`
        prefix and the head is `classifier.weight` / `classifier.bias`.
        """
        import torch
        from torch import nn
        from transformers import AutoModel

        class MeanPoolClassifier(nn.Module):
            def __init__(self, backbone, hidden_size: int):
                super().__init__()
                self.model = backbone
                self.classifier = nn.Linear(hidden_size, 1)

            def forward(self, input_ids, attention_mask):
                hidden = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(pooled)

        backbone = AutoModel.from_config(config)
        model = MeanPoolClassifier(backbone, int(config.hidden_size))

        state = self._fetch_state_dict()
        missing, unexpected = model.load_state_dict(state, strict=False)
        # position_ids-style buffers are routinely absent and harmless; a missing
        # classifier head is not, and would silently produce garbage scores.
        critical = [k for k in missing if k.startswith("classifier.")]
        if critical:
            raise RuntimeError(
                f"{self.model_id}: classifier head weights missing ({critical}); "
                "refusing to score with an untrained head."
            )
        if missing:
            log.debug("guidance detector: %d non-critical missing keys", len(missing))
        if unexpected:
            log.debug("guidance detector: %d unexpected keys ignored", len(unexpected))
        del torch  # only needed for the import side effect above
        return model

    def _fetch_state_dict(self) -> dict:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(self.model_id, "model.safetensors")
            from safetensors.torch import load_file

            return load_file(path)
        except Exception:
            import torch

            path = hf_hub_download(self.model_id, "pytorch_model.bin")
            return torch.load(path, map_location="cpu", weights_only=True)

    def warmup(self) -> None:
        self._ensure_loaded()
        self.score_document("A short warmup sentence to build the graph.")

    # --- scoring ----------------------------------------------------------

    def _ai_index(self) -> int:
        """Which logit column means "machine-generated".

        Never assume index 1: these checkpoints disagree. Read id2label, and
        fall back to 1 only when the labels are uninformative (LABEL_0/LABEL_1).
        """
        cfg = getattr(self._model, "config", None) or self._model.model.config
        labels = getattr(cfg, "id2label", None) or {}
        for idx, label in labels.items():
            lowered = str(label).lower()
            if any(k in lowered for k in ("ai", "fake", "machine", "generated", "gpt")):
                return int(idx)
        for idx, label in labels.items():
            if "human" in str(label).lower() or "real" in str(label).lower():
                return 1 - int(idx)
        return 1 if getattr(cfg, "num_labels", 2) > 1 else 0

    def _forward(self, texts: list[str]) -> list[float]:
        self._ensure_loaded()
        torch = self._torch
        out: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self._tokenizer(
                batch,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                if self._head == "mean_pool_sigmoid":
                    logits = self._model(
                        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                    )
                else:
                    logits = self._model(**enc).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
            else:
                probs = torch.softmax(logits, dim=-1)[:, self._ai_index()]
            out.extend(float(p) for p in probs)
        return out

    def score_sentences(self, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        # Very short fragments are noise to a document-trained classifier, and
        # would otherwise dominate the attribution ranking.
        scored = self._forward([s if len(s.split()) >= 4 else s + " " + s for s in sentences])
        return scored

    def score_document(self, text: str) -> float:
        """Windowed score for text longer than the encoder's limit.

        The maximum over windows, not the mean: a detector flags a document on
        its most machine-like passage, so the max is the number that has to come
        down. Averaging would let one very human window mask a bad one.
        """
        self._ensure_loaded()
        from app.scoring.metrics import strip_markdown

        plain = strip_markdown(text).strip()
        if not plain:
            return 0.0

        ids = self._tokenizer(plain, add_special_tokens=False)["input_ids"]
        if len(ids) <= self._max_length - 2:
            return self._forward([plain])[0]

        windows: list[str] = []
        for start in range(0, len(ids), _WINDOW_STRIDE):
            chunk = ids[start : start + self._max_length - 2]
            if len(chunk) < 32 and windows:
                break
            windows.append(self._tokenizer.decode(chunk, skip_special_tokens=True))
        scores = self._forward(windows)
        return max(scores) if scores else 0.0


class StylometricFallbackDetector(GuidanceDetector):
    """Used when the neural detector is disabled.

    Wraps the hand-crafted composite score so the pipeline has one code path.
    It is explicitly `neural = False` — callers must be able to tell a proxy
    number from a detector number, because they mean very different things.
    """

    name = "stylometric-proxy"
    neural = False

    def __init__(self, language: str = "en-US"):
        self.language = language

    def score_document(self, text: str) -> float:
        from app.scoring.metrics import analyse

        _, score, _ = analyse(text, self.language)
        return min(1.0, score.value / 100.0)

    def score_sentences(self, sentences: list[str]) -> list[float]:
        from app.scoring.metrics import analyse

        out = []
        for s in sentences:
            _, score, _ = analyse(s, self.language)
            out.append(min(1.0, score.value / 100.0))
        return out
