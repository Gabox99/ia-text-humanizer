# ia-text-humanizer

An HTTP service that rewrites machine-generated articles so they read as human-written,
preserving the heading and paragraph structure, and returns `title` and `content` as
separate fields.

Built for ~2000-word articles. EN-US and PT-BR have curated tell dictionaries; any other
language tag works and falls back to the structural rules alone.

```bash
curl -X POST https://your-app.easypanel.host/humanize \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "# My Title\n\nDraft body...", "language": "en-US"}'
```

---

## Read this before you deploy it

AI detectors stack three kinds of signal:

1. **Perplexity** — how predictable each word is to a reference language model.
2. **Burstiness** — how much sentence-level predictability varies across the document.
3. **Trained classifiers** — models fine-tuned on a learned decision boundary over token
   distributions. GPTZero, Copyleaks, Originality.

Rule-based rewriting handles (1) and (2) and stalls on (3). That is not a guess: an
earlier version of this service produced prose that ZeroGPT scored at 26% and **GPTZero
and Copyleaks both scored at 100% AI**. Optimising hand-picked surface features (em-dash
density, banned vocabulary, sentence-length variance) can drive those features to a human
distribution while a trained classifier does not move at all, because it was never
reading those features.

What closes the gap, per the literature, is optimising against a *real detector*:

- **Adversarial Paraphrasing** ([arXiv:2506.07001](https://arxiv.org/abs/2506.07001)) —
  score with a guidance detector, find the sentences carrying the machine signal,
  generate paraphrase candidates, keep whichever most reduces the score. Evasion
  **transfers to detectors never used during the attack**, because strong detectors
  converge on a shared notion of "human" to keep their false-positive rate low.
- **SICO** ([arXiv:2305.10847](https://arxiv.org/abs/2305.10847)) — greedy substitution
  guided by a proxy detector. Reported to drop GPTZero AUC from 0.779 to 0.184.
- **Recursive paraphrasing** ([arXiv:2303.11156](https://arxiv.org/abs/2303.11156)) —
  repetition compounds the effect; gains decay per round.
- **DIPPER** ([arXiv:2303.13408](https://arxiv.org/abs/2303.13408)) — the canonical
  result that paraphrasing evades detectors, dropping DetectGPT from 70.3% to 4.6%.

So this service runs a local neural detector and optimises against it. Measured on a real
document, guidance detector probability went **0.7326 → 0.2347 in one round**, four
sentence substitutions, one extra API call.

**The one thing that decides whether this works for you:** the guidance detector has to
*agree* with the detector that gates your publishing. Verify it, do not assume it:

```bash
python scripts/detector_check.py --file text_your_detector_flagged.md --expect ai
```

Measured on a pipeline output that GPTZero and Copyleaks both called 100% AI:

| Guidance detector | p(AI) | Verdict |
|---|---|---|
| `desklib/ai-text-detector-v1.01` (default) | 0.76 | **agrees** — usable guide |
| `fakespot-ai/roberta-base-ai-text-detection-v1` | 0.003 | disagrees — useless as a guide |

A detector that already believes your text is human offers no gradient to climb. That
is the whole reason `detector_check.py` exists.

Two remaining limits, stated plainly. **The pipeline never invents facts** — it will not
add a statistic or a quote to sound more lived-in, which caps how specific the output can
get from a vague draft. And **no system can guarantee a number on a detector it cannot
query**; treat every score here as evidence, and keep checking the detector that matters
to you.

So: **treat the `ai_score` in the response as a proxy, and validate against whichever
detector actually gates your work.** The `/analyze` endpoint is free and unlimited — use
it to build intuition about what your own drafts score before and after.

The other thing worth stating plainly: the pipeline never invents facts. It will not add
a statistic, a study, or a quote to make prose sound more lived-in. That is a deliberate
constraint, and it caps how "specific" the output can get from a vague draft.

---

## How it works

```
detect format (HTML -> Markdown, unescape literal 
)
  -> mask code blocks + tables            frozen, never sent to the model
  -> parse into blocks, chunk on headings (~700 words)
  -> PASS 1..N (N from `strength`):
       per chunk: model rewrite -> structural validation
                  -> deterministic regex clean-up -> stylometric score
                  -> if above target, retry with the failing signals as instructions
       pass 2+ swaps the cleanup prompt for a texture prompt that adds human grain
  -> ADVERSARIAL PASS (the part that moves trained classifiers):
       score with the local neural guidance detector
       rank sentences by attribution
       generate paraphrase candidates for the worst ones (one API call)
       score every candidate, greedily keep the best per sentence
       verify at document level; roll back the round if it did not improve
  -> restore frozen blocks -> lift the H1 into `title` -> render back to input format
```

Three things make this different from pasting a "make it sound human" prompt into a chat
window.

### The retry is directed, not blind

When a section scores above target, the pipeline does not just ask again. It measures
*which* signals failed — burstiness, em-dash density, opening connectives, lexical
diversity — and turns each into an imperative instruction that goes into the next
attempt. Blind retries wander. Directed retries converge.

### Editing strength varies by section

Uniformly humanized text has its own fingerprint. A document where every paragraph has
exactly one short sentence and no transition survives reads as *processed*, not written.
So each section gets one of four intensity profiles (heavy, moderate, light, surgical),
picked from a hash of the section's own content — deterministic, so runs are
reproducible, but varied across the article. The deterministic clean-up layer works the
same way: hard tells are replaced ~92% of the time, soft tells ~55%, connectives ~80%.
Leaving a residue is the point.

### Structure damage is detected, not hoped against

The model receives real Markdown, because sentence rhythm is a document-level property and
it needs the surrounding context. The cost of that choice is that the model *could* mangle
the structure. So every rewrite is fingerprinted against its original: heading tree, block
count, list item count, link URLs, frozen placeholders, word-count tolerance, and any
`preserve_terms` you passed. A rewrite that fails goes back with an explicit correction;
if every attempt fails, the original section ships unchanged and you get a warning in the
response rather than broken Markdown.

### The adversarial pass is the one that matters

Everything before it optimises a hand-crafted proxy. This stage optimises a real neural
detector, and it has two properties worth knowing.

**It cannot break your structure.** The pass only ever swaps one sentence for another
*inside a paragraph or blockquote*. Headings, list items, tables and code are never
touched. That is a property of the construction, not of a prompt instruction, so unlike a
free rewrite there is no failure mode where the heading tree comes back mangled.

**It rolls back.** Sentence-level gains do not always add up at document level, so each
round is verified against the whole-document score and reverted if it did not improve.
Accepting a worse document because the parts looked better is how these loops drift.

Fact safety is enforced mechanically, not asked for: a candidate is rejected outright if
it changes any number, drops a link, or drifts more than ~2x in length. That matters more
here than in the rewrite passes, because sentence-level paraphrasing of a statistics-heavy
article is exactly where a figure would quietly move.

Cost is modest: one extra API call per round, typically one or two rounds.

### What the score measures

`ai_score` is 0-100, lower is more human, and it is a weighted sum of penalties you can
inspect in `metrics.penalties`:

| Signal | Human reference band | Weight |
|---|---|---|
| Sentence-length CV (burstiness) | 0.45–0.75 | 18 |
| AI-marked vocabulary density | < 1.5 / 1k words | 14 |
| Sentence-opening connectives | < 6% of sentences | 10 |
| Runs of 3+ same-length sentences | rare | 10 |
| Em dashes | ≤ 1 per 300 words | 8 |
| Short sentences (≤ 6 words) | ≥ 1 per 150 words | 8 |
| Lexical diversity (MATTR-100) | > 0.78 | 8 |
| Assistant register phrases | zero | 8 |
| Hapax legomena ratio | > 0.45 | 6 |
| Contractions (EN) | ≥ 6 / 1k words | 6 |
| Semicolons | ≤ 1 / 1k words | 5 |
| Paragraph-length CV | > 0.45 | 5 |
| Bullet-list ratio | < 25% of lines | 5 |
| Nominalization (-tion/-ment/-ity) | < 35 / 1k words | 5 |
| Rule-of-three parallel lists | < 1 / 1k words | 4 |
| Passive voice | < 18% of sentences | 4 |
| Mid-sentence restatement colons | < 1.5 / 1k words | 3 |

For calibration: raw Claude/GPT article prose scores 45–70. Hand-written blog prose
scores 3–12.

---

## API

Auth: send `X-API-Key: <APP_API_KEY>` on every request. If `APP_API_KEY` is empty, auth is
disabled — only do that on a private network.

Interactive docs at `/docs`. Ready-to-paste curl commands and n8n HTTP Request node
settings are in [docs/n8n.md](docs/n8n.md).

### `POST /humanize`

```json
{
  "text": "# Title\n\nBody in Markdown...",
  "title": null,
  "language": "en-US",
  "tone": "skeptical industry analyst, first person",
  "preserve_terms": ["Acme Cloud", "SOC 2"],
  "model": null,
  "effort": null,
  "strength": "standard",
  "passes": null,
  "target_ai_score": 10,
  "max_attempts": 3,
  "rewrite_headings": true,
  "postprocess": null
}
```

Only `text` is required.

| Field | Default | Notes |
|---|---|---|
| `text` | — | Markdown. Headings, lists, blockquotes, tables and fenced code are all recognised. |
| `title` | `null` | If given and not already the H1, it is prepended as an H1 so it gets humanized too, then lifted back out. |
| `language` | `en-US` | `en-*` and `pt-*` get curated dictionaries. Anything else uses structural rules only. |
| `tone` | `null` | Free-form voice note passed to the model. |
| `preserve_terms` | `[]` | Must survive verbatim. A rewrite that drops one is rejected. |
| `model` | env `MODEL` | Override the model for this call. Must be a `claude-*` id. Lets you A/B models without redeploying. |
| `effort` | env `EFFORT` | `low`-`max` for this call. |
| `strength` | env `STRENGTH` | `standard` \| `aggressive` \| `max`. Higher = harder edit; `max` adds a second texture pass. Helps most vs perplexity checkers; modest vs trained classifiers. |
| `passes` | env `PASSES` | 1-3 full passes, overrides the count implied by `strength`. More passes = more cost and fact-drift risk. |
| `format` | `auto` | `auto` \| `markdown` \| `html`. `auto` detects HTML by its block tags and returns the same format it received. |
| `adversarial` | env | `false` skips the detector-guided pass. |
| `adversarial_rounds` | env | 1-8 substitution rounds. |
| `adversarial_target` | env | Stop once the guidance detector reports this probability (0.0-1.0). |
| `target_ai_score` | env | Retry until the section is at or below this. |
| `max_attempts` | env | Attempts per section, 1–6. |
| `rewrite_headings` | `true` | `false` freezes heading lines exactly, for SEO-locked headings. |
| `postprocess` | env | `false` skips the deterministic regex layer. |

Response:

```json
{
  "title": "What Nobody Tells You About Predictive Maintenance",
  "content": "Three years ago I watched a plant...",
  "language": "en-US",
  "metrics":        { "ai_score": 7.4, "verdict": "human-like", "...": "..." },
  "metrics_before": { "ai_score": 58.1, "verdict": "ai-like",   "...": "..." },
  "target_ai_score": 10.0,
  "target_met": true,
  "chunks": [
    {"index": 0, "heading": "...", "attempts": 2, "ai_score_before": 55.0,
     "ai_score_after": 8.1, "accepted_attempt": 1, "intensity": "heavy"}
  ],
  "usage": {"input_tokens": 12000, "output_tokens": 4200, "cache_read_input_tokens": 9000,
            "api_calls": 4, "estimated_cost_usd": 0.1421},
  "model": "claude-opus-5",
  "warnings": []
}
```

**`content` never contains the H1** — the article title comes back in `title`, which is
what a CMS wants. Every other heading stays in `content` at its original level. A document
that starts at `##` has no article title to lift, so nothing is removed and `title` is
derived from the first heading.

**Always check `warnings` and `target_met`.** `warnings` is where you find out that a
section could not be improved, that the model declined, or that a rewrite was rejected and
the original shipped.

### `POST /analyze`

Scores text without rewriting it. No API call, no cost. Use it on your drafts and on the
output.

```json
{ "text": "...", "language": "en-US" }
```

Returns `metrics` plus `suggestions`, a plain-language list of what is driving the score.

### `GET /health`

Liveness probe. Reports the model, whether auth is on, and whether the Anthropic key is
configured.

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your ANTHROPIC_API_KEY in it
uvicorn app.main:app --reload
```

Run the offline test suite (no API key needed — the model is stubbed):

```bash
python tests/test_pipeline.py && python tests/test_postprocess.py
```

Run a real end-to-end check with a built-in 2000-word sample article:

```bash
python scripts/live_check.py
```

That prints before/after scores, the per-section trace, and the cost, and writes
`sample.humanized.md` for you to paste into your detector. Point it at your own file with
`python scripts/live_check.py article.md --lang pt-BR`.

---

## Deploying to EasyPanel

The repo ships a `Dockerfile`, so EasyPanel needs no build configuration.

1. **In EasyPanel:** create a new **App** in your project.
2. **Source:** connect GitHub, pick this repository and the `main` branch. Enable
   *Auto Deploy* so every push redeploys.
3. **Build:** select **Dockerfile**. Leave the path as `Dockerfile`.
4. **Environment:** paste the variables from `.env.example` and fill in real values. At
   minimum:

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   APP_API_KEY=<a long random string>
   ```

5. **Port:** `8000`. The container also honours a `PORT` variable if you prefer another.
6. **Domain:** add your domain and enable HTTPS.
7. **Health check** (optional but recommended): path `/health`.

Deploy, then verify:

```bash
curl https://your-app.easypanel.host/health
```

### Resources — this changed with the guidance detector

The detector runs a 0.4B-parameter model in-process, which is a real commitment. Two
deployment shapes, chosen with a Docker build arg:

| | `WITH_DETECTOR=true` (default) | `WITH_DETECTOR=false` |
|---|---|---|
| Image | ~1.4 GB (~3 GB with the model baked in) | ~250 MB |
| RAM | **~2 GB resident** | ~512 MB |
| vCPU | 1-2 | 0.5 |
| Detector scoring | ~1.5-2.5 s per pass | n/a |
| Moves GPTZero / Copyleaks | yes, this is the point | **no** |

**A 1 GB droplet will OOM with the detector enabled.** Size for 2 GB minimum; 4 GB if the
droplet also runs EasyPanel and other apps, which it usually does.

To build the lean image, set the build arg in EasyPanel (Build → Build args):

```
WITH_DETECTOR=false
```

and set `GUIDANCE_DETECTOR=none`. You keep the rewrite passes, the structural guarantees
and the HTML round-trip, and you lose the only stage that moves trained classifiers.

If RAM is tight but you want a neural guide, a roberta-base-class model needs ~900 MB
instead of ~2 GB — but verify it agrees with your detector first, because the small ones
frequently do not:

```
GUIDANCE_MODEL=fakespot-ai/roberta-base-ai-text-detection-v1
```

Set `GUIDANCE_WARMUP=true` so the ~40 s weight load happens at boot rather than being
paid by whoever makes the first request. Give the EasyPanel health check a start period
of at least 90 s to match.

Timeouts are the thing to watch. A 2000-word article at `MAX_ATTEMPTS=3` can take 60–180
seconds. If EasyPanel's proxy or your client cuts the connection first, raise the proxy
timeout — the work is not resumable.

### Cost

Roughly, per 2000-word article on `claude-opus-5` at `effort=high`: 4–8 API calls,
$0.10–$0.35. The long rule set is cached, so calls after the first in a document pay ~10%
for that prefix. Dropping `MODEL=claude-sonnet-5` cuts the bill about 40% at some cost in
style-instruction adherence; `EFFORT=medium` is the cheaper lever to try first.

---

## Configuration

Every setting is an environment variable. See `.env.example` for the full annotated list.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. |
| `APP_API_KEY` | *(empty)* | `X-API-Key` callers must send. Empty disables auth. |
| `MODEL` | `claude-opus-5` | Also read as `ANTHROPIC_MODEL`; `MODEL` wins if both are set. Overridable per request via `"model"`. |
| `EFFORT` | `high` | `low`…`max`. Also read as `ANTHROPIC_EFFORT`. Below `high` the rhythm rules get followed loosely. Overridable per request. |
| `MAX_TOKENS` | `16000` | |
| `ENABLE_REFUSAL_FALLBACK` | `true` | Server-side fallback if the model declines. Auto-disables itself if the account lacks the beta. |
| `TARGET_AI_SCORE` | `10` | |
| `MAX_ATTEMPTS` | `3` | Retry attempts per section. |
| `STRENGTH` | `standard` | `standard` \| `aggressive` \| `max`. The main quality/cost dial. `max` runs two passes. |
| `PASSES` | `0` | Full passes 1-3; `0` derives from `STRENGTH`. |
| `GUIDANCE_DETECTOR` | `local` | `local` runs the neural detector (needs the detector image, ~2GB RAM). `none` uses the stylometric proxy and will **not** move trained classifiers. |
| `GUIDANCE_MODEL` | `desklib/ai-text-detector-v1.01` | Must agree with your real detector — verify with `scripts/detector_check.py`. |
| `GUIDANCE_BATCH_SIZE` | `16` | Candidate scoring batch. |
| `GUIDANCE_THREADS` | `0` | `0` lets torch decide; set 1-2 on a small droplet. |
| `GUIDANCE_WARMUP` | `false` | Load weights at boot (~40s) instead of on the first request. |
| `ENABLE_ADVERSARIAL` | `true` | The detector-guided pass. |
| `ADVERSARIAL_ROUNDS` | `3` | Max substitution rounds. |
| `ADVERSARIAL_CANDIDATES` | `4` | Paraphrase candidates per sentence. |
| `ADVERSARIAL_SENTENCES_PER_ROUND` | `12` | Sentences rewritten per round. |
| `ADVERSARIAL_TARGET_PROBABILITY` | `0.25` | Stop once the detector is at or below this. |
| `ADVERSARIAL_MIN_SENTENCE_PROBABILITY` | `0.35` | Substitution threshold per sentence. |
| `CHUNK_TARGET_WORDS` | `700` | Smaller chunks buy attention per sentence; larger chunks buy rhythmic context. |
| `CHUNK_MAX_WORDS` | `1000` | |
| `ENABLE_POSTPROCESS` | `true` | |
| `CONCURRENCY` | `3` | Sections rewritten in parallel. Raise it and you will meet your rate limit. |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

---

## Tuning notes

- **`MAX_ATTEMPTS` is the main dial.** Going 1 → 3 costs roughly 2× and typically takes
  15–25 points off the score. Beyond 3 the returns fall off sharply.
- **If output reads over-edited**, lower `TARGET_AI_SCORE` pressure by *raising* the
  target (e.g. 15). Chasing a very low score pushes the model into mannered prose, which
  a human reader notices even when a detector does not.
- **If specific words keep surviving**, add them to `app/rules/tells_en.py`. `HARD_TELLS`
  is for words to remove almost entirely, `SOFT_TELLS` for words to thin out. They feed
  the prompt, the regex layer and the score from one place.
- **PT-BR is more conservative than EN by design.** There is no Portuguese equivalent of
  the corpus frequency studies behind the English list, so `tells_pt.py` is built from
  structural principles instead of measured shifts. Expect to extend it as you see what
  your drafts actually produce.

---

## Layout

```
app/
  main.py               FastAPI routes and error mapping
  adversarial.py        detector-guided sentence substitution (the attack)
  detector/             guidance detector: interface, local neural model, fallback
  html_bridge.py        HTML <-> Markdown round trip for CMS payloads
  config.py             env-var settings
  schemas.py            request/response contracts
  pipeline.py           orchestration: chunk, rewrite, validate, score, retry
  llm.py                Anthropic client, usage accounting, cost estimate
  structure.py          Markdown parse, masking, chunking, structural validation
  prompts/builder.py    the style rule set and per-request message assembly
  rules/tells_en.py     EN-US tell dictionaries
  rules/tells_pt.py     PT-BR tell dictionaries
  rules/postprocess.py  deterministic regex clean-up
  scoring/metrics.py    stylometric measurement, scoring, retry feedback
scripts/live_check.py   real end-to-end check with before/after report
scripts/detector_check.py  verify a guidance detector agrees with your real one
docs/n8n.md             curl examples and n8n HTTP Request node settings
tests/                  offline tests with a stubbed model
```

## Prior art

The rule set is assembled from published open-source work rather than invented:

- [harshaneel/humanize](https://github.com/harshaneel/humanize) — the nine-lever framing
  and the numeric rhythm targets.
- [HugoLopes45/llmstrip](https://github.com/HugoLopes45/llmstrip) — the 34-rule split
  between word-level and structural tells.
- [nicojan/humanize-text-prompt](https://github.com/nicojan/humanize-text-prompt) — the
  psycholinguistic layer (lexical diversity gap, sentiment flattening, dependency
  distance).
- Kobak et al. 2025, on excess-word frequency shifts in post-ChatGPT academic writing —
  the empirical basis for the `HARD_TELLS` list.
- [Binoculars](https://github.com/ahans30/Binoculars) (Hans et al., ICML 2024) — the
  zero-shot detection method this service's scorer deliberately does *not* implement,
  because it needs two ~1B-parameter models in memory. If you later move to a droplet
  with room for them, that is the upgrade path from a proxy score to a real one.
