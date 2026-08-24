# Calling the service — curl and n8n

Set these once in your shell to make the examples copy-pasteable:

```bash
export HUMANIZER_URL="https://your-app.easypanel.host"
export APP_API_KEY="the-value-you-set-in-easypanel"
```

For local testing, `export HUMANIZER_URL="http://localhost:8000"`.

---

## 1. `GET /health` — is it up?

No auth required. Use this as the n8n connection test.

```bash
curl -s "$HUMANIZER_URL/health"
```

```json
{"status":"ok","model":"claude-opus-5","auth_required":true,
 "anthropic_key_configured":true,"target_ai_score":10.0}
```

If `anthropic_key_configured` is `false`, `ANTHROPIC_API_KEY` never reached the container.
If `auth_required` is `false`, your `APP_API_KEY` is empty and the endpoint is open.

---

## 2. `POST /analyze` — score text, no rewrite

Free and instant: no Anthropic call. Use it to score your draft before spending anything,
and to score the output afterwards.

```bash
curl -s -X POST "$HUMANIZER_URL/analyze" \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# Unlocking Growth\n\nIn today'"'"'s landscape, companies are leveraging robust frameworks. Furthermore, this facilitates comprehensive outcomes. Moreover, it is important to note that alignment is crucial. Additionally, teams must delve into a myriad of options.",
    "language": "en-US"
  }'
```

The `'"'"'` dance is just how you get an apostrophe inside a single-quoted shell string.
In n8n you will not need it.

Response (trimmed):

```json
{
  "language": "en-US",
  "metrics": {
    "ai_score": 52.46,
    "verdict": "ai-like",
    "words": 32,
    "sentence_length_cv": 0.306,
    "tell_density_per_1k": 468.75,
    "penalties": {"burstiness": 12.46, "monotone_runs": 10.0,
                  "ai_vocabulary": 14.0, "opening_transitions": 10.0,
                  "contractions": 6.0}
  },
  "suggestions": ["Only 32 words: the per-1000-word rates and the burstiness figure are unreliable below ~150 words. ...", "..."]
}
```

### Sending a whole file

Never paste multi-line Markdown straight into `-d` — the newlines break the JSON. Build the
body with a tool that escapes it. With `jq`:

```bash
jq -Rs '{text: ., language: "en-US"}' article.md \
  | curl -s -X POST "$HUMANIZER_URL/analyze" \
      -H "X-API-Key: $APP_API_KEY" \
      -H "Content-Type: application/json" \
      -d @-
```

`jq` is not installed by default on Windows or on most slim Linux images. Python is already
a dependency of this project, so this equivalent works anywhere:

```bash
python -c "import json,sys;print(json.dumps({'text':open(sys.argv[1],encoding='utf-8-sig').read(),'language':'en-US'}))" article.md \
  | curl -s -X POST "$HUMANIZER_URL/analyze" \
      -H "X-API-Key: $APP_API_KEY" \
      -H "Content-Type: application/json" \
      -d @-
```

`utf-8-sig` strips the byte-order mark that Windows editors like to add, which would
otherwise land inside your first heading.

---

## 3. `POST /humanize` — the real thing

### Minimal

```bash
curl -s -X POST "$HUMANIZER_URL/humanize" \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  --max-time 400 \
  -d '{
    "text": "# My Article Title\n\nFirst paragraph of the draft.\n\n## A Section\n\nMore body text here.",
    "language": "en-US"
  }'
```

**`--max-time 400` matters.** A 2000-word article at `MAX_ATTEMPTS=3` takes 60–180
seconds, sometimes more. curl's default has no timeout but n8n's does, and the work is
not resumable — a cut connection wastes the whole spend.

### From a file, with everything set

```bash
python -c "import json,sys;print(json.dumps({'text':open(sys.argv[1],encoding='utf-8-sig').read(),'language':'en-US','tone':'skeptical industry analyst, first person, dry','preserve_terms':['Acme Cloud','SOC 2'],'target_ai_score':10,'max_attempts':3,'rewrite_headings':True}))" article.md \
| curl -s -X POST "$HUMANIZER_URL/humanize" \
    -H "X-API-Key: $APP_API_KEY" \
    -H "Content-Type: application/json" \
    --max-time 600 \
    -d @- > result.json
```

Then read the summary and pull the article out for your detector, in one step:

```bash
python -c "
import json
r = json.load(open('result.json', encoding='utf-8'))
print('title      ', r['title'])
print('score      ', r['metrics_before']['ai_score'], '->', r['metrics']['ai_score'])
print('target met ', r['target_met'])
print('cost       \$%.4f' % r['usage']['estimated_cost_usd'])
print('warnings   ', r['warnings'] or 'none')
open('article.humanized.md','w',encoding='utf-8').write('# ' + r['title'] + '\n\n' + r['content'])
print('written     article.humanized.md')
"
```

### Push harder against detectors (`strength`)

```bash
curl -s -X POST "$HUMANIZER_URL/humanize" \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  --max-time 600 \
  -d '{
    "text": "# Title\n\nBody...",
    "language": "en-US",
    "strength": "max"
  }'
```

`strength` is the aggressiveness dial:

- `standard` (default) — one structure-preserving pass.
- `aggressive` — one hard pass, every section reworked, extra human-texture instructions, lower target.
- `max` — aggressive plus a second pass that injects human irregularity into the already-rewritten text.

**Honest expectation:** higher strength helps most against perplexity/burstiness checkers (ZeroGPT, QuillBot). Against trained classifiers (Copyleaks, and TruthScan/Undetectable) the gain is modest — those detect a model fingerprint that surface rewriting cannot fully remove. `max` also **doubles the cost and time** (two passes) and raises fact-drift risk, so check the `warnings` array: a `numbers may have changed` warning means a statistic moved and you should verify it.

The response reports `strength`, `passes_run`, and `score_trajectory` (the internal score after each pass) so you can see whether the second pass actually helped.

### Override the model per call

Handy for A/B testing without redeploying. Must be a `claude-*` id.

```bash
curl -s -X POST "$HUMANIZER_URL/humanize" \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  --max-time 400 \
  -d '{
    "text": "# Title\n\nBody...",
    "language": "en-US",
    "model": "claude-sonnet-5",
    "effort": "high"
  }'
```

### PT-BR

```bash
curl -s -X POST "$HUMANIZER_URL/humanize" \
  -H "X-API-Key: $APP_API_KEY" \
  -H "Content-Type: application/json" \
  --max-time 400 \
  -d '{
    "text": "# Como Impulsionar Resultados\n\nNo mundo de hoje, empresas buscam alavancar frameworks robustos. Alem disso, isso possibilita resultados abrangentes.",
    "language": "pt-BR",
    "tone": "jornalista de negocios, cetico, primeira pessoa"
  }'
```

### Freeze SEO-locked headings

```bash
-d '{"text": "...", "language": "en-US", "rewrite_headings": false}'
```

Heading lines come back character for character; only body text is rewritten.

---

## Wiring it into n8n

Use the **HTTP Request** node.

| Setting | Value |
|---|---|
| Method | `POST` |
| URL | `https://your-app.easypanel.host/humanize` |
| Authentication | None (the key goes in a header) |
| Send Headers | on |
| Header | `X-API-Key` = your `APP_API_KEY` |
| Send Body | on |
| Body Content Type | `JSON` |
| Specify Body | `Using JSON` |
| **Options → Timeout** | `600000` (ms — this is the one people forget) |

For the JSON body, use an expression so n8n handles the escaping:

```
{{ JSON.stringify({
  text: $json.article_markdown,
  language: "en-US",
  target_ai_score: 10,
  max_attempts: 3
}) }}
```

`JSON.stringify` is the important part. If you paste raw Markdown into the body field, the
newlines in your article will break the JSON. Let n8n serialise it.

Reading the response in the next node:

- `{{ $json.title }}` — the article title
- `{{ $json.content }}` — the body, **without** the H1
- `{{ $json.metrics.ai_score }}` — score after
- `{{ $json.metrics_before.ai_score }}` — score before
- `{{ $json.target_met }}` — boolean
- `{{ $json.warnings }}` — array; empty is good
- `{{ $json.usage.estimated_cost_usd }}`

### An IF node worth adding

Gate publication on the result rather than trusting it:

```
{{ $json.target_met && $json.warnings.length === 0 }}
```

`warnings` is where the service tells you a section could not be improved, that a rewrite
damaged the structure and the original shipped instead, or that the model declined. A run
can return `200 OK` with usable-looking text and still have a section that was never
actually rewritten.

### Retries

Leave n8n's node-level retry **off** for `/humanize`. A retry re-runs the whole article and
doubles the spend. The service already retries internally per section, which is the level
where retrying is cheap.

---

## Error responses

Errors come back as `{"detail": "..."}`, except `422`, where FastAPI makes `detail` an
*array* of per-field objects. If you parse errors in n8n, handle both shapes.

```json
{"detail":[{"type":"value_error","loc":["body","model"],
            "msg":"Value error, model must be a Claude model id, e.g. 'claude-sonnet-5'",
            "input":"gpt-5"}]}
```

| Status | Meaning | What to do |
|---|---|---|
| `400` | Server key missing/rejected, or Anthropic rejected the request | Read `detail`; check `/health` for `anthropic_key_configured` |
| `401` | Bad or missing `X-API-Key` | Check the header name and the env var |
| `402` | Anthropic account has no credit balance | Add credits at console.anthropic.com -> Plans & Billing |
| `422` | Body failed validation | Read `detail[].msg`; usually an empty `text` or a non-`claude-*` model |
| `429` | Anthropic rate limit | Honour the `Retry-After` header; lower `CONCURRENCY` |
| `502` | No key configured, key rejected, or Anthropic errored | Check `/health` for `anthropic_key_configured`; the `detail` says which |
| `504` | Could not reach Anthropic | Network or outage |
| `500` | Bug on our side | Check the container logs |

A `429` with `CONCURRENCY=3` on a fresh Anthropic account is common — new accounts have low
limits. Drop it to `1` and the article just takes longer.
