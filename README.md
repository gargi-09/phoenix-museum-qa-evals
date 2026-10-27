# Catching LLM False Passes with Deterministic Verifiers & Arize Phoenix

[![Live dashboard](https://img.shields.io/badge/live%20dashboard-gargi--09.github.io-10b981?style=flat-square&logo=githubpages&logoColor=white)](https://gargi-09.github.io/phoenix-museum-qa-evals/)
[![Arize Phoenix](https://img.shields.io/badge/observability-Arize%20Phoenix-6d28d9?style=flat-square)](https://docs.arize.com/phoenix)
[![OpenInference](https://img.shields.io/badge/tracing-OpenInference%20%2B%20OTel-0ea5e9?style=flat-square)](https://github.com/Arize-ai/openinference)

A museum-records QA pipeline (~5,000 records from the Cleveland Museum of Art
and the Art Institute of Chicago) instrumented end to end with OpenInference,
with its deterministic verifiers registered as custom Phoenix evaluators.

**The problem it demonstrates:** a reference-free LLM-as-a-judge sees only the
question and the answer. *"No violations found. The records are consistent and
every field is verified."* is fluent, on-topic and confident, so the judge
passes it. It also cites **zero** source records. The deterministic verifiers
see the citations, the retrieved candidate set and the billed `prompt_tokens`,
so they fail it, and Phoenix shows both verdicts on the same trace.

![The zero-sources trap in Phoenix: the LLM judge scores 1.00 while zero_source_false_pass and citation_groundedness score 0.00](docs/assets/trace-zero.png)

*The zero-sources trap in Phoenix. The reference-free judge (`llm_judge_surrogate`) scores the answer 1.00; the deterministic verifiers score it 0.00.*

## 60-second quickstart

```bash
pip install -r requirements.txt && python demo_phoenix_evals.py
```

Then open **http://localhost:6006**. No network, API key or GPU is needed: the
demo runs in **fixture mode** on 10 real public museum records that ship with
the repo (see [Demo modes](#demo-modes)). The Phoenix app it starts stays up
until you press Ctrl+C.

To run it against the full ~5,000-record public corpus instead:

```bash
python src/main.py fetch && python src/main.py ingest && python demo_phoenix_evals.py --mode corpus
```

| Flag | Effect |
|---|---|
| `--mode corpus` | full public corpus + the real BM25 + bge-small retriever, scripted model, no credentials |
| `--mode live` | full public corpus + a real model, Claude Haiku 5.5 by default (see [Live mode](#live-mode)). Paid, opt-in only |
| `--judge live` | replace the offline judge surrogate with a real LLM judge call per trace |
| `--no-launch` | attach to a Phoenix you started yourself (`phoenix serve`) |
| `--exit` | exit when done instead of keeping Phoenix alive |

## Architecture

```mermaid
flowchart LR
    Q[Question] --> R

    subgraph R[Retrieval · local, free]
        direction TB
        BM25[BM25 sparse] --> RRF[Reciprocal Rank Fusion]
        DENSE[bge-small dense] --> RRF
        RRF --> GATE{Abstention gate}
    end

    GATE -- abstain --> OUT[Answer + provenance + confidence]
    GATE -- top 10 --> CTX[Entity resolution<br/>+ contradiction flags]
    CTX --> AUDIT[Prompt token audit]
    AUDIT --> LLM[LLM generation<br/>one bounded call]
    LLM --> VER[4 deterministic verifiers<br/>+ confidence cap]
    VER --> OUT

    R -. RETRIEVER span .-> OTEL
    AUDIT -. GUARDRAIL span .-> OTEL
    LLM -. LLM span .-> OTEL
    VER -. EVALUATOR span .-> OTEL

    OTEL[OpenInference / OTLP] --> PX[(Arize Phoenix<br/>localhost:6006)]
    PX --> EV[Phoenix evaluators<br/>re-run from each trace]
    EV -- span annotations --> PX
```

```
Retrieval (BM25 + Dense RRF) → LLM Generation → OpenInference Tracing → Phoenix Deterministic Evaluators
```

![Phoenix trace view of a live Claude Haiku 5.5 run](docs/assets/phoenix-trace.png)

*A live run on Claude Haiku 5.5: the span tree on the left, and all 10 evaluator verdicts as span annotations.*

### What gets traced

There is no LangChain or LLM SDK in this pipeline: `src/api_client.py` posts to
the endpoint with raw `requests`, so no `openinference-instrumentation-*`
auto-instrumentor has anything to hook. [`src/tracing.py`](src/tracing.py)
emits spans by hand using OpenInference semantic conventions, by wrapping the
pipeline's own seams. `reasoning.py` is not modified, and a run without
`instrument_pipeline()` behaves exactly as before.

| Span | Kind | Key attributes |
|---|---|---|
| `museum_qa.answer_question` | CHAIN | `museum_qa.citations.count`, `museum_qa.confidence.*`, `museum_qa.system_prompt_dropped_suspected`, `museum_qa.eval_payload` |
| `retrieval.hybrid_rrf` | RETRIEVER | `retrieval.documents.N.document.{id,score,content}`, `museum_qa.retrieval.bm25_max`, `…best_dense_similarity`, `…abstain` |
| `prompt.token_audit` | GUARDRAIL | `museum_qa.prompt_tokens_expected`, `…instruction_tokens_expected`, `…context_block_ids` |
| `llm.generate` | LLM | `llm.input_messages`, `llm.output_messages`, `llm.token_count.{prompt,completion,total}`, `museum_qa.prompt_tokens`, `museum_qa.prompt_audit.*` |
| `verifier.deterministic_suite` | EVALUATOR | `museum_qa.citations.{count,valid_count,fabricated_count}`, `museum_qa.check.<name>` |

**Why the prompt token audit exists.** This endpoint was measured silently
discarding a separately supplied system prompt: a bare `"hi"` billed 8
`prompt_tokens`, and `"hi"` plus a ~13-token system prompt *also* billed 8,
with a 200 OK. Nothing errors, so the only evidence is the token count. The
GUARDRAIL span estimates the prompt size with and without the ~450-token
instruction block before the call; the LLM span compares the billed count
against both and sets `museum_qa.prompt_audit.status = dropped_suspected` when
the bill is closer to the no-instructions estimate. It is a heuristic (the
estimates are ~4 chars/token), which is why the field says *suspected*.

## Phoenix evaluators

[`src/phoenix_evals.py`](src/phoenix_evals.py). Each evaluator takes the
payload stored on a root span and returns `(label, score, explanation)`. The
demo pulls root spans back **out of Phoenix**
(`phoenix.client.Client().spans.get_spans_dataframe`), runs the evaluators on
them, and writes the verdicts back with `log_span_annotations`, so they show
up in the Annotations tab of the trace they judge.

| Evaluator | Fails when |
|---|---|
| `zero_source_false_pass` | the output asserts an answer, or an all-clear like "no violations", while citing 0 valid source records |
| `citation_groundedness` | any cited record ID is missing from the BM25 + dense RRF candidate set (score = fraction grounded) |
| `confidence_capping` | a confidence escaped its cap: a score emitted with 0 valid citations or a dropped prompt, or a `high` label despite a failed check |
| `prompt_integrity` | the billed `prompt_tokens` say the instruction block never arrived |
| `verifier.citation_membership`<br>`verifier.not_recorded_violation`<br>`verifier.factual_match`<br>`verifier.not_truncated` | the pipeline's own four checks (`reasoning.check_*`), **re-executed** against the context blocks recorded in the trace |
| `llm_judge_surrogate` / `llm_judge` | baseline: a standard reference-free judge that sees question + answer only |

A check that cannot fail does not report a pass. When a response cites
nothing, the per-citation verifiers return `INCONCLUSIVE`, not `PASS`, and
are left out of the pass rate.

### The side-by-side

The demo runs five questions. Run in fixture mode, it prints:

| # | Scenario | LLM judge | Deterministic |
|---|---|---|---|
| 1 | Clean answer, 1 valid citation | PASS | PASS |
| 2 | "No violations found", 0 citations | **PASS** | **FAIL**: zero sources, confidence capped to 0.0 |
| 3 | System prompt silently dropped | **PASS** | **FAIL**: `prompt_integrity`, 0 citations |
| 4 | Cites a record outside the candidate set | **PASS** | **FAIL**: `citation_groundedness` 0.5 |
| 5 | Out-of-corpus question | NOT_APPLICABLE (nothing to judge) | NOT_APPLICABLE: abstained before any paid call |

`--mode corpus` gives the same verdicts on rows 1–4. On row 5 the question
gets past the abstention gate on the public corpus (see
[the thresholds note](#a-note-on-the-thresholds)), and the scripted model
declines it. `zero_source_false_pass` correctly reads that as a decline, not
a false pass.

In fixture mode, the project's span-annotation summary in Phoenix shows the gap directly:
`llm_judge_surrogate` averages **1.00**, while `deterministic_verdict` averages
**0.25** (1 of the 4 generated answers passes).

![The dropped-system-prompt case: prompt_integrity scores 0.00 while the judge scores 1.00](docs/assets/trace-dropped.png)

**On a real model.** In a live run on Claude Haiku 5.5, every evaluator scored
1.00 on all 5 questions. Asked "Are there any violations…", Haiku answered
from the records it was shown and cited them, so it never fell into the
zero-sources trap. The verifiers passed it because the answer earned it. The
fixture run is what demonstrates the failure modes; the live run shows that the
verifiers don't produce false alarms on good answers.

### Demo modes

Every mode runs the real pipeline code: context assembly, entity resolution,
contradiction detection, prompt construction, parsing, all four verifiers,
confidence scoring and provenance. The modes differ in the corpus, the dense
retriever and the model:

| Mode | Corpus | Dense retrieval | Model | Needs |
|---|---|---|---|---|
| `fixture` (default) | 10 public records in `src/fixtures/` | TF-IDF | scripted | nothing |
| `corpus` | ~5,000 public records in `data/` | bge-small | scripted | `fetch` + `ingest`, ~4–5 min first embed |
| `live` | ~5,000 public records in `data/` | bge-small | your endpoint | `.env` credentials |

- **The fixture is real data, just small.** It holds the 10 works the demo
  questions are about, fetched from the public APIs and normalized by
  `ingest.py`. Nothing in it was typed by hand. Rebuild it with
  `python src/fetch_public_corpus.py --fixture`.
- **The scripted model** returns one response per question, each built to
  trigger a single failure mode. `prompt_tokens` is computed from the prompt
  the pipeline actually built, and the dropped-prompt case leaves out the
  instruction tokens, the way the original endpoint was measured to. Every
  record a script cites exists in both corpora, except the one fabricated on
  purpose (`aic-99999`), so the same scripts work in `fixture` and `corpus`
  mode.
- **The baseline judge** is an offline surrogate unless you pass
  `--judge live`. It is deterministic and gets exactly a reference-free
  judge's inputs.
- **In live mode** the model answers for itself, so whether a trap fires
  depends on what it actually says. The evaluators report the result either
  way.

## Data: public museum records

The corpus is built from the two museums' own public collection APIs. No
credentials are needed and there is no private endpoint.

```bash
python src/main.py fetch                 # -> data/base.jsonl + data/SOURCES.json
python src/main.py ingest                # -> data/normalized.jsonl
```

[`src/fetch_public_corpus.py`](src/fetch_public_corpus.py):

- **What it fetches.** 2,500 described artworks from each museum (change this
  with `--per-source N`), plus the 10 seed works the demo questions name,
  which are always fetched by ID. A seed that has been withdrawn is reported,
  not silently dropped.
- **Deterministic sample.** CMA is paged in its default order. AIC uses its
  search API with "has a description" filtered server-side, sorted by id,
  with keyset pagination (`id > last id`). The search API refuses to page
  past 1,000 results, which keyset pagination stays inside.
- **Polite.** It sends the `AIC-User-Agent` header AIC asks for (set
  `PUBLIC_API_CONTACT` to your own contact), keeps under AIC's 60
  requests/minute, and retries 429/5xx with backoff.
- **Cached and resumable.** Every API page is saved to `data/raw/`, so a
  re-run makes 0 network calls. Pass `--refresh` to re-download.
- **Nothing dropped silently.** Skipped records are counted by reason in the
  console and in `data/SOURCES.json`, along with the record counts, the seed
  status and the licences.
- **Output** is `ingest.py`'s `structured` format, so the rest of the pipeline
  is unchanged. CMA publishers and printers are left out of the artist field.
  No images are downloaded; `image_path` is filled only for openly licensed
  images.

A full fetch takes about 25 requests (CMA serves 1,000 per page, AIC 100) and
under a minute.

### Licences and attribution

| Source | API | Licence |
|---|---|---|
| Cleveland Museum of Art | [Open Access API](https://openaccess-api.clevelandart.org/) | metadata CC0 1.0 ([terms](https://www.clevelandart.org/open-access)) |
| Art Institute of Chicago | [Public API](https://api.artic.edu/docs/) | CC0 1.0, **except `description`, which is CC BY 4.0** ([terms](https://www.artic.edu/terms)) |

Artwork descriptions from the Art Institute of Chicago, including those in
`src/fixtures/museum_fixture.jsonl`, are © the Art Institute of Chicago and
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A
CMA record's `share_license_status` describes the rights to its *image*, not
its metadata.

### A note on the thresholds

The abstention thresholds in `src/retrieval.py` (BM25 19.5, dense 0.35) and
the numbers in its comments were measured on the corpus snapshot the pipeline
was first built against. That snapshot came from the same two museums, but it
was a different sample. On the public corpus:

| Question | Best BM25 | Best dense | Gate | Top-1 record |
|---|---|---|---|---|
| Coney Island Beach, Reginald Marsh | 50.93 | 0.767 | passes | `cma-117571` ✓ |
| Louise Bourgeois, Ode to My Mother | 43.62 | 0.714 | passes | `cma-160794` ✓ |
| Omer Fast, The Casting | 26.99 | 0.776 | passes | `cma-168184` ✓ |
| Jeff Brouws, Railroad Landscape #33 | 60.54 | 0.754 | passes | `cma-169223` ✓ |
| Melting point of gallium arsenide | **19.75** | 0.560 | **passes** (just over 19.5) | an unrelated glass work |
| Capital of Mongolia | 17.22 | 0.561 | abstains | n/a |

Retrieval puts the right record first for every in-corpus question. The
gallium-arsenide question, which abstained on the old snapshot at 18.50, now
clears the floor by 0.25, so in `corpus` mode it reaches the model and is
declined there instead of being stopped before the call. This is the overlap
`retrieval.py` already documents: no absolute BM25 cutoff separates the two
classes. The threshold has deliberately not been retuned on six samples.

Re-check them on your own fetch with:

```bash
python src/retrieval.py data/normalized.jsonl "your question" "an off-topic question"
```

`src/demo/transcript.md` is likewise a record of a run on that earlier
snapshot. Some of its record IDs, such as `aic-215406` and `aic-13026`, are no
longer published by AIC.

## Interactive dashboard

[`docs/index.html`](docs/index.html) is a standalone page (Tailwind CDN,
vanilla JS, Lucide icons) built for GitHub Pages. It has a scenario studio for
the clean, zero-sources and dropped-prompt cases, the raw response next to
both verdicts, the evaluator annotations, and a span waterfall with token
telemetry.

### Evaluator sandbox: try to fool the verifiers, no key needed

The dashboard's sandbox panel runs the evaluators **in the visitor's
browser**:

1. Pick one of the five traced runs.
2. Edit what the model said: strip citations, cite a record that was never
   retrieved, claim "no violations", state a wrong year, answer in prose, or
   untick "system prompt delivered".
3. The pipeline re-scores it as you type, with the judge surrogate next to
   the deterministic verifier and all eight evaluators listed.

There is no model call and no API key, so it's safe on a public page.

- **Engine.** [`docs/sandbox.js`](docs/sandbox.js) is a line-by-line port of
  everything after the model call: parsing, truncation detection, the four
  `reasoning.check_*` verifiers, confidence scoring, the prompt-token audit,
  and every evaluator in `phoenix_evals.py`.
- **Data.** [`docs/data/sandbox.json`](docs/data/sandbox.json) holds the
  payloads recorded on real fixture-mode traces: question, RRF candidate set,
  context blocks and raw output. Retrieval is fixed to those records;
  everything after the model call re-runs. Rebuild it with:
  ```bash
  python src/export_sandbox_data.py
  ```
- **Parity is checked, not claimed.** On load, the page re-scores every
  exported run and compares each verdict with the Python run's. It shows the
  result ("50/50 verdicts match"). From the CLI:
  ```bash
  node docs/sandbox.js docs/data/sandbox.json
  ```
  A wider test of 104 edited answers, run through both the Python pipeline
  and the JS port, matched on all 1,040 verdicts. Reproduce it (needs
  Node.js and the full `requirements.txt` environment) with:
  ```bash
  python -m pytest tests/test_sandbox_parity.py -v
  ```

Building it surfaced two real issues, both now fixed in the Python code:

- The fixture's scripted model skipped truncation detection, which
  `call_haiku()` normally runs, so a scripted answer cut off mid-JSON scored
  `not_truncated` PASS.
- `deterministic_verdict` only rolled up the four headline evaluators, so an
  answer with the wrong year (`verifier.factual_match` FAIL) still showed an
  overall PASS. The verdict now fails if any deterministic evaluator fails.

Screenshots from a real run go in [`docs/assets/`](docs/assets/README.md),
which lists each filename and what it should show. The dashboard picks them up
automatically: a "From a real run" gallery, plus a Phoenix trace image inside
each scenario. Any image that isn't there yet is hidden rather than shown
broken.

To deploy it, open the repo's **Settings → Pages**, choose **Deploy from a
branch**, then `main` / `/docs`.

---

## The underlying pipeline

Retrieval narrows 5,000 records to 10 using free local compute (BM25 +
`bge-small-en-v1.5`, fused by Reciprocal Rank Fusion). Each question makes
exactly one paid model call. Each answer returns the records it used, the
records it saw and didn't use, four deterministic verification checks, and a
computed confidence score.

- **Demo transcript:** [`src/demo/transcript.md`](src/demo/transcript.md), 10
  questions run end to end on an earlier corpus snapshot (see
  [the note above](#a-note-on-the-thresholds)).

### Setup

```bash
python -m venv venv
venv\Scripts\activate          # or: source venv/bin/activate
pip install -r requirements.txt
python src/main.py fetch       # public museum APIs -> data/base.jsonl
python src/main.py ingest      # -> data/normalized.jsonl (offline, free)
```

`data/` is gitignored, so each clone builds its own corpus. Retrieval needs
no credentials:

```bash
python src/retrieval.py data/normalized.jsonl "What year was Coney Island Beach made?"
```

### Live mode

Live mode makes real model calls, so it needs credentials of your own. It
never runs unless you ask for it: the default `auto` mode stays offline even
when a key is present.

**Anthropic API (recommended).** Put your key in `.env` in the repo root.
`.env` is gitignored, so it is never committed:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then run:

```bash
python demo_phoenix_evals.py --mode live
```

- **Model.** `claude-haiku-5-5` by default (Claude Haiku 5.5, the cheapest
  current model). Change it with `ANTHROPIC_MODEL`.
- **Cost.** A demo run is 5 calls of roughly 3,500 input and a few hundred
  output tokens each, which comes to well under a cent at Haiku 5.5's rates.
  `--judge live` adds 5 short judge calls. The demo prints the total tokens it
  spent. Setting a spend limit in the Anthropic Console is a sensible cap.
- **What's different on this path** (all of it is in `_call_anthropic()` in
  [`src/api_client.py`](src/api_client.py)):
  - It uses the official `anthropic` SDK.
  - The system prompt goes in the real `system` field.
  - No `temperature` is sent; Haiku 5.5 rejects sampling parameters.
  - Thinking runs at `low` effort (`ANTHROPIC_EFFORT`), with headroom so it
    can't crowd out the JSON answer.
  - A safety refusal is reported as an error, not parsed as an answer.
- **Prompt-token audit.** On this path the audit uses the free `count_tokens`
  endpoint for exact expected counts instead of the ~4 chars/token estimate.

**The original proxy.** The pipeline was first built against a hosted proxy
that speaks a small HTTP contract:

```
POST {MODEL_ENDPOINT}
Authorization: Bearer {API_KEY}
{"messages": [...], "max_tokens": N, "temperature": 0.0}
-> {"content": "...", "usage": {"prompt_tokens": N, "completion_tokens": N}}
```

Any gateway with that shape works. Set `API_KEY`, `MODEL_ENDPOINT` and
`MESSAGE_FORMAT=prepend` in `.env`, or `LLM_PROVIDER=proxy` to force this
path. `prepend` folds the instructions into the user message, because that
proxy silently discarded a separately supplied system prompt (measured both
ways: `prompt_tokens` didn't move for a top-level `system` key or for a
`{"role": "system"}` entry). That measurement is why the prompt-token audit
exists. See the header of [`src/api_client.py`](src/api_client.py).

### Usage

```bash
python src/main.py ask "What year was Coney Island Beach made?"
python src/main.py ask --dev "What year was Coney Island Beach made?"   # proxy only: sends X-Mode: dev
python src/main.py demo                                                  # 10-question suite
```

**Caching.** On the first run the pipeline embeds the ~5,000 records (~4–5
min) and saves them to `cache/` with a SHA-256 fingerprint of the passage
text. If ingestion changes the passages, the fingerprint no longer matches and
the cache is rebuilt instead of serving stale vectors.

**Curator corrections** persist across runs and are applied before retrieval,
entity resolution and contradiction checking:

```bash
python src/memory.py add cma-168184 year_raw 1999 "curator correction"
python src/memory.py list
```

**Token guardrails.** Ranking is local, so it makes no remote calls. Each
answer costs exactly one bounded model call. Verification runs after
generation and costs no extra tokens.

Original architecture diagram:

<img width="5890" height="8192" alt="Pipeline architecture" src="https://github.com/user-attachments/assets/05983f83-2227-493a-996a-619794b2c7bd" />

### Layout

```
demo_phoenix_evals.py  Phoenix demo: trace 5 questions, evaluate the traces
docs/index.html        GitHub Pages diagnostic dashboard
docs/sandbox.js        in-browser evaluator engine (JS port, parity-checked)
docs/data/             sandbox.json: real traced payloads + Python verdicts
docs/assets/           screenshots, shown on the dashboard and here (see its README.md)
src/
  main.py              CLI: fetch | ingest | ask | demo
  fetch_public_corpus.py  CMA + AIC public APIs -> data/base.jsonl
  ingest.py            normalisation, 4 deterministic extraction patterns
  retrieval.py         BM25 + bge-small + RRF + abstention gate
  entity_resolution.py duplicate folding, cross-record year checks
  contradiction.py     prose-vs-structured-field contradictions
  memory.py            correction persistence
  reasoning.py         orchestrator: prompt, parse, verify, score
  api_client.py        the only module that calls the endpoint
  tracing.py           OpenInference spans + Phoenix session
  phoenix_evals.py     deterministic Phoenix evaluators + judge baseline
  fixture_mode.py      offline retriever and scripted model
  export_sandbox_data.py  real traced payloads -> docs/data/sandbox.json
  fixtures/            10 public records, normalized (committed)
test_scripts/          test scripts used during building
data/                  base.jsonl, normalized.jsonl, SOURCES.json, raw/ API cache (gitignored)
cache/                 embedding cache (gitignored, rebuilt on first run)
outputs/               demo eval report (gitignored)
```

The root-level scripts are development tools, not pipeline code: data
forensics (`test_scripts/analyze_corpus.py`, `final_stress_test.py`,
`inspect_*.py`) and rejected experiments kept as evidence
(`nli_contradiction_test.py`, `contradiction_nli_approach.py`). NLI was tested
twice for contradiction detection and did worse both times.
`mock_reasoning_test.py` runs the full pipeline with `call_haiku`
monkeypatched: five failure scenarios, zero tokens.
