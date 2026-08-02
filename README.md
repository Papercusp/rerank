# @papercusp/rerank

Engine-agnostic **cross-encoder reranking** (stage B of a retrieve→rerank search
pipeline). Reorders candidate documents from *any* retriever (Typesense,
Postgres/pgvector, Elastic, …) by relevance to the query.

Two engines, one contract:

| engine | scoring | needs | use when |
|---|---|---|---|
| `local` | ONNX cross-encoder in-process | no key, no network | reranking every search — there is no per-call cost |
| `zeroentropy` *(default)* | hosted zerank API | `ZEROENTROPY_API_KEY` | you want calibrated scores + instruction following |

Built to be **shared across repos** (Restart + papercusp): no project-specific
dependencies, all behavior is config.

## Install

```bash
# from a registry (once published):
npm i @papercusp/rerank
# or, cross-repo without a registry, as a git/file dependency:
#   "@papercusp/rerank": "git+ssh://…/Papercusp/rerank.git"
```

The hosted engine needs `ZEROENTROPY_API_KEY` (or a per-call `apiKey`). The local
engine needs the optional peer `@huggingface/transformers` — nothing else.

## Usage

```ts
import { rerank, rerankAvailable } from '@papercusp/rerank';

// Local: no key, no network, no per-call cost.
const ranked = await rerank('dell laptop', candidates, {
  engine: 'local',
  topN: 24,
  // model: FAST_RERANKER_MODEL,  // ~7.5x cheaper per pair, lower quality
});

// Hosted zerank (the default engine).
const ranked2 = await rerank('dell laptop', candidates, { topN: 24 });
// ranked: [{ row, score, reranked }] best-first
```

Where `candidates` is `{ id, text, row }[]` — `text` is what the reranker scores,
`row` is your payload carried through untouched.

**`text` should be the MATCH-CENTRED passage**, not the head of the document. A
cross-encoder judges the text you hand it; give it a document's opening when the
query matched paragraph 12 and it scores the wrong thing.

### Warming the local model

Loading is seconds, scoring is milliseconds. A host that serves searches should
preload rather than pay the load on a user's first query:

```ts
import { loadCrossEncoder } from '@papercusp/rerank';
await loadCrossEncoder();          // caches per (model, dtype)
```

## Design notes
- **Fail-safe:** missing key, missing ONNX runtime, or any error → candidates come
  back in their original (retrieval) order with `reranked: false`. A rerank outage
  degrades to retrieval order; it never breaks search. Every engine funnels through
  one seam so this cannot be re-litigated per engine.
- **Stable tie-break:** equal scores keep retrieval order — the first-stage ranking
  is the right tiebreak.
- **Scores are 0..1** on both engines (zerank's are calibrated; the local engine
  sigmoids its logit), so `minScore` is meaningful — unlike raw vector distances.
  ⚠ On the local engine an absolute threshold is **dtype-specific**: quantization
  preserves *ordering* but compresses the score *range*, so fit any floor on the
  dtype you actually deploy.
- **Cost is bounded by the candidate cap, not by hardware.** Measured: per-pair
  latency does not improve with more cores, so cap candidates rather than hoping
  for a bigger box. ORT threads are capped for the same reason (`ORT_SESSION_OPTIONS`).
- **`instruction` is zerank-only** — injecting instruction prose into a plain
  cross-encoder's query degrades scoring rather than steering it.
- **Per-project config:** engine / model / topN / minScore differ per consumer; the
  code is identical.

> ⚠ This package must stay `"type": "module"`. It is named-imported by ESM
> consumers running under tsx; as a typeless package its `.ts` transpiles to CJS,
> and `cjs-module-lexer` cannot see `export … from` re-export chains — the whole
> namespace links empty and every named import throws at runtime. Unit tests and
> `tsc` do **not** catch it (vitest bundles instead of linking).

## Build / publish
```bash
npm run build   # tsc → dist (CommonJS + .d.ts)
npm publish     # runs build via prepublishOnly
```
