# @papercusp/rerank

Engine-agnostic **cross-encoder reranking** (stage B of a retrieve→rerank search
pipeline), backed by ZeroEntropy **zerank**. Reorders candidate documents from
*any* retriever (Typesense, Postgres/pgvector, Elastic, …) by calibrated
relevance to the query.

Built to be **shared across repos** (Restart + papercusp): no project-specific
dependencies, all behavior is config. Papercusp can consume it as a normal
dependency and supply its own settings.

## Install

```bash
# from a registry (once published):
npm i @papercusp/rerank
# or, cross-repo without a registry, as a git/file dependency:
#   "@papercusp/rerank": "git+ssh://…/Papercusp/rerank.git"
```

Requires `ZEROENTROPY_API_KEY` in the environment (or pass `apiKey` per call).

## Usage

```ts
import { rerank, rerankAvailable } from '@papercusp/rerank';

const ranked = await rerank('dell laptop', candidates, {
  model: 'zerank-2',   // default
  topN: 24,            // return top N after reranking
  // minScore: 0.5,    // optional calibrated-score floor
});
// ranked: [{ row, score, reranked }] best-first
```

Where `candidates` is `{ id, text, row }[]` — `text` is what the reranker scores
(e.g. a product title), `row` is your payload carried through untouched.

## Design notes
- **Fail-safe:** missing key or any API error → returns candidates in their
  original (retrieval) order with `reranked: false`. A rerank outage never
  breaks search.
- **Calibrated scores:** zerank's scores are calibrated 0..1, so `minScore`
  thresholds are meaningful (unlike raw vector distances).
- **Per-project config:** model / topN / minScore differ per consumer; the code
  is identical.

## Build / publish
```bash
npm run build   # tsc → dist (CommonJS + .d.ts)
npm publish     # runs build via prepublishOnly
```
