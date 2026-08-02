/**
 * @papercusp/rerank — engine-agnostic cross-encoder reranking.
 *
 * Stage B of the retrieve→rerank pipeline: given a query and candidate docs
 * (from ANY retriever — Typesense, Postgres, pgvector, …), reorder them by a
 * cross-encoder's relevance score. Has NO project-specific deps, so it can be
 * lifted into any consumer unchanged — the only per-project differences are the
 * config (engine, model, topN, threshold).
 *
 * Two engines, one contract:
 *   - `local`        — an ONNX cross-encoder in-process (no key, no network, no
 *                      per-call cost). See `local-engine.ts`.
 *   - `zeroentropy`  — the hosted zerank API (calibrated scores, instruction
 *                      following, someone else's GPU).
 *
 * Fail-safe by design: if the engine is unavailable or errors, it returns the
 * candidates in their ORIGINAL order — a rerank outage degrades to "retrieval
 * order", it never breaks search. Every engine funnels through the same
 * `EngineScores` seam below precisely so that guarantee is enforced in ONE
 * place and cannot be re-litigated per engine.
 */
import ZeroEntropy from 'zeroentropy';

import { type LocalRerankOptions, localEngineScores } from './local-engine';

export {
  DEFAULT_BATCH_SIZE,
  DEFAULT_MAX_LENGTH,
  DEFAULT_RERANK_DTYPE,
  FAST_RERANKER_MODEL,
  LOCAL_RERANKER_MODEL,
  ORT_SESSION_OPTIONS,
  loadCrossEncoder,
  localRerankAvailable,
  scoreCrossEncoder,
  type LocalRerankOptions,
} from './local-engine';

export interface RerankDoc<T> {
  /** Stable id (for caching / debugging). */
  id: string;
  /** The text the reranker scores against the query (e.g. product title). */
  text: string;
  /** Caller's payload, carried through untouched. */
  row: T;
}

/** Which cross-encoder does the scoring. Default `zeroentropy`. */
export type RerankEngine = 'zeroentropy' | 'local';

export interface RerankOptions extends LocalRerankOptions {
  /**
   * Scoring engine. `local` runs ONNX in-process (no key, no network);
   * `zeroentropy` calls the hosted zerank API. Default `zeroentropy` — callers
   * opt in to local, so an existing consumer's behavior is unchanged.
   */
  engine?: RerankEngine;
  /** Model id. Engine-specific: a zerank model, or an HF cross-encoder repo id. */
  model?: string;
  /** API key (hosted engine only). Default process.env.ZEROENTROPY_API_KEY. */
  apiKey?: string;
  /** Return only the top N after reranking. Default: all. */
  topN?: number;
  /**
   * Drop results below this relevance score. Meaningful because both engines
   * emit a 0..1 score (zerank's is calibrated; the local engine squashes its
   * logit through a sigmoid) — unlike uncalibrated vector distances.
   *
   * ⚠ Absolute thresholds are DTYPE-SPECIFIC on the local engine: quantization
   * preserves ordering but compresses the score range, so a floor fitted on
   * fp32 rejects too much at q8. Fit it on the dtype you deploy.
   */
  minScore?: number;
  /**
   * Instruction for an instruction-following reranker (zerank-2). The
   * ZeroEntropy SDK has no dedicated instruction field, so it's prepended to the
   * query — zerank-2 reads it as ranking guidance, e.g. "Rank the actual product
   * the shopper wants above parts/accessories for it." Default: none.
   *
   * IGNORED by the `local` engine: a plain cross-encoder is trained on
   * (query, passage) pairs and injecting instruction prose into the query
   * degrades its scoring rather than steering it.
   */
  instruction?: string;
}

export interface RerankResult<T> {
  /** Caller's payload, carried through untouched. */
  row: T;
  /** Relevance score (0..1), or 0 when reranking was skipped. */
  score: number;
  /** True when the score came from the reranker (vs. a fail-safe passthrough). */
  reranked: boolean;
}

/**
 * What every engine returns: scores index-aligned with the input docs.
 * - `null`      → the engine could not score at all; fail-safe passthrough.
 * - `undefined` at index i → this doc was not scored, and is dropped.
 */
type EngineScores = Array<number | undefined> | null;

let _client: ZeroEntropy | null = null;
let _clientKey: string | null = null;
function getClient(apiKey: string): ZeroEntropy {
  if (!_client || _clientKey !== apiKey) {
    _client = new ZeroEntropy({ apiKey });
    _clientKey = apiKey;
  }
  return _client;
}

/**
 * True if reranking is configured.
 *
 * The `local` engine needs no credential, so it is always configured — pass the
 * options object to ask about a specific engine. The legacy string form
 * (`rerankAvailable(apiKey)`) still asks the hosted-engine question.
 */
export function rerankAvailable(optsOrApiKey?: string | RerankOptions): boolean {
  const opts: RerankOptions = typeof optsOrApiKey === 'string' ? { apiKey: optsOrApiKey } : (optsOrApiKey ?? {});
  if (opts.engine === 'local') return true;
  return !!(opts.apiKey ?? process.env.ZEROENTROPY_API_KEY);
}

/** Hosted zerank. Returns index-aligned scores, or null to trigger passthrough. */
async function zeroEntropyScores<T>(
  query: string,
  docs: Array<RerankDoc<T>>,
  opts: RerankOptions,
): Promise<EngineScores> {
  const apiKey = opts.apiKey ?? process.env.ZEROENTROPY_API_KEY;
  if (!apiKey) return null;

  try {
    const res = await getClient(apiKey).models.rerank({
      model: opts.model ?? 'zerank-2',
      // zerank-2 instruction-following format (per ZeroEntropy docs): the
      // instruction is embedded in the query via XML tags, NOT a free-text prefix.
      query: opts.instruction ? `<query>${query}</query>\n<instruction>${opts.instruction}</instruction>` : query,
      documents: docs.map((d) => d.text),
    });
    const scores: Array<number | undefined> = new Array(docs.length).fill(undefined);
    for (const r of (res.results ?? []) as Array<{ index: number; relevance_score: number }>) {
      // Out-of-range indices from the API are dropped, not crashed on.
      if (docs[r.index] !== undefined) scores[r.index] = r.relevance_score;
    }
    return scores;
  } catch {
    return null;
  }
}

/** Local ONNX cross-encoder. Returns index-aligned scores, or null. */
async function localScores<T>(
  query: string,
  docs: Array<RerankDoc<T>>,
  opts: RerankOptions,
): Promise<EngineScores> {
  return await localEngineScores(query, docs.map((d) => d.text), opts);
}

/**
 * Reorder `docs` by cross-encoder relevance to `query`. Returns results sorted
 * best-first. On a missing credential, an unavailable engine, or any error,
 * returns `docs` in original order (reranked=false) so callers can always use
 * the result safely.
 */
export async function rerank<T>(
  query: string,
  docs: Array<RerankDoc<T>>,
  opts: RerankOptions = {},
): Promise<Array<RerankResult<T>>> {
  const passthrough = (): Array<RerankResult<T>> =>
    docs.slice(0, opts.topN ?? docs.length).map((d) => ({ row: d.row, score: 0, reranked: false }));

  if (!query.trim() || docs.length === 0) return passthrough();

  const scores =
    opts.engine === 'local' ? await localScores(query, docs, opts) : await zeroEntropyScores(query, docs, opts);
  if (!scores) return passthrough();

  let ordered = docs
    .map((d, i) => ({ row: d.row, score: scores[i], reranked: true }))
    .filter((r): r is { row: T; score: number; reranked: true } => typeof r.score === 'number')
    // Descending by score. Array.prototype.sort is stable, so equal scores keep
    // their retrieval order — the right tiebreak, since retrieval order already
    // encodes the first-stage ranking.
    .sort((a, b) => b.score - a.score);

  if (typeof opts.minScore === 'number') ordered = ordered.filter((r) => r.score >= opts.minScore!);
  if (typeof opts.topN === 'number') ordered = ordered.slice(0, opts.topN);
  // Guard: if shaping left nothing usable, fall back rather than drop results.
  return ordered.length > 0 ? ordered : passthrough();
}
