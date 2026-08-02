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
  activeExecutionTarget,
  loadCrossEncoder,
  localRerankAvailable,
  scoreCrossEncoder,
  type LocalRerankOptions,
} from './local-engine';

export {
  CPU_EXECUTION_TARGET,
  CUDA_EXECUTION_TARGET,
  RERANK_DEVICE_ENV,
  WEBGPU_EXECUTION_TARGET,
  executionTargetFor,
  isCoherentTarget,
  resolveExecutionTarget,
  type RerankDevice,
  type RerankExecutionTarget,
} from './execution-target';

export {
  DEFAULT_SIDECAR_MAX_ATTEMPTS,
  DEFAULT_SIDECAR_TIMEOUT_MS,
  RERANK_SIDECAR_URL_ENV,
  SIDECAR_MAX_TEXT_CHARS,
  SidecarRerankHttpError,
  buildSidecarFirstReranker,
  isNonRetryableSidecarError,
  resolveRerankSidecarUrl,
  sidecarRerankBatch,
  type RerankScoreFn,
  type SidecarFirstRerankerOpts,
  type SidecarRerankResponse,
} from './sidecar-reranker';

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
   * Scoring function override for `engine: 'local'` — returns scores
   * index-aligned with the docs, and MAY throw.
   *
   * This is how a sidecar-backed scorer (buildSidecarFirstReranker) reaches the
   * ranking logic without a third engine branch: whatever it throws is caught
   * by the SAME fail-safe seam as an in-process failure, so a sidecar outage
   * degrades to retrieval order exactly like a missing ONNX runtime does.
   */
  scorer?: (query: string, texts: string[]) => Promise<number[]>;
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
  /**
   * Wall-clock bound on SCORING. On expiry the rerank degrades through the
   * SAME fail-safe passthrough as any other scoring failure — it never throws
   * and never stalls the caller. Default {@link DEFAULT_RERANK_TIMEOUT_MS};
   * `0`/`Infinity` disables it.
   *
   * ⚠ This is a BACKSTOP against a hang, deliberately NOT a UX budget. It sits
   * ABOVE the sidecar client's own 15s retry budget (DEFAULT_SIDECAR_TIMEOUT_MS)
   * so it can never pre-empt that retry policy and convert a recoverable blip
   * into a silent degradation. A user-facing caller wanting search-grade
   * latency must pass its OWN, much tighter value — the right bound depends on
   * the caller's latency contract, which this library cannot know.
   *
   * The losing promise is not cancellable (neither ONNX inference nor an
   * in-flight fetch aborts here), so it runs to completion in the background
   * and its result is discarded; a late rejection is swallowed rather than
   * surfacing as an unhandled rejection after we have already degraded.
   */
  timeoutMs?: number;
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

/** Local cross-encoder — in-process by default, or an injected `scorer` (e.g.
 *  the sidecar client). Returns index-aligned scores, or null to fail safe. */
async function localScores<T>(
  query: string,
  docs: Array<RerankDoc<T>>,
  opts: RerankOptions,
): Promise<EngineScores> {
  const texts = docs.map((d) => d.text);
  if (!opts.scorer) return await localEngineScores(query, texts, opts);
  try {
    const scores = await opts.scorer(query, texts);
    // A length mismatch would misalign every score with its document, so treat
    // it as a failure rather than ranking on garbage.
    if (!Array.isArray(scores) || scores.length !== texts.length) return null;
    return scores.map((s) => (typeof s === 'number' && Number.isFinite(s) ? s : undefined));
  } catch {
    return null;
  }
}

/**
 * Reorder `docs` by cross-encoder relevance to `query`. Returns results sorted
 * best-first. On a missing credential, an unavailable engine, or any error,
 * returns `docs` in original order (reranked=false) so callers can always use
 * the result safely.
 */
/**
 * Backstop bound on scoring (ms). Deliberately ABOVE the sidecar client's own
 * 15s retry budget so it never truncates that retry policy — this catches a
 * genuine HANG (a wedged in-process engine, a scorer that never settles), not
 * a slow-but-progressing rerank. Callers with a latency contract (a user-facing
 * search route) pass their own, tighter `timeoutMs`.
 */
export const DEFAULT_RERANK_TIMEOUT_MS = 20_000;

/** Distinguishes "the timer won" from a legitimate `null` scoring result. */
const SCORE_TIMEOUT = Symbol('rerank-score-timeout');

/**
 * Resolve `scores`, or `null` if `timeoutMs` elapses first — `null` is exactly
 * what the engines already return when they cannot score, so a timeout reuses
 * the single fail-safe passthrough instead of adding a second degradation path.
 */
async function withScoreTimeout(scores: Promise<EngineScores>, timeoutMs: number): Promise<EngineScores> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return scores;

  // The race loser keeps running (inference/fetch are not cancellable here). If
  // it later REJECTS with nobody awaiting it, Node reports an unhandled
  // rejection — attach the sink now, before racing, so there is no window.
  scores.catch(() => {});

  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const winner = await Promise.race([
      scores,
      new Promise<typeof SCORE_TIMEOUT>((resolve) => {
        timer = setTimeout(() => resolve(SCORE_TIMEOUT), timeoutMs);
      }),
    ]);
    return winner === SCORE_TIMEOUT ? null : winner;
  } finally {
    // Always clear: a surviving timer would hold the event loop open for up to
    // timeoutMs after a fast rerank already returned.
    if (timer !== undefined) clearTimeout(timer);
  }
}

export async function rerank<T>(
  query: string,
  docs: Array<RerankDoc<T>>,
  opts: RerankOptions = {},
): Promise<Array<RerankResult<T>>> {
  const passthrough = (): Array<RerankResult<T>> =>
    docs.slice(0, opts.topN ?? docs.length).map((d) => ({ row: d.row, score: 0, reranked: false }));

  if (!query.trim() || docs.length === 0) return passthrough();

  const scores = await withScoreTimeout(
    opts.engine === 'local' ? localScores(query, docs, opts) : zeroEntropyScores(query, docs, opts),
    opts.timeoutMs ?? DEFAULT_RERANK_TIMEOUT_MS,
  );
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
