/**
 * @papercusp/rerank — engine-agnostic cross-encoder reranking (ZeroEntropy zerank).
 *
 * Stage B of the retrieve→rerank pipeline: given a query and candidate docs
 * (from ANY retriever — Typesense, Postgres, pgvector, …), reorder them by a
 * cross-encoder's calibrated relevance score. Has NO project-specific deps, so
 * it can be lifted into papercusp (or promoted to a shared package) unchanged —
 * the only per-project differences are the config (model, topN, threshold).
 *
 * Fail-safe by design: if the API key is missing or the rerank call errors, it
 * returns the candidates in their ORIGINAL order — a rerank outage degrades to
 * "retrieval order", it never breaks search.
 */
import ZeroEntropy from 'zeroentropy';

export interface RerankDoc<T> {
  /** Stable id (for caching / debugging). */
  id: string;
  /** The text the reranker scores against the query (e.g. product title). */
  text: string;
  /** Caller's payload, carried through untouched. */
  row: T;
}

export interface RerankOptions {
  /** ZeroEntropy model. Default 'zerank-2'. */
  model?: string;
  /** API key. Default process.env.ZEROENTROPY_API_KEY. */
  apiKey?: string;
  /** Return only the top N after reranking. Default: all. */
  topN?: number;
  /**
   * Drop results below this calibrated relevance score (zerank scores are
   * calibrated 0..1, so a threshold here is meaningful — unlike uncalibrated
   * vector distances). Default: keep all.
   */
  minScore?: number;
  /**
   * Instruction for an instruction-following reranker (zerank-2). The
   * ZeroEntropy SDK has no dedicated instruction field, so it's prepended to the
   * query — zerank-2 reads it as ranking guidance, e.g. "Rank the actual product
   * the shopper wants above parts/accessories for it." Default: none.
   */
  instruction?: string;
  /**
   * Hard cap (ms) on the rerank API call. zerank latency is variable — usually
   * fast, occasionally slow, and it can hang outright; without a cap a hung call
   * hangs the whole search ("never finished"). On timeout we fall back to
   * retrieval order (a rerank outage must only ever cost ranking quality, never
   * a response). `0`/undefined → use the default (2500ms).
   */
  timeoutMs?: number;
}

/** Default hard cap on the rerank call (ms). Override per-call via opts.timeoutMs
 *  (Restart wires it behind RERANK_TIMEOUT_MS). */
export const DEFAULT_RERANK_TIMEOUT_MS = 2500;

/** Reject `p` if it hasn't settled within `ms`; the caller treats a timeout the
 *  same as any rerank error → retrieval-order passthrough. The underlying request
 *  is left to settle and be GC'd (the ZeroEntropy SDK exposes no abort signal). */
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  if (!(ms > 0)) return p;
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`rerank timed out after ${ms}ms`)), ms);
    p.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

export interface RerankResult<T> {
  row: T;
  /** Calibrated relevance score (0..1), or 0 when reranking was skipped. */
  score: number;
  /** True when the score came from the reranker (vs. a fail-safe passthrough). */
  reranked: boolean;
}

let _client: ZeroEntropy | null = null;
let _clientKey: string | null = null;
function getClient(apiKey: string): ZeroEntropy {
  if (!_client || _clientKey !== apiKey) {
    _client = new ZeroEntropy({ apiKey });
    _clientKey = apiKey;
  }
  return _client;
}

/** True if reranking is configured (an API key is available). */
export function rerankAvailable(apiKey?: string): boolean {
  return !!(apiKey ?? process.env.ZEROENTROPY_API_KEY);
}

/**
 * Reorder `docs` by cross-encoder relevance to `query`. Returns results sorted
 * best-first. On missing key or any error, returns `docs` in original order
 * (reranked=false) so callers can always use the result safely.
 */
export async function rerank<T>(
  query: string,
  docs: Array<RerankDoc<T>>,
  opts: RerankOptions = {},
): Promise<Array<RerankResult<T>>> {
  const apiKey = opts.apiKey ?? process.env.ZEROENTROPY_API_KEY;
  const passthrough = (): Array<RerankResult<T>> =>
    docs.slice(0, opts.topN ?? docs.length).map((d) => ({ row: d.row, score: 0, reranked: false }));

  if (!apiKey || !query.trim() || docs.length === 0) return passthrough();

  try {
    const res = await withTimeout(
      getClient(apiKey).models.rerank({
        model: opts.model ?? 'zerank-2',
        // zerank-2 instruction-following format (per ZeroEntropy docs): the
        // instruction is embedded in the query via XML tags, NOT a free-text prefix.
        query: opts.instruction ? `<query>${query}</query>\n<instruction>${opts.instruction}</instruction>` : query,
        documents: docs.map((d) => d.text),
      }),
      opts.timeoutMs ?? DEFAULT_RERANK_TIMEOUT_MS,
    );
    let ordered = (res.results ?? [])
      .filter((r: { index: number }) => docs[r.index] !== undefined)
      .map((r: { index: number; relevance_score: number }) => ({
        row: docs[r.index].row,
        score: r.relevance_score,
        reranked: true,
      }));
    if (typeof opts.minScore === 'number') ordered = ordered.filter((r) => r.score >= opts.minScore!);
    if (typeof opts.topN === 'number') ordered = ordered.slice(0, opts.topN);
    // Guard: if the API returned nothing usable, fall back rather than drop results.
    return ordered.length > 0 ? ordered : passthrough();
  } catch {
    return passthrough();
  }
}
