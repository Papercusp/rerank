/**
 * Local ONNX cross-encoder engine for @papercusp/rerank.
 *
 * The lib's original engine calls a hosted API (ZeroEntropy zerank). This one
 * runs the cross-encoder IN-PROCESS via Transformers.js/ONNX Runtime, so
 * reranking costs no API key, no network hop, and no per-call spend — which is
 * what makes it usable on every search, not just the ones someone paid for.
 *
 * Contract: identical to every other engine here — index-aligned scores, and
 * `null` for "I could not score this" so the caller degrades to retrieval order
 * rather than breaking search. Nothing in this file may throw to its caller.
 *
 * Model shape it expects: a sequence-classification cross-encoder that emits
 * ONE logit per (query, doc) pair — `ModernBertForSequenceClassification` with
 * num_labels=1, e.g. `Alibaba-NLP/gte-reranker-modernbert-base`. Two-label
 * cross-encoders are also handled (softmax, take the "relevant" class), so the
 * engine is not welded to one architecture.
 *
 * Deliberately NOT here (they belong to later plan items, not to the engine):
 * an HTTP sidecar with one warm model per host, and host-aware dtype selection.
 * This file only has to be correct and fail-safe in-process.
 */

import {
  CPU_EXECUTION_TARGET,
  type RerankDevice,
  type RerankExecutionTarget,
  demotedTarget,
  resolveExecutionTarget,
} from './execution-target';
import { getRerankWorkerState, scoreViaWorker, warnRerankFallback } from './local-reranker-worker';
import { isDeadlineFailure, runInScorerGate } from './scorer-gate';

/** Cross-encoder that emits one logit per pair. num_labels=1, ModernBERT. */
export const LOCAL_RERANKER_MODEL = 'Alibaba-NLP/gte-reranker-modernbert-base';

/**
 * Fast tier: ~7.5x cheaper per pair (23M params / 6 layers vs 149M), at a real
 * quality cost. Measured, not assumed — see the plan's D-001. Use it where the
 * candidate pool is large or the latency budget is tight.
 */
export const FAST_RERANKER_MODEL = 'Xenova/ms-marco-MiniLM-L-6-v2';

/**
 * ONNX Runtime defaults intraOp threads to EVERY core and spin-waits them — on
 * a many-core host that grows a huge spin pool per loading process and stutters
 * the whole box (the same trap the local embedder hit, WI-3792). Cap it.
 *
 * Capping costs us nothing here: D-001 MEASURED that cross-encoder latency on
 * this workload does not improve with more cores (4 pinned cores matched a
 * 128-core box), so per-pair cost is bounded by the candidate cap, not by
 * threads. Kept identical to the embedder's cap so the two can share a host.
 */
export const ORT_SESSION_OPTIONS = { intraOpNumThreads: 4, interOpNumThreads: 1 } as const;

/**
 * q8 is the CPU-correct default, MEASURED rather than assumed (plan D-006): on
 * 4 pinned cores it is 1.60x faster per pair than fp32 and never changed the
 * top result (top-1 agreement 1.000 across 6 queries, NDCG@5 0.9855).
 *
 * fp16/q4f16 are GPU formats. On the CPU provider fp16 is not merely slower —
 * the graph FAILS TO INITIALISE outright (ORT's precision-free-cast insertion
 * breaks on ModernBERT's SimplifiedLayerNormFusion), so the engine's fail-safe
 * would silently drop reranking entirely. That is why dtype is chosen as a PAIR
 * with the execution provider; see `execution-target.ts`.
 *
 * Prefer `resolveExecutionTarget()` over this constant — it is retained for the
 * callers that legitimately pin a dtype (tests, benchmarks).
 */
export const DEFAULT_RERANK_DTYPE = 'q8';

/** Pairs are truncated to this many tokens. The model supports 8192, but a
 *  cross-encoder's cost is superlinear in sequence length and a reranker scores
 *  a match-centred passage, not a whole document — 512 is the useful window. */
export const DEFAULT_MAX_LENGTH = 512;

/** Pairs per forward pass. Padding is to the longest member of the batch, so a
 *  huge batch wastes compute on padding; 16 keeps memory flat and padding cheap. */
export const DEFAULT_BATCH_SIZE = 16;

export interface LocalRerankOptions {
  /** HF model id. Default `LOCAL_RERANKER_MODEL`. */
  model?: string;
  /**
   * ONNX weight format. Defaults WITH `device` as a coherent pair (see
   * `execution-target.ts`) — do not set this alone unless you mean to: `fp16`
   * without a GPU `device` is not a slow configuration, it is an unloadable one.
   */
  dtype?: string;
  /**
   * ONNX Runtime execution provider. Defaults WITH `dtype` as a pair. Passing a
   * GPU device does not force it: the engine verifies the session actually
   * constructs and falls back to the whole CPU pair if it does not.
   */
  device?: RerankDevice;
  /** Truncate each (query, doc) pair to this many tokens. Default 512. */
  maxLength?: number;
  /** Pairs per forward pass. Default 16. */
  batchSize?: number;
  /**
   * Absolute epoch-ms instant after which scoring STOPS, checked between
   * batches. Throws on expiry, which `localEngineScores` converts to the same
   * fail-safe `null` as any other scoring failure — so an over-budget rerank
   * degrades to retrieval order rather than running to completion unwatched.
   *
   * Distinct from the caller's `timeoutMs` and NOT a replacement for it. That
   * one bounds how long the CALLER waits; this one stops the WORK. Both are
   * needed because the losing side of a `Promise.race` is not cancellable: with
   * only the race, an abandoned rerank keeps burning a core for the rest of its
   * natural life, on the very host (no sidecar ⇒ in-process) least able to
   * spare one. `rerank()` derives this from `timeoutMs` automatically.
   */
  deadline?: number;
}

// --- Minimal structural types for the optional dependency ------------------
// Typed structurally rather than imported, so this lib does not take a hard
// build-time dependency on @huggingface/transformers.

interface LogitsTensor {
  dims: number[];
  data: ArrayLike<number>;
}
type Tokenizer = (
  text: string[],
  opts: { text_pair: string[]; padding: boolean; truncation: boolean; max_length: number },
) => Record<string, unknown>;
type SequenceClassifier = (inputs: Record<string, unknown>) => Promise<{ logits: LogitsTensor }>;

interface TransformersModule {
  AutoTokenizer: { from_pretrained(model: string, opts?: Record<string, unknown>): Promise<Tokenizer> };
  AutoModelForSequenceClassification: {
    from_pretrained(model: string, opts?: Record<string, unknown>): Promise<SequenceClassifier>;
  };
}

const TRANSFORMERS_PACKAGE = '@huggingface/transformers';

// Dodge bundler static analysis so the optional @huggingface dependency is only
// required when the local engine is actually selected. Same idiom as the local
// embedder — a static import would force every consumer of this lib (including
// ones that only use the hosted engine) to install ONNX Runtime.
const dynamicImport = new Function('specifier', 'return import(specifier)') as <T>(
  specifier: string,
) => Promise<T>;

interface LoadedModel {
  tokenizer: Tokenizer;
  model: SequenceClassifier;
}

/** One warm model per (id, dtype). Loading is seconds; scoring is milliseconds,
 *  so the cache is the difference between usable and unusable. */
const _models = new Map<string, Promise<LoadedModel>>();

/** Injected in tests so the engine's own logic is testable without ONNX. */
let _loaderOverride: ((model: string, dtype: string, device: RerankDevice) => Promise<LoadedModel>) | null =
  null;

/**
 * Test-only: permit the WORKER path even while a loader override is installed.
 *
 * Normally an injected loader forces the inline path (it is an in-process object
 * and cannot cross a thread boundary). That rule makes the worker→inline
 * FALLBACK unreachable from tests, because the override short-circuits before
 * the worker is ever tried — so this seam exists purely to exercise "the worker
 * failed, did we degrade correctly?" against a fake model. Production never
 * installs a loader override, so this combination cannot occur outside tests.
 */
let _workerWithLoaderForTest = false;
export function _setWorkerWithLoaderForTest(enabled: boolean): void {
  _workerWithLoaderForTest = enabled;
}

async function defaultLoader(model: string, dtype: string, device: RerankDevice): Promise<LoadedModel> {
  const transformers = await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
  const [tokenizer, classifier] = await Promise.all([
    transformers.AutoTokenizer.from_pretrained(model),
    transformers.AutoModelForSequenceClassification.from_pretrained(model, {
      dtype,
      device,
      session_options: ORT_SESSION_OPTIONS,
    }),
  ]);
  return { tokenizer, model: classifier };
}

/**
 * Load (and cache) the cross-encoder. Exported so a warm-model host can preload
 * it at startup instead of paying the load on a user's first search.
 *
 * Rejects on failure — and de-caches, so a transient failure (a half-downloaded
 * model, a cold network) does not poison the process for its whole lifetime.
 */
export function loadCrossEncoder(opts: LocalRerankOptions = {}): Promise<LoadedModel> {
  return loadOnTarget(opts.model ?? LOCAL_RERANKER_MODEL, effectiveTarget(opts));
}

/**
 * Resolve the (device, dtype) PAIR for a call.
 *
 * The pair is the unit: when the caller says nothing we take the host's whole
 * target, never one field from the host and one from a constant. An explicit
 * `dtype`/`device` is still honoured (tests and experiments need to pin one),
 * but it can only ever narrow a coherent starting point — it cannot silently
 * produce `fp16` on the CPU provider by defaulting the other half.
 */
function effectiveTarget(opts: LocalRerankOptions): RerankExecutionTarget {
  const host = resolveExecutionTarget();
  if (opts.device === undefined && opts.dtype === undefined) return host;
  const device = opts.device ?? host.device;
  const dtype = opts.dtype ?? host.dtype;
  return { device, dtype, why: 'explicit caller override' };
}

/**
 * Has the GPU target been proven to work on this host? A GPU that is present is
 * not a GPU that is usable — the CUDA provider fails to load without cuDNN, and
 * the fp16 graph fails to initialise on the CPU provider. We therefore find out
 * by CONSTRUCTING a session, and remember the answer so a broken GPU host pays
 * the failed load once rather than on every search.
 */
let _gpuUnusable = false;

/**
 * The demotion itself, REMEMBERED rather than only logged.
 *
 * A `console.warn` fires once per process and is then gone: it cannot be read
 * by a health check, a test, or an operator asking "why is search slow?" three
 * weeks later. Keeping the verdict here is what lets `activeExecutionTarget()`
 * and `rerankExecutionHealth()` distinguish a demoted host from a deliberate
 * CPU one — see `demotedTarget` for why that distinction is load-bearing.
 */
let _demotion: { from: RerankExecutionTarget; cause: string; target: RerankExecutionTarget } | null =
  null;

/** Reported once per process when a GPU target is demoted, so the fallback is
 *  visible in a log instead of silently costing quality. */
function reportDemotion(target: RerankExecutionTarget, err: unknown): void {
  const cause = String((err as Error)?.message ?? err).slice(0, 200);
  _demotion = { from: target, cause, target: demotedTarget(target, cause) };
  console.warn(
    `[rerank] ${target.device}/${target.dtype} session failed to construct — falling back to ` +
      `${CPU_EXECUTION_TARGET.device}/${CPU_EXECUTION_TARGET.dtype}. ` +
      `Reranking continues on CPU. Cause: ${cause}`,
  );
}

/**
 * Load on `target`, falling back to the CPU PAIR — both fields together — if a
 * non-CPU target cannot construct a session.
 *
 * Falling back on `device` alone would leave `dtype: 'fp16'` pointed at the CPU
 * provider, which is the single worst configuration available and is exactly
 * what a naive "GPU didn't work, use the CPU" retry produces.
 */
function loadOnTarget(model: string, target: RerankExecutionTarget): Promise<LoadedModel> {
  const effective = target.device !== 'cpu' && _gpuUnusable ? CPU_EXECUTION_TARGET : target;
  const key = `${model}::${effective.dtype}::${effective.device}`;

  const cached = _models.get(key);
  if (cached) return cached;

  const loading = (_loaderOverride ?? defaultLoader)(model, effective.dtype, effective.device).catch(
    (err) => {
      _models.delete(key);
      if (effective.device === 'cpu') throw err;
      // Demote the WHOLE pair, once, then retry on CPU.
      _gpuUnusable = true;
      reportDemotion(effective, err);
      return loadOnTarget(model, CPU_EXECUTION_TARGET);
    },
  );
  _models.set(key, loading);
  return loading;
}

/** The pair this host will actually run on, after any verified demotion. Exposed
 *  so a warm-model host / health check can REPORT the real configuration rather
 *  than the requested one. */
export function activeExecutionTarget(): RerankExecutionTarget {
  const host = resolveExecutionTarget();
  if (host.device === 'cpu' || !_gpuUnusable) return host;
  // Report the DEMOTED pair, not the bare CPU constant: same device/dtype, but a
  // `why` that says a GPU was asked for and refused. Falling back to the plain
  // constant here is what made a degraded host unreadable from a healthy one.
  return _demotion?.target ?? CPU_EXECUTION_TARGET;
}

/**
 * Requested vs actual execution target, and whether the gap is a demotion.
 *
 * `activeExecutionTarget()` alone answers "what is running"; it cannot answer
 * "is that what we asked for?" without the caller separately resolving the host
 * target and comparing — which nothing in this repo did, so a GPU host that
 * quietly reverted to CPU looked exactly like a CPU host. This returns both
 * sides plus the verdict, so the comparison cannot be forgotten.
 *
 * `demoted: false` before the first load is HONEST, not a clean bill: the GPU
 * session is only verified by constructing one. Read it after a load.
 */
export interface RerankExecutionHealth {
  /** What this host asked for (`PAPERCUSP_RERANK_DEVICE`). */
  requested: RerankExecutionTarget;
  /** What it will actually run on, after any verified demotion. */
  active: RerankExecutionTarget;
  /** True once a requested GPU target has failed to construct a session. */
  demoted: boolean;
  /** The construction error that forced the demotion; null when not demoted. */
  demotionCause: string | null;
}

export function rerankExecutionHealth(): RerankExecutionHealth {
  return {
    requested: resolveExecutionTarget(),
    active: activeExecutionTarget(),
    demoted: _demotion !== null,
    demotionCause: _demotion?.cause ?? null,
  };
}

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

/**
 * Collapse a row of logits to one relevance score.
 *
 * - 1 logit  → sigmoid. The single-logit cross-encoder convention.
 * - N logits → softmax, take class 1 ("relevant" by the binary cross-encoder
 *   convention). Only the 0/1 pair matters, so softmax over those two is both
 *   correct for N=2 and a sane reading for N>2.
 *
 * Both are MONOTONIC in the underlying logit, so ordering is unaffected by the
 * squashing — which matters because ordering is the only thing quantization
 * preserves exactly (D-001: q8 keeps the ranking but compresses the score
 * range, so any absolute threshold must be fitted on the deployed dtype).
 */
function scoreFromLogits(row: number[]): number {
  if (row.length === 1) return sigmoid(row[0]);
  const [negative, positive] = row;
  return sigmoid(positive - negative);
}

/**
 * Hand the event loop a turn.
 *
 * `setImmediate` fires in the CHECK phase, which the loop reaches only after
 * passing through the TIMERS phase — so this is what lets an armed `setTimeout`
 * (every wall-clock bound in this lib is one) actually run. `await`-ing an
 * already-resolved promise would NOT do it: that only queues a microtask, and
 * the microtask queue is drained to empty before the loop advances a phase, so
 * a loop of `await` over synchronous work starves timers exactly as a bare
 * `while` loop would. That distinction IS the bug this yield fixes.
 *
 * Falls back to `setTimeout(0)` where `setImmediate` is absent (browser/worker
 * builds); this lib is deliberately environment-agnostic.
 */
function yieldToEventLoop(): Promise<void> {
  return new Promise<void>((resolve) => {
    if (typeof setImmediate === 'function') setImmediate(resolve);
    else setTimeout(resolve, 0);
  });
}

/**
 * Score every (query, text) pair. Returns scores index-aligned with `texts`.
 *
 * Throws on load/inference failure — the fail-safe conversion to "return the
 * input order untouched" happens in the engine wrapper below, so a caller that
 * wants to SEE the failure (a warm-up probe, a health check, the sidecar) can.
 *
 * ⚠ COOPERATIVE WHEN INLINE (EI-20005411672741677). A cross-encoder forward pass
 * is CPU-bound and holds the thread, so while it runs on the MAIN thread nothing
 * else in this process makes progress — including the `setTimeout` behind every
 * one of this lib's wall-clock bounds. A timer cannot preempt it; there is no
 * third option between "the work yields" and "the work runs elsewhere". So the
 * inline loop yields between batches and re-checks its deadline there, bounding
 * the caller's wait to at most ONE batch of overrun instead of the whole stage.
 *
 * The preferred path takes the third option: `scoreCrossEncoder` dispatches to a
 * WORKER THREAD (`local-reranker-worker.ts`, WI-37555), which makes the caller's
 * bound EXACT rather than bounded-to-one-batch. This inline function remains the
 * fallback for hosts where a worker cannot spawn — degraded, never broken.
 */
export async function scoreCrossEncoderInline(
  query: string,
  texts: string[],
  opts: LocalRerankOptions = {},
): Promise<number[]> {
  if (texts.length === 0) return [];

  const { tokenizer, model } = await loadCrossEncoder(opts);
  const maxLength = opts.maxLength ?? DEFAULT_MAX_LENGTH;
  const batchSize = Math.max(1, opts.batchSize ?? DEFAULT_BATCH_SIZE);
  const { deadline } = opts;

  const scores: number[] = [];
  for (let start = 0; start < texts.length; start += batchSize) {
    // Before each pass, not after: the yield has to precede the thread-holding
    // work for a bound armed by the caller to have any chance of firing, and
    // the deadline is only meaningful once the loop has actually been given a
    // turn to notice time passing.
    await yieldToEventLoop();
    if (deadline !== undefined && Date.now() >= deadline) {
      throw new Error(
        `rerank: scoring deadline exceeded after ${start} of ${texts.length} pairs — degrading to retrieval order`,
      );
    }

    const batch = texts.slice(start, start + batchSize);
    const inputs = tokenizer(
      // The cross-encoder scores a PAIR: the query is repeated against each doc.
      new Array(batch.length).fill(query),
      { text_pair: batch, padding: true, truncation: true, max_length: maxLength },
    );
    const { logits } = await model(inputs);

    // dims is [batch, numLabels]; a 1-D [batch] export means one logit per pair.
    const numLabels = logits.dims.length > 1 ? logits.dims[logits.dims.length - 1] : 1;
    if (logits.data.length !== batch.length * numLabels) {
      throw new Error(
        `rerank: logits length ${logits.data.length} != ${batch.length} pairs x ${numLabels} labels`,
      );
    }
    for (let i = 0; i < batch.length; i++) {
      const row: number[] = [];
      for (let l = 0; l < numLabels; l++) row.push(Number(logits.data[i * numLabels + l]));
      scores.push(scoreFromLogits(row));
    }
  }
  return scores;
}

/**
 * Score every (query, text) pair, PREFERRING a worker thread.
 *
 * This is the exported entry point every caller already used; the dispatch is
 * deliberately invisible to them, so the sidecar, the warm-model host and
 * `localEngineScores` all get the exact bound with no wiring change.
 *
 * Two cases take the inline path, both on purpose:
 *  - a `_setLoaderForTest` override is installed. An injected loader is an
 *    in-process JavaScript object; it cannot cross a thread boundary, so
 *    routing it to the worker would not merely fail, it would silently start
 *    exercising the REAL ONNX model in tests that believe they injected a fake.
 *  - the worker is unavailable or failed. Degraded (bound accurate only to one
 *    batch), never broken.
 */
export async function scoreCrossEncoder(
  query: string,
  texts: string[],
  opts: LocalRerankOptions = {},
): Promise<number[]> {
  if (texts.length === 0) return [];

  // ONE shared scorer, so admission is decided here rather than per engine path:
  // both the worker and the inline fallback contend for the same CPU, and this
  // is the single funnel they both pass through. See `scorer-gate.ts` for the
  // measurement — concurrency buys no throughput, so serializing costs nothing
  // and lets a call that provably cannot meet its budget shed at ~0ms instead of
  // spending the whole budget discovering it (WI-37676).
  return await runInScorerGate(texts.length, opts.deadline, () =>
    scoreCrossEncoderUngated(query, texts, opts),
  );
}

async function scoreCrossEncoderUngated(
  query: string,
  texts: string[],
  opts: LocalRerankOptions,
): Promise<number[]> {
  if ((_loaderOverride === null || _workerWithLoaderForTest) && !getRerankWorkerState().disabled) {
    // Resolve the (device, dtype) pair HERE — including any verified GPU
    // demotion — so that logic lives in one place rather than being duplicated
    // across the thread seam.
    const target = effectiveTarget(opts);
    const effective = target.device !== 'cpu' && _gpuUnusable ? CPU_EXECUTION_TARGET : target;
    try {
      return await scoreViaWorker({
        query,
        texts,
        model: opts.model ?? LOCAL_RERANKER_MODEL,
        dtype: effective.dtype,
        device: effective.device,
        maxLength: opts.maxLength ?? DEFAULT_MAX_LENGTH,
        batchSize: Math.max(1, opts.batchSize ?? DEFAULT_BATCH_SIZE),
        deadline: opts.deadline,
      });
    } catch (err) {
      // A deadline expiry is a real verdict from the worker, not a worker
      // fault: re-running it inline would burn the very budget that just
      // expired, on the main thread, which is the opposite of the fix.
      if (isDeadlineFailure(err)) throw err;
      warnRerankFallback(err);
    }
  }

  return await scoreCrossEncoderInline(query, texts, opts);
}

/**
 * The engine seam: score `texts`, or return `null` if anything at all goes
 * wrong. `null` is the lib's fail-safe signal — the caller keeps retrieval
 * order. A reranker outage must never be able to break search, and this
 * function is the boundary where that guarantee is enforced.
 */
export async function localEngineScores(
  query: string,
  texts: string[],
  opts: LocalRerankOptions = {},
): Promise<number[] | null> {
  try {
    return await scoreCrossEncoder(query, texts, opts);
  } catch {
    return null;
  }
}

/** True when a local rerank is possible at all — i.e. the optional ONNX
 *  dependency is installed. Unlike the hosted engine, no key is involved. */
export async function localRerankAvailable(): Promise<boolean> {
  try {
    await dynamicImport<TransformersModule>(TRANSFORMERS_PACKAGE);
    return true;
  } catch {
    return false;
  }
}

/** Test seam — inject a fake loader and drop every warm model.
 *
 *  Also clears the remembered GPU verdict: it is process-lifetime state by
 *  design (a broken GPU host must not re-pay a failed load per search), which
 *  would otherwise leak between tests and make a demotion test pass only when
 *  it happened to run first. */
export function _setLoaderForTest(
  loader: ((model: string, dtype: string, device: RerankDevice) => Promise<LoadedModel>) | null,
): void {
  _loaderOverride = loader;
  _models.clear();
  _gpuUnusable = false;
  _demotion = null;
  // Reset the worker-dispatch escape too, so a test that enabled it cannot leak
  // the worker path into an unrelated later test via the shared afterEach.
  _workerWithLoaderForTest = false;
}
