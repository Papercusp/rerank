/**
 * Deadline-aware admission control for the shared in-process cross-encoder.
 *
 * WHY (WI-37676). Every wall-clock bound in this library is PER-CALL, but the
 * in-process scorer is ONE shared resource. Measured on this box 2026-08-10,
 * over the real worker path, at 24 and at 100 pairs:
 *
 *     concurrency   1      2      4      8
 *     per-call     1.00x  2.04x  3.98x  7.68x   (of the solo wall clock)
 *     throughput   30.9   30.3   31.1   32.2    pairs/s
 *
 * Per-call latency is EXACTLY linear in the number of callers, and aggregate
 * throughput is flat. Two consequences, and the second is the whole point:
 *
 *  1. Concurrency buys NOTHING here. Admitting N callers at once does not
 *     finish the work sooner; it only spreads the same completion time over
 *     everyone. So serializing costs no throughput.
 *  2. Un-admitted concurrency makes failure CORRELATED. Once N exceeds
 *     budget/solo (~5 at the shipped 4s bound), every in-flight call blows its
 *     budget at roughly the same moment and the whole stage degrades at once —
 *     after each caller has spent its entire budget waiting. Nobody gets a
 *     reranked page and the CPU was burned producing none.
 *
 * This gate turns that into the strictly better outcome: callers that CAN make
 * their budget run one after another and succeed; a caller that provably cannot
 * is shed IMMEDIATELY, at ~0ms, with the same fail-safe passthrough it would
 * have reached the slow way. It gets the identical page, several seconds sooner.
 *
 * ⚠ SAFETY ARGUMENT FOR THE ESTIMATOR. The shed decision rests on a measured
 * cost estimate, which can be wrong in both directions — and neither direction
 * is worse than the status quo, which is what makes this safe to default ON:
 *  - over-shed  ⇒ a call that would have just squeaked in returns retrieval
 *                 order EARLY. Same rows, same order as the timeout it replaces.
 *  - under-shed ⇒ the call proceeds and may time out. Exactly today's behaviour.
 * The gate can therefore only move latency down and the success count up; it
 * cannot invent a failure that did not already exist.
 *
 * NOT for the sidecar's HTTP leg. This gates in-process scoring only — the
 * shared CPU. A sidecar call is someone else's capacity with its own retry
 * budget, and throttling it here would be throttling the wrong resource.
 */

/**
 * A scoring attempt abandoned because it could not meet its caller's deadline.
 *
 * Shaped to match the message the engine and the worker already throw, so the
 * existing `/deadline exceeded/` handling in `scoreCrossEncoder` — "this is a
 * real verdict, do NOT retry it inline" — applies to a shed unchanged. Retrying
 * a shed inline would burn the very budget that just proved insufficient, on the
 * main thread.
 */
export class RerankDeadlineError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RerankDeadlineError';
  }
}

/**
 * Did this failure mean "too slow for your budget" rather than "could not score"?
 *
 * Message matching is not laziness: the worker's throw crosses a thread
 * boundary, where a structured-clone of an Error keeps `message` and loses the
 * prototype, so `instanceof` is unavailable on exactly the path that matters
 * most. The class check is kept for the in-process throws it does work for.
 */
export function isDeadlineFailure(err: unknown): boolean {
  if (err instanceof RerankDeadlineError) return true;
  return err instanceof Error && /deadline exceeded/.test(err.message);
}

/** Env override for hosts where scoring genuinely parallelizes (a GPU target). */
export const RERANK_MAX_CONCURRENT_ENV = 'RERANK_MAX_CONCURRENT_SCORERS';

interface Waiter {
  pairs: number;
  deadline: number | undefined;
  admit: () => void;
  refuse: (err: Error) => void;
}

let _maxConcurrent: number | null = null;
let _active = 0;
let _activePairs = 0;
const _queue: Waiter[] = [];

/**
 * Exponentially-weighted mean ms per (query, doc) pair, measured from this
 * process's own completed jobs. `null` until calibrated — and while it is null
 * the gate NEVER sheds, because a shed decision taken from a guess is a made-up
 * failure. Serialization still applies; only the shedding waits for evidence.
 */
let _msPerPair: number | null = null;
let _observations = 0;

function maxConcurrent(): number {
  if (_maxConcurrent !== null) return _maxConcurrent;
  const raw = Number(process.env[RERANK_MAX_CONCURRENT_ENV]);
  _maxConcurrent = Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : 1;
  return _maxConcurrent;
}

/**
 * Cost of everything that must finish before a newly-arriving job of `pairs`
 * pairs could complete, in ms. Uses TOTAL pairs of in-flight work as an upper
 * bound on what remains: a job halfway done is charged as if it had not started.
 * That errs toward shedding, which is the harmless direction (see the header).
 */
function estimatedWaitMs(pairs: number): number | null {
  if (_msPerPair === null) return null;
  const backlog = _activePairs + _queue.reduce((n, w) => n + w.pairs, 0);
  return (backlog + pairs) * _msPerPair;
}

function wouldMissDeadline(pairs: number, deadline: number | undefined): boolean {
  if (deadline === undefined) return false;
  const wait = estimatedWaitMs(pairs);
  if (wait === null) return false;
  return Date.now() + wait > deadline;
}

function shedError(pairs: number, deadline: number): RerankDeadlineError {
  const wait = Math.round(estimatedWaitMs(pairs) ?? 0);
  const budget = Math.max(0, deadline - Date.now());
  return new RerankDeadlineError(
    `rerank: scoring deadline exceeded after 0 of ${pairs} pairs — the shared scorer is ` +
      `${_active} deep with ${_queue.length} queued, so this call needs ~${wait}ms against ` +
      `${budget}ms of budget. Shedding now rather than spending the budget waiting.`,
  );
}

/** Drain as many queued callers as the concurrency limit allows. */
function pump(): void {
  while (_queue.length > 0 && _active < maxConcurrent()) {
    const w = _queue.shift()!;
    // Re-check AT THE HEAD, not only on arrival: a caller admitted to the queue
    // when the backlog was short can become undeliverable while it waits, and
    // starting it then would spend a budget already known to be insufficient.
    if (w.deadline !== undefined && wouldMissDeadline(w.pairs, w.deadline)) {
      w.refuse(shedError(w.pairs, w.deadline));
      continue;
    }
    _active++;
    _activePairs += w.pairs;
    w.admit();
  }
}

/**
 * Run `fn` under the gate. Rejects with a {@link RerankDeadlineError} — without
 * ever calling `fn` — when the shared scorer provably cannot deliver in time.
 */
export async function runInScorerGate<T>(
  pairs: number,
  deadline: number | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  if (pairs <= 0) return await fn();

  if (deadline !== undefined && wouldMissDeadline(pairs, deadline)) {
    throw shedError(pairs, deadline);
  }

  await new Promise<void>((admit, refuse) => {
    _queue.push({ pairs, deadline, admit, refuse });
    pump();
  });

  const startedAt = Date.now();
  const soloRun = _active === 1;
  try {
    return await fn();
  } finally {
    const elapsed = Date.now() - startedAt;
    _active--;
    _activePairs -= pairs;
    // Calibrate only from a job that had the scorer to ITSELF — otherwise the
    // sample measures contention, not cost, and the estimator would inflate
    // under exactly the load it is meant to reason about.
    //
    // The FIRST observation is dropped: it pays this process's one-time model
    // load (~1.2s here), which is not a per-pair cost and would otherwise seed
    // the estimator an order of magnitude high and shed everything after it.
    if (soloRun && elapsed > 0) {
      _observations++;
      if (_observations > 1) {
        const sample = elapsed / pairs;
        _msPerPair = _msPerPair === null ? sample : _msPerPair * 0.7 + sample * 0.3;
      }
    }
    pump();
  }
}

/** Diagnostics; also the seam the tests assert the invariants through. */
export function getScorerGateState(): {
  active: number;
  queued: number;
  maxConcurrent: number;
  msPerPair: number | null;
  observations: number;
} {
  return {
    active: _active,
    queued: _queue.length,
    maxConcurrent: maxConcurrent(),
    msPerPair: _msPerPair,
    observations: _observations,
  };
}

/** Test seam: set the calibration directly instead of measuring it. */
export function _setScorerGateCalibrationForTest(msPerPair: number | null, observations = 2): void {
  _msPerPair = msPerPair;
  _observations = observations;
}

/** Test seam: override the concurrency limit (`null` ⇒ re-read the env). */
export function _setScorerGateConcurrencyForTest(n: number | null): void {
  _maxConcurrent = n;
}

/** Test seam — same function under the codebase's `_reset*` convention. */
export function _resetScorerGateForTest(): void {
  _maxConcurrent = null;
  _active = 0;
  _activePairs = 0;
  _queue.length = 0;
  _msPerPair = null;
  _observations = 0;
}
