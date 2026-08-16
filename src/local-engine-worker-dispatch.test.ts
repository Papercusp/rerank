/**
 * Dispatch rules for `scoreCrossEncoder`: when does scoring go to the WORKER,
 * when does it stay INLINE, and what happens when the worker fails (WI-37555).
 *
 * The worker module is mocked here on purpose — the subject under test is the
 * ENGINE'S DECISION, not the thread itself. The thread is covered for real, with
 * no mocks, in `local-reranker-worker.test.ts`; faking it there would test the
 * fake. Split that way, each file tests the thing it can actually observe.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const scoreViaWorker = vi.fn();
const getRerankWorkerState = vi.fn(() => ({ alive: true, disabled: false, pendingCount: 0 }));
const warnRerankFallback = vi.fn();

vi.mock('./local-reranker-worker', () => ({
  scoreViaWorker: (...args: unknown[]) => scoreViaWorker(...args),
  getRerankWorkerState: () => getRerankWorkerState(),
  warnRerankFallback: (...args: unknown[]) => warnRerankFallback(...args),
}));

const {
  _setLoaderForTest,
  _setWorkerWithLoaderForTest,
  scoreCrossEncoder,
  DEFAULT_BATCH_SIZE,
  DEFAULT_MAX_LENGTH,
} = await import('./local-engine');

/** A fake in-process model — the INLINE path's stand-in. Every score it produces
 *  is distinguishable from the worker's, so a test can tell which path ran. */
function inlineLoader(logit: number) {
  return async () => ({
    tokenizer: (texts: string[]) => ({ n: texts.length }),
    model: async (inputs: Record<string, unknown>) => ({
      logits: { dims: [inputs.n as number, 1], data: new Array(inputs.n as number).fill(logit) },
    }),
  });
}

const INLINE_SCORE = 1 / (1 + Math.exp(-7)); // sigmoid(7), from inlineLoader(7)

beforeEach(() => {
  scoreViaWorker.mockReset();
  warnRerankFallback.mockReset();
  getRerankWorkerState.mockReturnValue({ alive: true, disabled: false, pendingCount: 0 });
});

afterEach(() => _setLoaderForTest(null));

describe('scoreCrossEncoder dispatch', () => {
  it('prefers the worker when no loader override is installed', async () => {
    scoreViaWorker.mockResolvedValue([0.25, 0.75]);

    await expect(scoreCrossEncoder('q', ['a', 'b'])).resolves.toEqual([0.25, 0.75]);
    expect(scoreViaWorker).toHaveBeenCalledTimes(1);
  });

  it('passes the resolved batch/length settings and the deadline through to the worker', async () => {
    scoreViaWorker.mockResolvedValue([1]);
    const deadline = Date.now() + 5_000;

    await scoreCrossEncoder('q', ['a'], { deadline });

    expect(scoreViaWorker).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'q',
        texts: ['a'],
        deadline,
        maxLength: DEFAULT_MAX_LENGTH,
        batchSize: DEFAULT_BATCH_SIZE,
        // Resolved on the MAIN thread so GPU-demotion logic is not duplicated
        // across the thread seam.
        device: expect.any(String),
        dtype: expect.any(String),
      }),
    );
  });

  it('stays INLINE when a loader override is installed', async () => {
    // This is the guard that keeps the rest of the suite meaningful: every other
    // test in this package injects a fake loader. If the worker path ran anyway,
    // those tests would silently start exercising the REAL ONNX model while
    // believing they had injected a fake.
    _setLoaderForTest(inlineLoader(7));

    await expect(scoreCrossEncoder('q', ['a'])).resolves.toEqual([INLINE_SCORE]);
    expect(scoreViaWorker).not.toHaveBeenCalled();
  });

  it('stays INLINE when the worker is permanently unavailable', async () => {
    getRerankWorkerState.mockReturnValue({ alive: false, disabled: true, pendingCount: 0 });
    _setLoaderForTest(inlineLoader(7));

    await expect(scoreCrossEncoder('q', ['a'])).resolves.toEqual([INLINE_SCORE]);
    expect(scoreViaWorker).not.toHaveBeenCalled();
  });

  it('falls back to inline scoring when the worker fails, rather than failing the rerank', async () => {
    _setLoaderForTest(inlineLoader(7));
    _setWorkerWithLoaderForTest(true); // see its doc — production never combines these
    scoreViaWorker.mockRejectedValue(new Error('worker spawn failed'));

    await expect(scoreCrossEncoder('q', ['a'])).resolves.toEqual([INLINE_SCORE]);
    expect(scoreViaWorker).toHaveBeenCalledTimes(1);
    expect(warnRerankFallback).toHaveBeenCalledTimes(1);
  });

  it('does NOT re-run inline when the worker reports a deadline expiry', async () => {
    // A deadline expiry is a VERDICT, not a worker fault. Re-running it inline
    // would spend the budget that just expired — on the main thread — which is
    // precisely the blocking this whole change removes.
    _setLoaderForTest(inlineLoader(7));
    _setWorkerWithLoaderForTest(true);
    scoreViaWorker.mockRejectedValue(
      new Error('rerank: scoring deadline exceeded after 0 of 1 pairs — degrading to retrieval order'),
    );

    await expect(scoreCrossEncoder('q', ['a'])).rejects.toThrow(/deadline exceeded/);
    expect(warnRerankFallback).not.toHaveBeenCalled();
  });

  it('short-circuits an empty batch without touching the worker', async () => {
    await expect(scoreCrossEncoder('q', [])).resolves.toEqual([]);
    expect(scoreViaWorker).not.toHaveBeenCalled();
  });
});
