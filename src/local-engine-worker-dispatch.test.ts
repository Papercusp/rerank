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
import { withTestTokenEncoding } from './test-tokenizer';
import { _resetScorerGateForTest, _setScorerGateCalibrationForTest } from './scorer-gate';

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
    tokenizer: withTestTokenEncoding((texts: string[]) => ({ n: texts.length })),
    model: async (inputs: Record<string, unknown>) => ({
      logits: { dims: [inputs.n as number, 1], data: new Array(inputs.n as number).fill(logit) },
    }),
  });
}

const INLINE_SCORE = 1 / (1 + Math.exp(-7)); // sigmoid(7), from inlineLoader(7)

beforeEach(() => {
  _resetScorerGateForTest();
  scoreViaWorker.mockReset();
  warnRerankFallback.mockReset();
  getRerankWorkerState.mockReturnValue({ alive: true, disabled: false, pendingCount: 0 });
});

afterEach(() => {
  _setLoaderForTest(null);
  _resetScorerGateForTest();
  vi.restoreAllMocks();
});

describe('scoreCrossEncoder dispatch', () => {
  it('does not dispatch an abandoned scorer waiter or retry it inline', async () => {
    let release!: (scores: number[]) => void;
    let started!: () => void;
    const firstStarted = new Promise<void>(resolve => { started = resolve; });
    scoreViaWorker.mockImplementationOnce(() => {
      started();
      return new Promise<number[]>(resolve => { release = resolve; });
    }).mockResolvedValue([0.75]);
    const first = scoreCrossEncoder('active', ['a']);
    await firstStarted;
    const controller = new AbortController();
    const abandoned = scoreCrossEncoder('abandoned', ['b'], { signal: controller.signal }).catch(error => error);
    const reason = new Error('HTTP caller disconnected');
    controller.abort(reason);
    release([0.25]);
    await expect(first).resolves.toEqual([0.25]);
    expect(await abandoned).toBe(reason);
    expect(scoreViaWorker).toHaveBeenCalledTimes(1);
    expect(warnRerankFallback).not.toHaveBeenCalled();
  });

  it('distinguishes shared scorer waiting from scoring for each concurrent caller', async () => {
    let clock = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => clock);
    let finishFirst!: (scores: number[]) => void;
    let firstStarted!: () => void;
    const started = new Promise<void>(resolve => { firstStarted = resolve; });
    scoreViaWorker.mockImplementationOnce(() => {
      firstStarted();
      return new Promise<number[]>(resolve => { finishFirst = resolve; });
    }).mockImplementationOnce(async () => { clock = 90; return [0.75]; });
    const firstTimings: unknown[] = [], secondTimings: unknown[] = [];
    const first = scoreCrossEncoder('first', ['a'], { onTiming: event => firstTimings.push(event) });
    await started;
    clock = 10;
    const second = scoreCrossEncoder('second', ['b'], { onTiming: event => secondTimings.push(event) });
    clock = 60;
    finishFirst([0.25]);
    await expect(Promise.all([first, second])).resolves.toEqual([[0.25], [0.75]]);
    expect(firstTimings).toEqual([
      { phase: 'admission', outcome: 'returned', durationMs: 0 },
      { phase: 'scoring', outcome: 'returned', durationMs: 60 },
    ]);
    expect(secondTimings).toEqual([
      { phase: 'admission', outcome: 'returned', durationMs: 50 },
      { phase: 'scoring', outcome: 'returned', durationMs: 30 },
    ]);
  });

  it('attributes a shed to admission and a worker deadline to scoring without changing either error', async () => {
    _setScorerGateCalibrationForTest(1);
    const shed: unknown[] = [];
    await expect(scoreCrossEncoder('q', ['a'], {
      deadline: Date.now() - 1, onTiming: event => shed.push(event),
    })).rejects.toThrow(/deadline exceeded/);
    expect(shed).toEqual([{ phase: 'admission', outcome: 'threw', durationMs: expect.any(Number) }]);
    expect(scoreViaWorker).not.toHaveBeenCalled();
    const timings: unknown[] = [];
    const error = new Error('rerank: scoring deadline exceeded in worker');
    scoreViaWorker.mockRejectedValue(error);
    await expect(scoreCrossEncoder('q', ['a'], { onTiming: event => timings.push(event) })).rejects.toBe(error);
    expect(timings).toEqual([
      { phase: 'admission', outcome: 'returned', durationMs: expect.any(Number) },
      { phase: 'scoring', outcome: 'threw', durationMs: expect.any(Number) },
    ]);
    expect(warnRerankFallback).not.toHaveBeenCalled();
  });

  it('isolates timing observer failures from scores, worker errors and empty batches', async () => {
    const onTiming = vi.fn(() => { throw new Error('observer failed'); });
    scoreViaWorker.mockResolvedValueOnce([0.25]);
    await expect(scoreCrossEncoder('q', ['a'], { onTiming })).resolves.toEqual([0.25]);
    const error = new Error('rerank: scoring deadline exceeded');
    scoreViaWorker.mockRejectedValueOnce(error);
    await expect(scoreCrossEncoder('q', ['a'], { onTiming })).rejects.toBe(error);
    expect(onTiming).toHaveBeenCalledTimes(4);
    onTiming.mockClear();
    await expect(scoreCrossEncoder('q', [], { onTiming })).resolves.toEqual([]);
    expect(onTiming).not.toHaveBeenCalled();
  });

  it('prefers the worker when no loader override is installed', async () => {
    scoreViaWorker.mockResolvedValue([0.25, 0.75]);

    await expect(scoreCrossEncoder('q', ['a', 'b'])).resolves.toEqual([0.25, 0.75]);
    expect(scoreViaWorker).toHaveBeenCalledTimes(1);
  });

  it('forwards a bounded CPU thread setting and rejects invalid values before worker dispatch', async () => {
    scoreViaWorker.mockResolvedValue([0.25]);
    await expect(scoreCrossEncoder('q', ['a'], { intraOpNumThreads: 1 })).resolves.toEqual([0.25]);
    expect(scoreViaWorker).toHaveBeenCalledWith(expect.objectContaining({ intraOpNumThreads: 1 }));
    for (const intraOpNumThreads of [0, 5, 1.5, NaN, Infinity]) {
      await expect(scoreCrossEncoder('q', ['a'], { intraOpNumThreads })).rejects.toThrow('integer from 1 to 4');
    }
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
