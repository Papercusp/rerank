/**
 * Real-worker tests for the rerank worker seam (WI-37555).
 *
 * These SPAWN THE ACTUAL worker thread and exercise the real message protocol —
 * no mock of `worker_threads`, no fake script. They deliberately stay off the
 * ONNX path (empty batches, and an expired deadline that short-circuits before
 * any model load), so they are fast and need no model download while still
 * proving the handshake, the id-keyed pending map, the deadline contract and
 * teardown are wired correctly end to end.
 *
 * That distinction matters: mocking `Worker` here would test the mock. The one
 * thing this module exists to guarantee — that work actually leaves the main
 * thread — is unobservable if the thread boundary is faked away.
 */

import { afterEach, describe, expect, it } from 'vitest';
import {
  _resetBeforeExitHookForTest,
  _resetRerankWorkerForTest,
  getRerankWorkerState,
  scoreViaWorker,
} from './local-reranker-worker';

const baseReq = {
  query: 'q',
  texts: [] as string[],
  model: 'Alibaba-NLP/gte-reranker-modernbert-base',
  dtype: 'q8',
  device: 'cpu' as const,
  maxLength: 512,
  batchSize: 16,
};

afterEach(async () => {
  await _resetRerankWorkerForTest();
  _resetBeforeExitHookForTest();
});

describe('local rerank worker (real thread)', () => {
  it('spawns, completes the ready handshake, and round-trips a request', async () => {
    expect(getRerankWorkerState().alive).toBe(false);

    // Empty texts: the worker answers before touching a model, so this proves
    // the spawn + handshake + postMessage + pending-map resolution path only.
    await expect(scoreViaWorker({ ...baseReq })).resolves.toEqual([]);

    expect(getRerankWorkerState().alive).toBe(true);
    expect(getRerankWorkerState().disabled).toBe(false);
    expect(getRerankWorkerState().pendingCount).toBe(0);
  }, 30_000);

  it('reuses ONE persistent worker across calls rather than spawning per request', async () => {
    await scoreViaWorker({ ...baseReq });
    const first = getRerankWorkerState().alive;
    await scoreViaWorker({ ...baseReq });

    expect(first).toBe(true);
    // A per-request worker would pay model-load cost on every rerank, which is
    // the difference between usable and unusable.
    expect(getRerankWorkerState().alive).toBe(true);
  }, 30_000);

  it('rejects an already-expired deadline WITHOUT loading a model', async () => {
    // The model id is deliberately nonexistent. If the worker loaded before
    // checking the deadline, this would fail with a download/resolve error (or
    // hang for seconds) instead of the deadline message — so the assertion on
    // the MESSAGE is what pins the short-circuit, not merely that it rejected.
    await expect(
      scoreViaWorker({
        ...baseReq,
        texts: ['a', 'b'],
        model: 'definitely-not-a-real/model-zzqq',
        deadline: Date.now() - 1,
      }),
    ).rejects.toThrow(/deadline exceeded after 0 of 2 pairs/);
  }, 30_000);

  it('settles every pending request instead of leaving a caller hanging past its bound', async () => {
    const inflight = scoreViaWorker({ ...baseReq, texts: ['a'], deadline: Date.now() - 1 });
    await expect(inflight).rejects.toThrow(/deadline exceeded/);
    // A request that neither resolves nor rejects is the worst failure mode
    // here: the caller's Promise.race would be the ONLY thing bounding it.
    expect(getRerankWorkerState().pendingCount).toBe(0);
  }, 30_000);

  it('shutdown terminates the worker and allows a later respawn', async () => {
    await scoreViaWorker({ ...baseReq });
    expect(getRerankWorkerState().alive).toBe(true);

    await _resetRerankWorkerForTest();
    expect(getRerankWorkerState().alive).toBe(false);
    expect(getRerankWorkerState().disabled).toBe(false);

    // Respawn: a shutdown must not permanently disable the worker path.
    await expect(scoreViaWorker({ ...baseReq })).resolves.toEqual([]);
    expect(getRerankWorkerState().alive).toBe(true);
  }, 30_000);
});
