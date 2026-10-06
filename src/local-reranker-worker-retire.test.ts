/**
 * WI-10006567 guard for the rerank worker (the embedder's sibling seam, see
 * libs/generic/memory/src/local-embedder-worker-retire.test.ts for the real-binding
 * reproduction): a shutdown with requests owed must RETIRE the worker — terminate it
 * only after it reports no scoring in flight, because terminating mid-run aborts the
 * process — and the retired worker's late messages must never settle the
 * replacement's requests (request ids restart at 0).
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import {
  _resetBeforeExitHookForTest,
  getRerankWorkerState,
  scoreViaWorker,
  shutdownLocalReranker,
} from './local-reranker-worker';

type WorkerInstance = InstanceType<typeof import('node:worker_threads').Worker>;

const req = { query: 'q', texts: ['a'], model: 'retire-probe', dtype: 'fp32', device: 'cpu' as const, maxLength: 16, batchSize: 1 };

async function swallowScorePosts() {
  const { Worker } = await import('node:worker_threads');
  const realPost = Worker.prototype.postMessage;
  return vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(function (this: WorkerInstance, msg: unknown) {
    if ((msg as { kind?: string } | null)?.kind === 'score') return;
    return realPost.call(this, msg);
  });
}

async function captureWorkers() {
  const { Worker } = await import('node:worker_threads');
  const realRef = Worker.prototype.ref;
  const seen: WorkerInstance[] = [];
  vi.spyOn(Worker.prototype, 'ref').mockImplementation(function (this: WorkerInstance) {
    if (!seen.includes(this)) seen.push(this);
    return realRef.call(this);
  });
  return seen;
}

afterEach(async () => {
  vi.restoreAllMocks();
  await shutdownLocalReranker();
  _resetBeforeExitHookForTest();
});

describe('local rerank worker retirement (WI-10006567)', () => {
  it("a retired worker's late messages never settle the replacement's requests", async () => {
    const post = await swallowScorePosts();
    const seen = await captureWorkers();

    const first = scoreViaWorker(req).then(() => 'answered', (e: Error) => `rejected: ${e.message}`);
    await vi.waitFor(() => expect(getRerankWorkerState().pendingCount).toBe(1));
    const oldWorker = seen.at(-1)!;
    await shutdownLocalReranker();
    expect(await first).toMatch(/shut down while 1 request/);

    let settled: string | null = null;
    const second = scoreViaWorker(req).then(
      (v) => { settled = `resolved ${JSON.stringify(v)}`; },
      (e: Error) => { settled = `rejected: ${e.message}`; },
    );
    await vi.waitFor(() => expect(getRerankWorkerState().pendingCount).toBe(1));
    expect(seen.at(-1)).not.toBe(oldWorker);

    // Pre-fix the first of these resolved `second` with [7] (same id 0) and the
    // exit nulled the live handle.
    oldWorker.emit('message', { kind: 'score_ok', id: 0, scores: [7] });
    oldWorker.emit('exit', 1);
    await new Promise((r) => setTimeout(r, 20));
    expect(settled).toBeNull();
    expect(getRerankWorkerState()).toMatchObject({ alive: true, pendingCount: 1 });

    post.mockRestore();
    await shutdownLocalReranker();
    await second;
    expect(settled).toMatch(/shut down while 1 request/);
  }, 30_000);

  it('terminates a worker that owed answers only after it reports retired', async () => {
    const { Worker } = await import('node:worker_threads');
    await swallowScorePosts();
    const seen = await captureWorkers();
    const owed = scoreViaWorker(req).catch((e: Error) => e);
    await vi.waitFor(() => expect(getRerankWorkerState().pendingCount).toBe(1));
    const w = seen.at(-1)!;
    let retiredSeen = false;
    w.on('message', (m: { kind?: string }) => { if (m?.kind === 'retired') retiredSeen = true; });
    const realTerminate = Worker.prototype.terminate;
    const order: boolean[] = [];
    vi.spyOn(Worker.prototype, 'terminate').mockImplementation(function (this: WorkerInstance) {
      if (this === w) order.push(retiredSeen);
      return realTerminate.call(this);
    });

    await shutdownLocalReranker();
    expect(await owed).toBeInstanceOf(Error);
    expect(order).toEqual([true]);
    expect(getRerankWorkerState()).toMatchObject({ alive: false, pendingCount: 0 });
  }, 30_000);
});
