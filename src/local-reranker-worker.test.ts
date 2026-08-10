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

import { execFile } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

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

  it('holds the event loop open for exactly as long as a request is in flight', async () => {
    await scoreViaWorker({ ...baseReq }); // warm: the worker is ready and idle

    // IDLE ⇒ unref'd. This half is a real control, not decoration: "always ref"
    // would pass the in-flight assertion below while re-breaking the property
    // the unref exists for (a one-off script that can never exit).
    expect(getRerankWorkerState().keepAlive).toBe(false);

    const inflight = scoreViaWorker({ ...baseReq });
    // The worker answers on a MACROTASK, so draining microtasks cannot let the
    // reply land — this observes the in-flight state deterministically rather
    // than guessing at a number of ticks.
    for (let i = 0; i < 50 && getRerankWorkerState().pendingCount === 0; i++) await Promise.resolve();
    expect(getRerankWorkerState().pendingCount).toBe(1);

    // IN FLIGHT ⇒ ref'd. Without this the loop looks idle while the caller
    // awaits, `beforeExit` fires, and the hook terminates this very request
    // (WI-37680) — which reads to the caller as "no worker available".
    expect(getRerankWorkerState().keepAlive).toBe(true);

    await inflight;
    expect(getRerankWorkerState().pendingCount).toBe(0);
    expect(getRerankWorkerState().keepAlive).toBe(false);
  }, 30_000);

  it('does not tear the worker down from beforeExit while a request is pending', async () => {
    await scoreViaWorker({ ...baseReq });

    const inflight = scoreViaWorker({ ...baseReq });
    for (let i = 0; i < 50 && getRerankWorkerState().pendingCount === 0; i++) await Promise.resolve();
    expect(getRerankWorkerState().pendingCount).toBe(1);

    // Fire the hook by hand. `syncWorkerRef` should make this unreachable in a
    // real process; the guard behind it is what keeps a future ref regression
    // from silently re-opening the same hole.
    await Promise.all(process.listeners('beforeExit').map((fn) => (fn as () => unknown)()));

    await expect(inflight).resolves.toEqual([]);
    expect(getRerankWorkerState().alive).toBe(true);
  }, 30_000);

  it('answers a host whose event loop is otherwise idle (child process)', async () => {
    // THE reproduction. The defect only exists when nothing else holds the loop
    // open, which is unreproducible inside vitest — its own runner always has
    // work pending. So this spawns a bare host that awaits one rerank and does
    // nothing else, which is every CLI, bench and one-off script.
    //
    // Zero pairs on purpose: the worker replies without loading a model, so this
    // needs no download and still crosses the full thread boundary. Pre-fix it
    // failed 100% of the time with `rerank worker exited with code 1`.
    const dir = mkdtempSync(join(tmpdir(), 'rerank-idle-loop-'));
    try {
      const modulePath = resolve(__dirname, 'local-reranker-worker.ts');
      const child = join(dir, 'idle-host.mts');
      writeFileSync(
        child,
        `const { scoreViaWorker } = await import(${JSON.stringify(modulePath)});\n` +
          `try {\n` +
          `  const scores = await scoreViaWorker({ query: 'q', texts: [], model: 'm',\n` +
          `    dtype: 'q8', device: 'cpu', maxLength: 512, batchSize: 16 });\n` +
          `  console.log('CHILD_OK ' + JSON.stringify(scores));\n` +
          `} catch (err) {\n` +
          `  console.log('CHILD_FAIL ' + (err instanceof Error ? err.message : String(err)));\n` +
          `}\n`,
      );

      const { stdout } = await promisify(execFile)('npx', ['tsx', child], {
        cwd: resolve(__dirname, '..'),
        timeout: 90_000,
      });

      expect(stdout).toContain('CHILD_OK');
      expect(stdout).not.toContain('CHILD_FAIL');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);

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
