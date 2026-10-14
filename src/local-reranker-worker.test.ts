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
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { afterEach, describe, expect, it, vi } from 'vitest';
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

function workerMessages(worker: Worker) {
  const queued: Record<string, unknown>[] = [];
  const waiters: Array<{
    kind: string;
    resolve: (message: Record<string, unknown>) => void;
    reject: (error: Error) => void;
  }> = [];
  let workerFailure: Error | null = null;
  worker.on('error', (error) => {
    workerFailure = error;
    for (const waiter of waiters.splice(0)) waiter.reject(error);
  });
  worker.on('exit', (code) => {
    if (waiters.length > 0) {
      const error = new Error(`rerank worker exited with code ${code} before the expected message`);
      for (const waiter of waiters.splice(0)) waiter.reject(error);
    }
  });
  worker.on('message', (value: unknown) => {
    if (!value || typeof value !== 'object') return;
    const message = value as Record<string, unknown>;
    if (message.kind === 'score_err') {
      const waiter = waiters.shift();
      if (waiter) waiter.reject(new Error(String(message.error)));
      else queued.push(message);
      return;
    }
    const index = waiters.findIndex((waiter) => waiter.kind === message.kind);
    if (index >= 0) waiters.splice(index, 1)[0]!.resolve(message);
    else queued.push(message);
  });
  return (kind: string) => {
    if (workerFailure) return Promise.reject(workerFailure);
    const failureIndex = queued.findIndex((message) => message.kind === 'score_err');
    if (failureIndex >= 0) return Promise.reject(new Error(String(queued.splice(failureIndex, 1)[0]!.error)));
    const index = queued.findIndex((message) => message.kind === kind);
    if (index >= 0) return Promise.resolve(queued.splice(index, 1)[0]!);
    return new Promise<Record<string, unknown>>((resolve, reject) => waiters.push({ kind, resolve, reject }));
  };
}

describe('local rerank worker (real thread)', () => {
  it('shares worker ownership and shutdown across distinct module records', async () => {
    const { Worker } = await import('node:worker_threads');
    const first = await import('./local-reranker-worker');
    let second: typeof first | undefined;
    const post = vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(() => {});
    const pending = first.scoreViaWorker({ ...baseReq }).catch((error: unknown) => error);
    try {
      await vi.waitFor(() => expect(first.getRerankWorkerState().pendingCount).toBe(1));
      vi.resetModules();
      second = await import('./local-reranker-worker');
      expect(second.scoreViaWorker).not.toBe(first.scoreViaWorker);
      expect(second.getRerankWorkerState()).toEqual(first.getRerankWorkerState());
      // Restore delivery first: a worker that owes an answer is retired, not
      // killed (WI-10006567), and the retire request must reach it.
      post.mockRestore();
      await second.shutdownLocalReranker();
      expect(await pending).toBeInstanceOf(Error);
      expect(first.getRerankWorkerState()).toMatchObject({ alive: false, pendingCount: 0 });
    } finally {
      post.mockRestore();
      await first.shutdownLocalReranker();
      await second?.shutdownLocalReranker();
      await pending;
      first._resetBeforeExitHookForTest();
      second?._resetBeforeExitHookForTest();
    }
  });

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

    // IN FLIGHT ⇒ ref'd. Without this the loop looks idle while the caller
    // awaits, `beforeExit` fires, and the hook terminates this very request
    // (WI-37680) — which reads to the caller as "no worker available".
    //
    // ⚠ ONE atomic snapshot, asserted as an invariant — deliberately NOT two
    // separate `expect(getRerankWorkerState()...)` reads. Reading `pendingCount`
    // and `keepAlive` in separate assertions let the reply land BETWEEN them, so
    // this red under fleet load while the property itself was fine
    // (EI-20053788422725852). The property is real; the measurement was fragile.
    const snap = getRerankWorkerState();
    expect({ pending: snap.pendingCount > 0, keepAlive: snap.keepAlive })
      .toEqual({ pending: true, keepAlive: true });

    await inflight;
    const after = getRerankWorkerState();
    expect({ pending: after.pendingCount > 0, keepAlive: after.keepAlive })
      .toEqual({ pending: false, keepAlive: false });
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

describe('local rerank worker Transformers.js cache', () => {
  it('writes to PAPERCUSP_HOME and keeps installed package cache assets readable', async () => {
    const root = mkdtempSync(join(tmpdir(), 'rerank-transformers-cache-'));
    const home = join(root, 'tenant-home');
    const packageRoot = join(root, 'transformers-package');
    const packageCache = join(packageRoot, '.cache');
    const assetPath = join(packageCache, 'fixture', 'config.json');
    mkdirSync(join(packageCache, 'fixture'), { recursive: true });
    writeFileSync(assetPath, 'installed-model-asset');

    const fakeModulePath = join(root, 'fake-transformers.mjs');
    writeFileSync(fakeModulePath, `
      import { parentPort } from 'node:worker_threads';
      import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
      import { join } from 'node:path';
      export const env = {};
      function record(stage) {
        const preservedAsset = readFileSync(join(env.localModelPath, 'fixture', 'config.json'), 'utf8');
        mkdirSync(env.cacheDir, { recursive: true });
        writeFileSync(join(env.cacheDir, 'worker-write-probe'), 'rerank-cache-write');
        parentPort.postMessage({ kind: 'cache_config', stage, localModelPath: env.localModelPath, cacheDir: env.cacheDir, preservedAsset });
      }
      export class AutoTokenizer {
        static async from_pretrained() {
          record('tokenizer');
          return (_queries, options) => ({ batchSize: options.text_pair.length });
        }
      }
      export class AutoModelForSequenceClassification {
        static async from_pretrained() {
          record('model');
          return async (inputs) => ({ logits: { dims: [inputs.batchSize, 1], data: new Float32Array(inputs.batchSize).fill(0.25) } });
        }
      }
    `);

    const scriptPath = resolve(dirname(fileURLToPath(import.meta.url)), 'local-reranker-worker.script.mjs');
    const worker = new Worker(scriptPath, {
      // `test:file` may place committed-source-loader in process.execArgv; this test must run the worker under test.
      execArgv: [],
      env: { ...process.env, PAPERCUSP_HOME: home },
      workerData: {
        transformersSpecifier: pathToFileURL(fakeModulePath).href,
        transformersPackageRoot: packageRoot,
      },
    });
    const next = workerMessages(worker);
    try {
      expect(await next('ready')).toMatchObject({ kind: 'ready' });
      worker.postMessage({
        kind: 'score', id: 1, query: 'q', texts: ['candidate'], model: 'fixture/model',
        dtype: 'q8', device: 'cpu', maxLength: 16, batchSize: 1,
      });
      const configA = await next('cache_config');
      const configB = await next('cache_config');
      for (const config of [configA, configB]) {
        expect(config).toMatchObject({
          localModelPath: packageCache,
          cacheDir: join(home, 'cache', 'transformers'),
          preservedAsset: 'installed-model-asset',
        });
      }
      expect(new Set([configA.stage, configB.stage])).toEqual(new Set(['tokenizer', 'model']));
      expect(await next('score_ok')).toMatchObject({ kind: 'score_ok', id: 1, scores: [expect.any(Number)] });
      expect(readFileSync(join(home, 'cache', 'transformers', 'worker-write-probe'), 'utf8'))
        .toBe('rerank-cache-write');
      expect(readFileSync(assetPath, 'utf8')).toBe('installed-model-asset');
    } finally {
      await worker.terminate();
      rmSync(root, { recursive: true, force: true });
    }
  }, 30_000);
});
