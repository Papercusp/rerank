/**
 * Worker-thread isolated local cross-encoder.
 *
 * WHY (EI-20005411672741677, finished by WI-37555). A cross-encoder forward pass
 * is CPU-bound and holds its thread. On the MAIN thread that means nothing else
 * in the process progresses while it runs — including the `setTimeout` behind
 * every wall-clock bound in this lib, so no timer can preempt it. `local-engine`
 * yields between batches to make those bounds REAL, but that only bounds the
 * overrun to one batch (~720ms at the default 16 pairs). With the model here,
 * the caller's `Promise.race` bound becomes EXACT — the main thread is free to
 * service its own timers — and head-of-line blocking of unrelated work goes away.
 *
 * Architecture mirrors `@papercusp/memory`'s `local-embedder-worker.ts`: one
 * persistent worker per process, lazily spawned, holding the warm model(s);
 * main-thread requests marshal a batch over `postMessage` and await a pending
 * Promise keyed by id. Falls back to inline (main-thread) scoring whenever the
 * worker is unavailable — degraded, never broken.
 */

import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import type { RerankDevice } from './execution-target';

interface PendingRequest {
  resolve: (v: number[]) => void;
  reject: (err: Error) => void;
}

let _worker: Worker | null = null;
let _workerReady: Promise<void> | null = null;
let _nextId = 0;
const _pending = new Map<number, PendingRequest>();
/**
 * Set ONLY for genuine, permanent unavailability — the worker could not be
 * CONSTRUCTED at all (no `worker_threads`, missing script, spawn threw). A
 * runtime crash deliberately does NOT set this: it clears the worker handle so
 * the next call respawns one.
 *
 * That asymmetry is the point (the lesson EI-16184 taught the embedder path):
 * treating a transient crash as permanent condemns every later call in the
 * process to the inline, main-thread-blocking path — i.e. one hiccup silently
 * undoes this whole module for the rest of the process's life.
 */
let _workerDisabled = false;
/** Guards the process-level `beforeExit` hook so it is installed at most once,
 *  however many times a worker (re)spawns. */
let _beforeExitHookInstalled = false;
let _beforeExitListener: (() => Promise<void>) | null = null;

/**
 * Whether the worker is currently holding the event loop open. Mirrors the last
 * `ref()`/`unref()` we issued, because `Worker` exposes no way to read it back.
 */
let _refd = false;

/**
 * Hold the loop open for EXACTLY as long as a request is in flight, and not one
 * moment longer.
 *
 * WHY BOTH HALVES ARE LOAD-BEARING (WI-37680). An always-ref'd worker keeps a
 * one-shot script alive forever — the bug the original `unref()` fixed. But an
 * always-UNREF'd worker is worse in a way that is silent: while the caller
 * awaits a reply, the unref'd worker (and its port) no longer count as loop
 * work, so a host with nothing else ref'd is considered IDLE **mid-request**.
 * `beforeExit` then fires, the hook below terminates the worker, and the exit
 * handler rejects the in-flight request with `rerank worker exited with code 1`
 * — which `scoreCrossEncoder` reads as "the worker is unavailable" and answers
 * by falling back to inline main-thread scoring.
 *
 * So the whole worker-thread mechanism silently did not apply on any host whose
 * loop is otherwise empty: every CLI, bench and one-off script. Those are
 * exactly the hosts where it was measured, which is how the inline path's
 * serialization got recorded as a property of the worker path (WI-37676).
 * Measured 2026-08-10: a `scoreViaWorker` with ZERO pairs — no model load at
 * all — failed this way 100% of the time from a standalone script.
 *
 * Ref'ing only while `_pending` is non-empty satisfies both: a script that
 * awaits a rerank stays alive until its answer arrives, then exits naturally.
 */
function syncWorkerRef(): void {
  const want = _pending.size > 0;
  if (!_worker || want === _refd) return;
  if (want) _worker.ref();
  else _worker.unref();
  _refd = want;
}

function workerPath(): string {
  // Co-located with this module, resolved at RUNTIME so the path survives
  // build-time bundling — provided the build copies the script beside the
  // bundle. It must: without that copy every rerank silently falls back to
  // inline main-thread ONNX (the WI-4196 trap, which has already recurred once
  // for a sibling worker). `bundle-host.sh` does the copy;
  // `worker-scripts-bundled.test.ts` fails if a new worker script is missed.
  const here = typeof __filename !== 'undefined' ? __filename : fileURLToPath(import.meta.url);
  return resolve(dirname(here), 'local-reranker-worker.script.mjs');
}

function ensureWorker(): Promise<void> {
  if (_workerDisabled) return Promise.reject(new Error('rerank worker disabled'));
  if (_workerReady) return _workerReady;

  _workerReady = new Promise<void>((resolveReady, rejectReady) => {
    try {
      _worker = new Worker(workerPath());
    } catch (err) {
      // Construction failed — genuinely unavailable, not a transient fault.
      _workerDisabled = true;
      _workerReady = null;
      rejectReady(err as Error);
      return;
    }

    let initialized = false;
    _worker.on(
      'message',
      (msg: { kind: string; id?: number; scores?: number[]; error?: string }) => {
        if (msg.kind === 'ready') {
          initialized = true;
          // A REF'd worker keeps the event loop alive forever, so a one-off
          // script that reranks once and ends could never exit naturally — and
          // reaching for `process.exit()` tears the ONNX native addon down
          // mid-flight (an unlabelled `Napi::Error` at teardown). Unref'ing lets
          // such a script finish on its own; the beforeExit hook below then
          // terminates the worker cleanly. Same reasoning as the embedder's
          // EI-19464316359123796.
          //
          // A freshly constructed Worker is ref'd, so record that and let
          // `syncWorkerRef` decide: idle ⇒ unref (the behaviour above), a
          // request already in flight ⇒ stay ref'd until it answers (WI-37680).
          _refd = true;
          syncWorkerRef();
          installBeforeExitHook();
          resolveReady();
          return;
        }
        if (typeof msg.id !== 'number') return;
        const p = _pending.get(msg.id);
        if (!p) return;
        _pending.delete(msg.id);
        // Release the loop as soon as the LAST request settles, so a one-off
        // script exits on its own.
        syncWorkerRef();
        if (msg.kind === 'score_ok' && Array.isArray(msg.scores)) p.resolve(msg.scores);
        else p.reject(new Error(msg.error ?? 'rerank worker error'));
      },
    );

    _worker.on('error', (err) => {
      // The worker crashed: fail every in-flight request, then clear the handle
      // so the NEXT call respawns. Deliberately not `_workerDisabled` — see its
      // declaration.
      for (const [, p] of _pending) p.reject(err);
      _pending.clear();
      _worker = null;
      _workerReady = null;
      _refd = false;
      if (!initialized) rejectReady(err);
    });

    _worker.on('exit', (code) => {
      if (code !== 0 && !initialized) {
        rejectReady(new Error(`rerank worker exited with code ${code} before ready`));
      }
      // Reject anything still waiting — an exited worker will never answer it,
      // and a request left pending forever would hang the caller past its bound.
      for (const [, p] of _pending) p.reject(new Error(`rerank worker exited with code ${code}`));
      _pending.clear();
      _worker = null;
      _workerReady = null;
      _refd = false;
    });
  }).catch((err) => {
    _workerReady = null;
    throw err;
  });

  return _workerReady;
}

/** The resolved, worker-side scoring request. Device/dtype are resolved on the
 *  MAIN thread and passed explicitly, so the engine's GPU-demotion logic stays
 *  in exactly one place rather than being duplicated across the thread seam. */
export interface ScoreViaWorkerRequest {
  query: string;
  texts: string[];
  model: string;
  dtype: string;
  device: RerankDevice;
  maxLength: number;
  batchSize: number;
  deadline?: number;
}

/**
 * Score a batch on the worker thread. Rejects when the worker is unavailable or
 * the scoring itself failed — callers fall back to inline scoring.
 */
export async function scoreViaWorker(req: ScoreViaWorkerRequest): Promise<number[]> {
  await ensureWorker();
  if (!_worker) throw new Error('rerank worker not initialized');

  const id = _nextId++;
  return new Promise<number[]>((resolveScore, rejectScore) => {
    _pending.set(id, { resolve: resolveScore, reject: rejectScore });
    // Ref BEFORE posting: between the post and the reply the caller is awaiting
    // a Promise, which is not loop work — an unref'd worker would leave the loop
    // looking idle and let `beforeExit` terminate this very request.
    syncWorkerRef();
    _worker!.postMessage({ kind: 'score', id, ...req });
  });
}

/** Terminate the persistent rerank worker, if any. Await this before an
 *  explicit `process.exit()` in a standalone script — `terminate()` runs the
 *  worker isolate's normal cleanup (including the ONNX addon's finalizers),
 *  which `process.exit()` skips. */
export async function shutdownLocalReranker(): Promise<void> {
  if (_worker) {
    try {
      await _worker.terminate();
    } catch {
      /* noop */
    }
  }
  _worker = null;
  _workerReady = null;
  _workerDisabled = false;
  _refd = false;
  _pending.clear();
  _nextId = 0;
}

/** Test seam — same function under the codebase's `_reset*` convention. */
export const _resetRerankWorkerForTest = shutdownLocalReranker;

function installBeforeExitHook(): void {
  if (_beforeExitHookInstalled) return;
  _beforeExitHookInstalled = true;
  // `beforeExit` (unlike `exit`) permits async work, which is required because
  // `terminate()` returns a Promise. It fires only once the loop would go idle,
  // which is exactly why the worker unrefs itself above.
  //
  // The pending guard is belt-and-braces: `syncWorkerRef` should make an idle
  // loop impossible while a request is in flight, so this branch is unreachable
  // by design. It stays because the failure it prevents (terminating a worker
  // mid-request, which the caller then reads as "no worker available") is
  // silent, and because a future ref bug would otherwise re-open it.
  _beforeExitListener = async () => {
    if (_pending.size > 0) return;
    await shutdownLocalReranker();
  };
  process.on('beforeExit', _beforeExitListener);
}

/** Test-only: remove the installed hook and clear the guard. */
export function _resetBeforeExitHookForTest(): void {
  if (_beforeExitListener) process.off('beforeExit', _beforeExitListener);
  _beforeExitListener = null;
  _beforeExitHookInstalled = false;
}

/** Telemetry for health checks and diagnostics. */
export function getRerankWorkerState(): {
  alive: boolean;
  disabled: boolean;
  pendingCount: number;
  /**
   * True while the worker is holding the event loop open — i.e. a request is in
   * flight. Must track `pendingCount > 0` exactly: a `false` here with pending
   * work is the WI-37680 defect (`beforeExit` fires mid-request and terminates
   * the worker), and a `true` with none keeps a one-off script alive forever.
   */
  keepAlive: boolean;
} {
  return {
    alive: _worker !== null,
    disabled: _workerDisabled,
    pendingCount: _pending.size,
    keepAlive: _refd,
  };
}

/**
 * Rate-limited warning for a fall back to inline (main-thread-blocking)
 * scoring. Every call retries the worker (a crashed one respawns), so a
 * SUSTAINED failure would otherwise warn on every single rerank.
 */
let _lastFallbackWarnAt = 0;
const FALLBACK_WARN_COOLDOWN_MS = 30_000;
export function warnRerankFallback(err: unknown): void {
  const now = Date.now();
  if (now - _lastFallbackWarnAt < FALLBACK_WARN_COOLDOWN_MS) return;
  _lastFallbackWarnAt = now;
  if (process.env.NODE_ENV !== 'test') {
    console.warn(
      '[rerank] worker path failed — falling back to inline (main-thread, blocks the event loop) ' +
        `for this call, so the timeout bound is only accurate to one batch: ${
          err instanceof Error ? err.message : String(err)
        }`,
    );
  }
}

/** Test-only: reset the fallback-warn cooldown. */
export function _resetFallbackWarnForTest(): void {
  _lastFallbackWarnAt = 0;
}
