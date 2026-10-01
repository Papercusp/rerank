/**
 * WI-10005090 guard for the reranker worker: spawning it must first pin the
 * onnxruntime-node binding in the spawning thread, so a terminated reranker
 * worker cannot leave the process unable to load the binding again ("Module
 * did not self-register"). The hazard itself, and the pinned/unpinned reload
 * measurement, is proven in
 * libs/generic/memory/src/local-embedder-worker-onnx-pin.test.ts; this file
 * asserts the reranker's spawn path reaches the same pin.
 *
 * No model is loaded: the score request is kept from reaching the worker and
 * is rejected by shutdown.
 */
import { describe, it, expect, vi } from 'vitest';
import { Worker } from 'node:worker_threads';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import {
  _resetBeforeExitHookForTest,
  _resetOnnxBindingPinForTest,
  getOnnxBindingPin,
  getRerankWorkerState,
  pinOnnxRuntimeBinding,
  scoreViaWorker,
  shutdownLocalReranker,
} from './local-reranker-worker';

const scriptPath = resolve(dirname(fileURLToPath(import.meta.url)), 'local-reranker-worker.script.mjs');

function resolveOnnxFromTransformers(): string | null {
  try {
    const transformersEntry = createRequire(scriptPath).resolve('@huggingface/transformers');
    return createRequire(transformersEntry).resolve('onnxruntime-node');
  } catch {
    return null;
  }
}

const onnxPath = resolveOnnxFromTransformers();

describe.skipIf(!onnxPath)('reranker ONNX binding pin (WI-10005090)', () => {
  it('spawning the reranker worker pins the binding before the worker starts', async () => {
    _resetOnnxBindingPinForTest();
    expect(getOnnxBindingPin()).toBeNull();
    const post = vi.spyOn(Worker.prototype, 'postMessage').mockImplementation(() => {});
    const pending = scoreViaWorker({
      query: 'q',
      texts: ['a'],
      model: 'pin-order-probe',
      dtype: 'fp32',
      device: 'cpu',
      maxLength: 16,
      batchSize: 1,
    }).catch((error: unknown) => error);
    try {
      await vi.waitFor(() => expect(getRerankWorkerState().pendingCount).toBe(1));
      expect(getOnnxBindingPin()).toEqual({ status: 'pinned', path: onnxPath });
    } finally {
      post.mockRestore();
      await shutdownLocalReranker();
      await pending;
      _resetBeforeExitHookForTest();
    }
  }, 60_000);
});

describe('reranker ONNX binding pin when transformers is not installed', () => {
  it('reports unavailable without throwing', () => {
    _resetOnnxBindingPinForTest();
    try {
      expect(pinOnnxRuntimeBinding('/nonexistent/dir/local-reranker-worker.script.mjs')).toMatchObject({
        status: 'unavailable',
      });
    } finally {
      _resetOnnxBindingPinForTest();
    }
  });
});
