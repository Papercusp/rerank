/**
 * A GPU demotion must be DISTINGUISHABLE from a deliberate CPU host.
 *
 * The engine is fail-safe: a CUDA session that will not construct is demoted to
 * cpu/q8 and search keeps working. That is correct, and it is also why the
 * failure is so easy to miss — the only symptom is slowness. The sidecar's
 * `/healthz` was built to reveal it, but `activeExecutionTarget()` returned the
 * frozen `CPU_EXECUTION_TARGET` on demotion, so a demoted host and a deliberate
 * CPU host rendered BYTE-IDENTICALLY. Measured on this box 2026-09-05: both
 * rerank models reported cpu/q8 with the ordinary "int8 kernels are native"
 * rationale while the requested CUDA provider was failing to load, with 21.4GB
 * of VRAM free.
 *
 * These tests pin the distinction itself. `deliberate CPU host` is the positive
 * control: without it, an assertion that a demoted host "looks demoted" would
 * also pass if EVERY host looked demoted.
 *
 * Seam: `_setLoaderForTest` swaps the loader, so the real demotion logic runs
 * with no ONNX and no model download; the fake fails exactly the GPU attempt.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { CPU_EXECUTION_TARGET, RERANK_DEVICE_ENV } from './execution-target';
import {
  _setLoaderForTest,
  activeExecutionTarget,
  loadCrossEncoder,
  rerankExecutionHealth,
} from './local-engine';

const GPU_LOAD_FAILURE = 'OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library';

/** Succeeds on the CPU provider, fails on any GPU one — the real shape of a host
 *  whose CUDA provider library is missing. */
function loaderFailingOnGpu() {
  return async (_model: string, _dtype: string, device: string) => {
    if (device !== 'cpu') throw new Error(GPU_LOAD_FAILURE);
    return {
      tokenizer: ((text: string[], o: { text_pair: string[] }) => ({ __texts: o.text_pair })) as never,
      model: (async (inputs: Record<string, unknown>) => {
        const texts = inputs.__texts as string[];
        return { logits: { dims: [texts.length, 1], data: texts.map(() => 0) } };
      }) as never,
    };
  };
}

let savedDevice: string | undefined;

beforeEach(() => {
  savedDevice = process.env[RERANK_DEVICE_ENV];
});

afterEach(() => {
  // Clears the loader override AND the remembered demotion, so a demoted verdict
  // cannot leak into the next test and make it pass for the wrong reason.
  _setLoaderForTest(null);
  if (savedDevice === undefined) delete process.env[RERANK_DEVICE_ENV];
  else process.env[RERANK_DEVICE_ENV] = savedDevice;
});

describe('GPU demotion is observable', () => {
  it('POSITIVE CONTROL: a deliberate CPU host reports the plain CPU pair and is not demoted', async () => {
    delete process.env[RERANK_DEVICE_ENV];
    _setLoaderForTest(loaderFailingOnGpu());

    await loadCrossEncoder({});

    expect(activeExecutionTarget()).toEqual(CPU_EXECUTION_TARGET);
    const health = rerankExecutionHealth();
    expect(health.demoted).toBe(false);
    expect(health.demotionCause).toBeNull();
  });

  it('a demoted host does NOT render identically to a deliberate CPU host', async () => {
    process.env[RERANK_DEVICE_ENV] = 'cuda';
    _setLoaderForTest(loaderFailingOnGpu());

    await loadCrossEncoder({});

    const active = activeExecutionTarget();
    // The regression this file exists to catch: returning the frozen CPU
    // constant here made the degraded state unreadable from the healthy one.
    expect(active).not.toEqual(CPU_EXECUTION_TARGET);
    expect(active.why).not.toBe(CPU_EXECUTION_TARGET.why);
    expect(active.why).toContain('DEMOTED');
    expect(active.why).toContain('cuda/fp16');
  });

  it('demotion changes only the RATIONALE — device and dtype still run the CPU kernels', async () => {
    process.env[RERANK_DEVICE_ENV] = 'cuda';
    _setLoaderForTest(loaderFailingOnGpu());

    await loadCrossEncoder({});

    // Every consumer branching on device/dtype must be unaffected; only a reader
    // of `why` (or of rerankExecutionHealth) sees the difference.
    const active = activeExecutionTarget();
    expect(active.device).toBe(CPU_EXECUTION_TARGET.device);
    expect(active.dtype).toBe(CPU_EXECUTION_TARGET.dtype);
  });

  it('rerankExecutionHealth reports requested vs active, the verdict, and the cause', async () => {
    process.env[RERANK_DEVICE_ENV] = 'cuda';
    _setLoaderForTest(loaderFailingOnGpu());

    await loadCrossEncoder({});

    const health = rerankExecutionHealth();
    expect(health.requested.device).toBe('cuda');
    expect(health.active.device).toBe('cpu');
    expect(health.demoted).toBe(true);
    // The cause is what makes the report actionable: "missing provider library"
    // and "out of VRAM" demand completely different responses.
    expect(health.demotionCause).toContain('Failed to load shared library');
  });

  it('the remembered demotion does not leak across a loader reset', async () => {
    process.env[RERANK_DEVICE_ENV] = 'cuda';
    _setLoaderForTest(loaderFailingOnGpu());
    await loadCrossEncoder({});
    expect(rerankExecutionHealth().demoted).toBe(true);

    _setLoaderForTest(null);

    expect(rerankExecutionHealth().demoted).toBe(false);
    expect(rerankExecutionHealth().demotionCause).toBeNull();
  });
});
