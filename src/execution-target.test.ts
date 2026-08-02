/**
 * Host-aware execution target (plan P-009).
 *
 * The invariant under test is narrow and load-bearing: **a (device, dtype) pair
 * is never split.** Both halves of the trap were confirmed empirically on the
 * development host (see the plan's D-006), and both are silent rather than
 * loud, because the engine is fail-safe:
 *
 *  - `fp16` on the CPU provider does not construct a session at all (ORT's
 *    precision-free-cast insertion breaks ModernBERT's layernorm fusion).
 *  - The CUDA provider does not load without cuDNN — on a box that has an
 *    RTX 3090, CUDA 12, and the provider binary all present.
 *
 * In both cases `localEngineScores` catches, returns `null`, and search quietly
 * serves retrieval order forever. So these tests care much less about "did we
 * pick fp16 on a GPU" than about "can any code path leave us holding a GPU
 * dtype on a CPU device".
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CPU_EXECUTION_TARGET,
  CUDA_EXECUTION_TARGET,
  RERANK_DEVICE_ENV,
  WEBGPU_EXECUTION_TARGET,
  executionTargetFor,
  isCoherentTarget,
  resolveExecutionTarget,
  type RerankDevice,
} from './execution-target';
import { _setLoaderForTest, activeExecutionTarget, loadCrossEncoder } from './local-engine';

describe('resolveExecutionTarget', () => {
  it('defaults to the CPU pair when nothing is set', () => {
    expect(resolveExecutionTarget({})).toEqual(CPU_EXECUTION_TARGET);
  });

  it('opts a CUDA host in, as a whole pair', () => {
    const t = resolveExecutionTarget({ [RERANK_DEVICE_ENV]: 'cuda' });
    expect(t.device).toBe('cuda');
    expect(t.dtype).toBe('fp16');
  });

  it('accepts the `gpu` alias and is case/whitespace insensitive', () => {
    for (const raw of ['gpu', 'CUDA', '  Cuda  ']) {
      expect(resolveExecutionTarget({ [RERANK_DEVICE_ENV]: raw })).toEqual(CUDA_EXECUTION_TARGET);
    }
  });

  it('resolves webgpu to its own pair', () => {
    expect(resolveExecutionTarget({ [RERANK_DEVICE_ENV]: 'webgpu' })).toEqual(WEBGPU_EXECUTION_TARGET);
  });

  it('falls back to CPU on a typo rather than throwing — a bad env var must not take search down', () => {
    for (const raw of ['cdua', 'nvidia', 'true', '']) {
      expect(resolveExecutionTarget({ [RERANK_DEVICE_ENV]: raw })).toEqual(CPU_EXECUTION_TARGET);
    }
  });
});

describe('pair coherence', () => {
  it('every exported target is self-consistent', () => {
    for (const t of [CPU_EXECUTION_TARGET, CUDA_EXECUTION_TARGET, WEBGPU_EXECUTION_TARGET]) {
      expect(isCoherentTarget(t)).toBe(true);
    }
  });

  it('every device in the type has a declared dtype — a new device cannot be added without one', () => {
    const devices: RerankDevice[] = ['cpu', 'cuda', 'webgpu'];
    for (const d of devices) {
      const t = executionTargetFor(d);
      expect(t, `no target declared for device ${d}`).toBeDefined();
      expect(t.dtype).toBeTruthy();
      expect(t.device).toBe(d);
    }
  });

  it('REJECTS the trap pair: a GPU dtype on the CPU provider', () => {
    expect(isCoherentTarget({ device: 'cpu', dtype: 'fp16' })).toBe(false);
    expect(isCoherentTarget({ device: 'cpu', dtype: 'q4f16' })).toBe(false);
  });

  it('rejects a CPU dtype on a GPU provider too — the split is wrong in both directions', () => {
    expect(isCoherentTarget({ device: 'cuda', dtype: 'q8' })).toBe(false);
  });

  it('rejects an unknown device', () => {
    expect(isCoherentTarget({ device: 'tpu', dtype: 'fp16' })).toBe(false);
  });
});

describe('verified GPU selection (engine)', () => {
  const loads: Array<{ model: string; dtype: string; device: string }> = [];

  /** Fails for any non-CPU device, exactly as a host without cuDNN does. */
  const loaderFailingOnGpu = async (model: string, dtype: string, device: RerankDevice) => {
    loads.push({ model, dtype, device });
    if (device !== 'cpu') {
      throw new Error('Failed to load library libonnxruntime_providers_cuda.so: libcudnn.so.9 missing');
    }
    return { tokenizer: (() => ({})) as never, model: (async () => ({ logits: { dims: [0, 1], data: [] } })) as never };
  };

  beforeEach(() => {
    loads.length = 0;
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    _setLoaderForTest(null);
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it('uses the CPU pair on a default host', async () => {
    _setLoaderForTest(loaderFailingOnGpu);
    await loadCrossEncoder();
    expect(loads).toEqual([{ model: expect.any(String), dtype: 'q8', device: 'cpu' }]);
  });

  it('demotes a GPU host to the WHOLE CPU pair — never fp16 left on the CPU device', async () => {
    vi.stubEnv(RERANK_DEVICE_ENV, 'cuda');
    _setLoaderForTest(loaderFailingOnGpu);

    await loadCrossEncoder();

    // It must have TRIED the GPU pair...
    expect(loads[0]).toMatchObject({ dtype: 'fp16', device: 'cuda' });
    // ...and landed on the CPU pair, with the dtype moved too. This is the
    // whole point: a fallback that only reset `device` would leave fp16 here,
    // which is the one configuration that cannot construct a session at all.
    expect(loads[1]).toMatchObject({ dtype: 'q8', device: 'cpu' });
    expect(isCoherentTarget(loads[1])).toBe(true);
  });

  it('remembers the demotion so a broken GPU host pays the failed load once, not per search', async () => {
    vi.stubEnv(RERANK_DEVICE_ENV, 'cuda');
    _setLoaderForTest(loaderFailingOnGpu);

    await loadCrossEncoder();
    const afterFirst = loads.length;
    await loadCrossEncoder({ model: 'second-model' });

    const gpuAttempts = loads.filter((l) => l.device !== 'cpu');
    expect(gpuAttempts).toHaveLength(1);
    expect(loads.length).toBeGreaterThan(afterFirst);
  });

  it('reports the ACTIVE target after demotion, not the requested one', async () => {
    vi.stubEnv(RERANK_DEVICE_ENV, 'cuda');
    _setLoaderForTest(loaderFailingOnGpu);

    await loadCrossEncoder();

    expect(activeExecutionTarget()).toEqual(CPU_EXECUTION_TARGET);
  });

  it('still surfaces a CPU load failure instead of recursing forever', async () => {
    _setLoaderForTest(async () => {
      throw new Error('model file corrupt');
    });
    await expect(loadCrossEncoder()).rejects.toThrow('model file corrupt');
  });
});
