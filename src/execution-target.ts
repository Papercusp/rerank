/**
 * Host-aware execution target for the local cross-encoder (plan P-009).
 *
 * ## The one rule this file exists to enforce
 *
 * **dtype is not independently selectable — it is a property of the DEVICE.**
 * A weight format and an execution provider are chosen TOGETHER or not at all,
 * because the wrong half of the pair is worse than either whole:
 *
 * - `fp16` is a GPU format. x86 has no native fp16 compute, so ONNX Runtime's
 *   CPU provider casts every fp16 weight up to fp32 at inference time and pays
 *   the conversion on top of full-precision math. MEASURED on this stack, not
 *   assumed — see the plan's D-006.
 * - `q8` is the CPU-correct format: int8 kernels are native, the weights are ~4x
 *   smaller, and it preserves the ranking (D-006 measured top-1 agreement 1.000
 *   against fp32).
 *
 * So the selectable unit here is a `RerankExecutionTarget` — a (device, dtype)
 * PAIR — and every transition moves both fields at once. An API that let a
 * caller set `dtype: 'fp16'` while the device stayed `cpu` would make the single
 * worst configuration the easiest one to reach by accident.
 *
 * ## Why "is there a GPU?" is the wrong question
 *
 * Detecting a GPU is not the same as being able to USE one. The box this was
 * developed on has an RTX 3090, the CUDA 12 runtime, and ships ORT's CUDA
 * provider — and still cannot execute on it, because cuDNN 9 is absent, so
 * `libonnxruntime_providers_cuda.so` fails to load. Anything that keyed off
 * `nvidia-smi` finding a card would have selected the GPU target on that host,
 * failed to construct a session, and — because the engine is fail-safe — turned
 * every search into silent retrieval order with no reranking at all.
 *
 * Hence: the GPU target is opt-IN (`PAPERCUSP_RERANK_DEVICE`), and even when it
 * is opted into it is VERIFIED — see `local-engine.ts`, which falls back to the
 * CPU pair when a GPU session does not actually construct.
 */

/** Execution providers this engine knows how to target. */
export type RerankDevice = 'cpu' | 'cuda' | 'webgpu';

export interface RerankExecutionTarget {
  /** ONNX Runtime execution provider. */
  device: RerankDevice;
  /** ONNX weight format, valid ONLY in combination with `device`. */
  dtype: string;
  /** Why this pair — carried so a log line can explain itself. */
  why: string;
}

/**
 * The safe default, and the fallback for every failed GPU attempt.
 *
 * D-006 measured on 4 pinned cores: q8 is 1.58x faster per pair than fp32 while
 * keeping top-1 agreement at 1.000 and NDCG@5 at 0.986, so it is the right CPU
 * default on quality grounds as well as cost.
 */
export const CPU_EXECUTION_TARGET: RerankExecutionTarget = Object.freeze({
  device: 'cpu',
  dtype: 'q8',
  why: 'CPU provider: int8 kernels are native on x86, weights are ~4x smaller, and q8 preserves the ranking head (D-006)',
});

/** The GPU pair. fp16 is only ever correct WITH a GPU execution provider. */
export const CUDA_EXECUTION_TARGET: RerankExecutionTarget = Object.freeze({
  device: 'cuda',
  dtype: 'fp16',
  why: 'CUDA provider: fp16 is native on the GPU, halves bandwidth, and avoids the int8 quantisation error entirely',
});

/** WebGPU (desktop shell / browser). fp16 is native there too. */
export const WEBGPU_EXECUTION_TARGET: RerankExecutionTarget = Object.freeze({
  device: 'webgpu',
  dtype: 'fp16',
  why: 'WebGPU provider: fp16 is the native shader format',
});

/**
 * Opt a GPU host in. Unset/`cpu` ⇒ the CPU pair; `cuda`/`webgpu` ⇒ the matching
 * GPU pair, still subject to the verified-load fallback in the engine.
 *
 * Opt-IN rather than auto-detect, deliberately: an auto-probe would pay a failed
 * multi-second model load on every CPU host, and (see the file header) presence
 * of a GPU does not imply a usable one.
 */
export const RERANK_DEVICE_ENV = 'PAPERCUSP_RERANK_DEVICE';

const TARGETS_BY_DEVICE: Record<RerankDevice, RerankExecutionTarget> = {
  cpu: CPU_EXECUTION_TARGET,
  cuda: CUDA_EXECUTION_TARGET,
  webgpu: WEBGPU_EXECUTION_TARGET,
};

/** The pair for a device. Total over `RerankDevice`, so a new device cannot be
 *  added without also stating its dtype. */
export function executionTargetFor(device: RerankDevice): RerankExecutionTarget {
  return TARGETS_BY_DEVICE[device];
}

/**
 * Resolve this host's execution target from the environment.
 *
 * Unknown/misspelled values resolve to the CPU pair rather than throwing: a typo
 * in a deployment env var must not be able to take search down, and the CPU pair
 * is always a correct answer.
 */
export function resolveExecutionTarget(
  env: Record<string, string | undefined> = typeof process === 'undefined' ? {} : process.env,
): RerankExecutionTarget {
  const raw = (env[RERANK_DEVICE_ENV] ?? '').trim().toLowerCase();
  if (raw === 'cuda' || raw === 'gpu') return CUDA_EXECUTION_TARGET;
  if (raw === 'webgpu') return WEBGPU_EXECUTION_TARGET;
  return CPU_EXECUTION_TARGET;
}

/** True when the pair is one this module vouches for. Guards the trap the whole
 *  file exists to prevent: a GPU weight format on the CPU provider. */
export function isCoherentTarget(target: { device: string; dtype: string }): boolean {
  const known = TARGETS_BY_DEVICE[target.device as RerankDevice];
  return known !== undefined && known.dtype === target.dtype;
}
