/**
 * Sidecar-first reranker client — the consumer side of the shared sidecar's
 * `POST {url}/rerank` wire.
 *
 * AVAILABILITY CONTRACT — deliberately identical to the embedder's
 * (`@papercusp/memory` sidecar-embedder.ts, sidecar-REQUIRED since WI-4021):
 * when a sidecar URL is configured the sidecar is REQUIRED. A failure is
 * retried briefly inside ONE total budget and then THROWN — never silently
 * absorbed by loading an in-process model. Silent failover is what let a
 * stalling sidecar drag every host into duplicate model loads and hid sidecar
 * sickness instead of surfacing it. The in-process engine is used ONLY when NO
 * sidecar is configured (desktop installs, tests, bench rigs), where it is the
 * sole engine rather than a fallback.
 *
 * ⚠ Throwing here does NOT make search brittle, and the two behaviours are not
 * in tension: this client throws so sidecar sickness is VISIBLE, and the lib's
 * single fail-safe seam one layer up (`localEngineScores` → `rerank`'s
 * passthrough) converts that throw into "keep retrieval order". So a rerank
 * outage still degrades rather than breaking search — it just stops being
 * invisible on the way.
 *
 * Same sidecar PROCESS as the embedder, so it is the same env var and the same
 * base URL — a host runs one sidecar serving both /embed and /rerank.
 *
 * Plain `fetch`, no new dependencies.
 *
 * ⚠ The retry/budget/transition POLICY below intentionally mirrors
 * sidecar-embedder.ts line for line. It is duplicated rather than shared
 * because these are two independently-borrowable libs and neither may depend
 * on the other. If you change the policy here, change it there too — and see
 * EI-19310953148940030, which proposes extracting it once a third client
 * appears (each bullet of that policy is a bug someone already paid for, so a
 * one-sided fix is the real risk).
 */

import { LOCAL_RERANKER_MODEL, scoreCrossEncoder } from './local-engine';
import { RerankDeadlineError, isDeadlineFailure } from './scorer-gate';

/**
 * Per-call scoring controls. Optional so that every existing
 * `(query, texts) => …` scorer still satisfies {@link RerankScoreFn} unchanged.
 */
export interface RerankScoreCallOpts {
  /**
   * Absolute epoch-ms instant after which scoring should STOP.
   *
   * Honoured by both engines. The in-process engine uses it to stop scoring
   * after the caller has degraded; the sidecar path uses it for admission and
   * for the HTTP attempt budget. Without this, concurrent callers can queue
   * behind the sidecar's serialized scorer and then all abort together, while
   * the client incorrectly reports a healthy-but-busy sidecar as down.
   */
  deadline?: number;
}

/** Scores `texts` against `query`, index-aligned. Throws on failure. */
export type RerankScoreFn = (
  query: string,
  texts: string[],
  opts?: RerankScoreCallOpts,
) => Promise<number[]>;

/** The SAME env var the embedder uses — one sidecar process serves both. */
export const RERANK_SIDECAR_URL_ENV = 'PAPERCUSP_EMBED_SIDECAR_URL';

/** The sidecar base URL this process should use, or null when none is
 *  configured (→ pure in-process reranking). */
export function resolveRerankSidecarUrl(env: NodeJS.ProcessEnv = process.env): string | null {
  const url = env[RERANK_SIDECAR_URL_ENV]?.trim();
  return url ? url.replace(/\/$/, '') : null;
}

export const DEFAULT_SIDECAR_TIMEOUT_MS = 15_000;
/** Attempts per call when the sidecar is required (fast failures only — the
 *  shared `timeoutMs` budget caps total wall time regardless). */
export const DEFAULT_SIDECAR_MAX_ATTEMPTS = 3;

/** Kept in sync with the sidecar server's own per-text cap. Truncating at this
 *  ONE client seam means no individual caller has to reimplement it, and the
 *  server's 400 for an oversized text becomes unreachable in practice. */
export const SIDECAR_MAX_TEXT_CHARS = 32_000;

export interface SidecarRerankResponse {
  scores: number[];
  runtime: string;
  modelRev: string;
}

/** Thrown on a non-2xx sidecar response. Carries the HTTP status so a retry
 *  policy can tell a DETERMINISTIC client-side rejection (4xx — this exact
 *  request can never succeed) apart from a transient failure (5xx, connect
 *  refused, timeout). */
export class SidecarRerankHttpError extends Error {
  readonly status: number;
  constructor(status: number, detail: string) {
    super(`sidecar_rerank_${status}: ${detail}`);
    this.name = 'SidecarRerankHttpError';
    this.status = status;
  }
}

/** True for a failure retrying can never fix (a 4xx). False for anything else
 *  (network/connect/timeout/5xx), which IS worth a bounded retry. */
export function isNonRetryableSidecarError(e: unknown): boolean {
  return e instanceof SidecarRerankHttpError && e.status >= 400 && e.status < 500;
}

export interface SidecarRerankBatchOpts {
  model: string;
  query: string;
  texts: string[];
  timeoutMs?: number;
  fetchFn?: typeof fetch;
}

/**
 * One wire call. Throws on ANY failure (network, timeout, non-200, malformed
 * or mis-sized response) — callers own the retry/fallback decision.
 */
export async function sidecarRerankBatch(url: string, opts: SidecarRerankBatchOpts): Promise<SidecarRerankResponse> {
  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_SIDECAR_TIMEOUT_MS;
  const texts = opts.texts.map((t) => (t.length > SIDECAR_MAX_TEXT_CHARS ? t.slice(0, SIDECAR_MAX_TEXT_CHARS) : t));
  const query =
    opts.query.length > SIDECAR_MAX_TEXT_CHARS ? opts.query.slice(0, SIDECAR_MAX_TEXT_CHARS) : opts.query;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchFn(`${url.replace(/\/$/, '')}/rerank`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: opts.model, query, texts }),
      signal: ctl.signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new SidecarRerankHttpError(res.status, detail.slice(0, 200));
    }
    const body = (await res.json()) as SidecarRerankResponse;
    // A length mismatch would silently misalign every score with its document,
    // producing a plausible-looking but wrong ordering — reject it loudly.
    if (
      !Array.isArray(body.scores) ||
      body.scores.length !== texts.length ||
      body.scores.some((s) => typeof s !== 'number' || !Number.isFinite(s))
    ) {
      throw new Error('sidecar_rerank_bad_shape: scores missing/mismatched');
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

interface SidecarAdmissionEntry {
  pairs: number;
  deadline: number;
  run: () => Promise<unknown>;
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
}

/**
 * The sidecar's default scorer is one serialized inference resource. A
 * per-scorer HTTP queue keeps concurrent callers from piling whole batches
 * into the server FIFO, and the measured cost of the last successful batch
 * lets the queue shed a call that cannot fit its remaining deadline before it
 * opens a socket. This is deliberately a separate gate from scorer-gate.ts:
 * the latter protects local CPU, while this one protects the remote sidecar.
 */
function createSidecarAdmissionGate(now: () => number): {
  enqueue<T>(pairs: number, deadline: number, run: () => Promise<T>): Promise<T>;
} {
  let active = false;
  const waiting: SidecarAdmissionEntry[] = [];
  let msPerPair: number | null = null;

  const deadlineError = (entry: SidecarAdmissionEntry, remaining: number, predictedMs: number | null): Error =>
    new RerankDeadlineError(
      `rerank: scoring deadline exceeded before sidecar dispatch for ${entry.pairs} pairs — ` +
        `the sidecar queue needs ~${predictedMs ?? 'unknown'}ms but only ${Math.max(0, Math.round(remaining))}ms remain; ` +
        'shedding now rather than labeling a healthy sidecar as down',
    );

  const pump = (): void => {
    if (active) return;
    const entry = waiting.shift();
    if (!entry) return;

    const remaining = entry.deadline - now();
    const predictedMs = msPerPair === null ? null : Math.ceil(msPerPair * entry.pairs);
    if (remaining <= 0 || (predictedMs !== null && predictedMs > remaining)) {
      entry.reject(deadlineError(entry, remaining, predictedMs));
      // A queue can contain several calls whose deadlines have already
      // expired. Drain those synchronously instead of leaving them to be
      // mistaken for sidecar failures on a later wake.
      pump();
      return;
    }

    active = true;
    const startedAt = now();
    let completed = false;
    void Promise.resolve()
      .then(() => entry.run())
      .then(
        (value) => {
          completed = true;
          entry.resolve(value);
        },
        (err) => entry.reject(err),
      )
      .finally(() => {
        if (completed) {
          const elapsed = Math.max(0, now() - startedAt);
          if (elapsed > 0 && entry.pairs > 0) {
            const sample = elapsed / entry.pairs;
            msPerPair = msPerPair === null ? sample : msPerPair * 0.3 + sample * 0.7;
          }
        }
        active = false;
        pump();
      });
  };

  return {
    enqueue<T>(pairs: number, deadline: number, run: () => Promise<T>): Promise<T> {
      return new Promise<T>((resolve, reject) => {
        waiting.push({
          pairs,
          deadline,
          run,
          resolve: (value) => resolve(value as T),
          reject,
        });
        pump();
      });
    },
  };
}

export interface SidecarFirstRerankerOpts {
  /** Sidecar-side model name ('rerank' | 'rerank-fast'). */
  model?: string;
  /** In-process model id, used only when no sidecar is configured. */
  localModel?: string;
  /** In-process dtype, used only when no sidecar is configured. */
  dtype?: string;
  /** Sidecar base URL; defaults to resolveRerankSidecarUrl(). null/absent ⇒
   *  pure in-process. */
  url?: string | null;
  /** TOTAL sidecar budget per call across every attempt (default 15s), unless
   * the caller supplies a tighter absolute deadline. */
  timeoutMs?: number;
  /** Attempts within the budget on sidecar failure (default 3). */
  maxAttempts?: number;
  fetchFn?: typeof fetch;
  /** Clock seam for tests. */
  now?: () => number;
  /** Backoff-sleep seam for tests. */
  sleepFn?: (ms: number) => Promise<void>;
  /** In-process scorer seam (tests inject; default is the local ONNX engine). */
  localScorer?: RerankScoreFn;
  /** Down/up/rejected transition logging seam (transition-only, so a sustained
   *  outage logs once rather than per call). 'rejected' = the sidecar is UP and
   *  correctly refusing a bad request — distinct from 'down' so the log never
   *  claims unavailability when the sidecar is fine. */
  onTransition?: (state: 'down' | 'up' | 'rejected', detail: string) => void;
}

/**
 * Build a sidecar-first RerankScoreFn. With a url: sidecar-REQUIRED — brief
 * retries inside one total budget, then throw (`sidecar_required_unavailable`);
 * the in-process engine is never touched. Without a url: the plain in-process
 * cross-encoder.
 */
export function buildSidecarFirstReranker(opts: SidecarFirstRerankerOpts = {}): RerankScoreFn {
  const url = opts.url === undefined ? resolveRerankSidecarUrl() : opts.url;
  const model = opts.model ?? 'rerank';
  const now = opts.now ?? Date.now;
  const sleep = opts.sleepFn ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const timeoutMs = opts.timeoutMs ?? DEFAULT_SIDECAR_TIMEOUT_MS;
  const maxAttempts = Math.max(1, opts.maxAttempts ?? DEFAULT_SIDECAR_MAX_ATTEMPTS);
  const admission = createSidecarAdmissionGate(now);
  const onTransition =
    opts.onTransition ??
    ((state: 'down' | 'up' | 'rejected', detail: string) =>
      console.warn(`[sidecar-reranker] ${model} sidecar ${state}: ${detail}`));

  if (!url) {
    // No sidecar configured: the in-process engine is the sole engine.
    //
    // `dtype` is forwarded ONLY when the caller actually set one. Defaulting it
    // here to DEFAULT_RERANK_DTYPE would look harmless but would break the
    // (device, dtype) pair: an explicitly-set dtype makes the engine treat the
    // call as a caller override and take `device` from the host, so on a GPU
    // host this line would silently request q8 weights on the CUDA provider.
    // Passing nothing lets the engine resolve the whole coherent pair.
    const local =
      opts.localScorer ??
      ((query: string, texts: string[], callOpts?: RerankScoreCallOpts) =>
        scoreCrossEncoder(query, texts, {
          model: opts.localModel ?? LOCAL_RERANKER_MODEL,
          ...(opts.dtype === undefined ? {} : { dtype: opts.dtype }),
          // THE path EI-20005411672741677 is about: no sidecar ⇒ this in-process
          // engine holds the main thread. Forwarding the caller's deadline is
          // what lets it stop scoring once the caller has already degraded,
          // instead of running on unwatched. Without this line the deadline
          // reaches every direct `localEngineScores` caller but NOT the shipped
          // desktop path, which is the only one that actually has the problem.
          ...(callOpts?.deadline === undefined ? {} : { deadline: callOpts.deadline }),
        }));
    return local;
  }

  let wasDown = false;

  return async (query: string, texts: string[], callOpts?: RerankScoreCallOpts): Promise<number[]> => {
    if (texts.length === 0) return [];
    const ownDeadline = now() + timeoutMs;
    const deadline = callOpts?.deadline === undefined ? ownDeadline : Math.min(callOpts.deadline, ownDeadline);

    return await admission.enqueue(texts.length, deadline, async () => {
      let lastErr: unknown;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const remaining = deadline - now();
        if (remaining <= 0) break;
        try {
          const res = await sidecarRerankBatch(url, {
            model,
            query,
            texts,
            timeoutMs: remaining,
            fetchFn: opts.fetchFn,
          });
          if (wasDown) {
            wasDown = false;
            onTransition('up', `sidecar answering again (attempt ${attempt})`);
          }
          return res.scores;
        } catch (e) {
          lastErr = e;
          if (isDeadlineFailure(e)) break;
          if (isNonRetryableSidecarError(e)) {
            // Deterministic 4xx: the sidecar is UP and correctly rejecting THIS
            // request. Retrying the same payload can never succeed, so stop
            // instead of burning the budget. Logged 'rejected', never 'down'.
            onTransition(
              'rejected',
              `${e instanceof Error ? e.message : String(e)} — sidecar correctly rejected the request (non-retryable, not a downtime issue)`,
            );
            break;
          }
          if (!wasDown) {
            wasDown = true;
            onTransition(
              'down',
              `${e instanceof Error ? e.message : String(e)} — sidecar is REQUIRED (no in-process fallback); retrying within budget`,
            );
          }
          const backoffMs = 250 * attempt;
          if (attempt < maxAttempts && deadline - now() > backoffMs + 250) await sleep(backoffMs);
          else break;
        }
      }
      if (isDeadlineFailure(lastErr)) throw lastErr;
      throw isNonRetryableSidecarError(lastErr)
        ? new Error(
            `sidecar_rejected_request: ${lastErr instanceof Error ? lastErr.message : String(lastErr)} ` +
              `(${url}, ${model}) — the sidecar rejected this request (non-retryable); ` +
              'check payload shape/size — this is not a downtime issue',
          )
        : new Error(
            `sidecar_required_unavailable: ${lastErr instanceof Error ? lastErr.message : String(lastErr)} ` +
              `(${url}, ${model}, budget ${timeoutMs}ms) — reranking requires the sidecar; ` +
              'search degrades to retrieval order while it is down',
          );
    });
  };
}
