/**
 * P-531 (plan agent-capacity-and-cost-gcp-2026-09-30, WI-10005523): the
 * sidecar-first reranker's per-attempt `ensure` hook — the rerank analog of
 * the embedder's (libs/generic/memory/src/sidecar-embedder-ensure.test.ts).
 *
 * A locally spawned sidecar can exit after an idle period and is NOT
 * crash-respawned, so the client must re-establish it before an attempt. A
 * hanging ensure must stay inside the caller's deadline: search degrades to
 * retrieval order on a deadline failure rather than waiting on a cold start.
 *
 * Each test uses its own URL because the admission gate is cached per
 * (url, model).
 */
import { describe, expect, it } from 'vitest';

import { buildSidecarFirstReranker } from './sidecar-reranker';
import { RerankDeadlineError } from './scorer-gate';

const okResponse = (scores: number[]) =>
  new Response(JSON.stringify({ scores, runtime: 'node-onnx-worker', modelRev: 'm' }), { status: 200 });

const neverLocal = async (): Promise<number[]> => {
  throw new Error('in-process scorer must never run when a url is set');
};

describe('buildSidecarFirstReranker ensure hook (P-531)', () => {
  it('re-ensures before the retry, so a sidecar that exited is re-launched and scoring succeeds', async () => {
    const events: string[] = [];
    let sidecarUp = false;
    const score = buildSidecarFirstReranker({
      url: 'http://p531-reensure',
      localScorer: neverLocal,
      sleepFn: async () => {},
      onTransition: (state) => events.push(`transition:${state}`),
      ensure: async () => {
        events.push('ensure');
        if (events.filter((e) => e === 'ensure').length >= 2) sidecarUp = true;
      },
      fetchFn: (async () => {
        events.push('fetch');
        if (!sidecarUp) throw new Error('connect ECONNREFUSED');
        return okResponse([0.9]);
      }) as typeof fetch,
    });

    expect(await score('q', ['a'])).toEqual([0.9]);
    expect(events).toEqual(['ensure', 'fetch', 'transition:down', 'ensure', 'fetch', 'transition:up']);
  });

  it('a hanging ensure is bounded by the caller deadline and fails as a deadline, never sending work', async () => {
    let fetches = 0;
    const score = buildSidecarFirstReranker({
      url: 'http://p531-hang',
      localScorer: neverLocal,
      sleepFn: async () => {},
      onTransition: () => {},
      ensure: () => new Promise<void>(() => {}), // never settles
      fetchFn: (async () => {
        fetches++;
        return okResponse([0.1]);
      }) as typeof fetch,
    });
    const started = Date.now();
    await expect(score('q', ['a'], { deadline: Date.now() + 60 })).rejects.toBeInstanceOf(RerankDeadlineError);
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(fetches).toBe(0);
  });

  it('without an ensure hook the client behaves exactly as before', async () => {
    let fetches = 0;
    const score = buildSidecarFirstReranker({
      url: 'http://p531-nohook',
      localScorer: neverLocal,
      fetchFn: (async () => {
        fetches++;
        return okResponse([0.4, 0.6]);
      }) as typeof fetch,
    });
    expect(await score('q', ['a', 'b'])).toEqual([0.4, 0.6]);
    expect(fetches).toBe(1);
  });
});

/**
 * WI-10005932: a null url WITH an ensure hook is a spawned sidecar that is not
 * up yet, not "no sidecar". The reranker must stay sidecar-only (never run the
 * in-process cross-encoder in the main Server) and use the address a later
 * ensure reports.
 */
describe('buildSidecarFirstReranker with a null url and an ensure hook (WI-10005932)', () => {
  it('never runs the in-process scorer and scores via the URL a later ensure reports', async () => {
    const fetched: string[] = [];
    let ensures = 0;
    const score = buildSidecarFirstReranker({
      model: 'wi10005932-late',
      url: null,
      localScorer: neverLocal,
      sleepFn: async () => {},
      onTransition: () => {},
      ensure: async () => {
        ensures++;
        return ensures >= 2 ? 'http://127.0.0.1:41225/' : null;
      },
      fetchFn: (async (input: string | URL | Request) => {
        fetched.push(String(input));
        return okResponse([0.7]);
      }) as typeof fetch,
    });

    expect(await score('q', ['a'])).toEqual([0.7]);
    expect(ensures).toBe(2);
    expect(fetched).toEqual(['http://127.0.0.1:41225/rerank']);
  });

  it('fails without fetching or running the in-process scorer when no address ever arrives', async () => {
    let fetches = 0;
    const score = buildSidecarFirstReranker({
      model: 'wi10005932-never',
      url: null,
      localScorer: neverLocal,
      sleepFn: async () => {},
      onTransition: () => {},
      maxAttempts: 3,
      ensure: async () => null,
      fetchFn: (async () => {
        fetches++;
        return okResponse([0.1]);
      }) as typeof fetch,
    });

    await expect(score('q', ['a'])).rejects.toThrow(/sidecar_not_ready/);
    expect(fetches).toBe(0);
  });
});
