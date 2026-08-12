/**
 * The sidecar-first reranker client.
 *
 * The load-bearing property is the AVAILABILITY CONTRACT: with a sidecar
 * configured, the sidecar is REQUIRED — retries happen inside ONE total budget
 * and then it THROWS, and the in-process engine is never silently substituted.
 * Silent failover is what previously let a stalling sidecar drag every host
 * into duplicate model loads while hiding that the sidecar was sick.
 *
 * Seams: `fetchFn`, `now`, `sleepFn`, `localScorer` — all injected, so nothing
 * here opens a socket, sleeps in real time, or loads ONNX.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  RERANK_SIDECAR_URL_ENV,
  SIDECAR_MAX_TEXT_CHARS,
  SidecarRerankHttpError,
  buildSidecarFirstReranker,
  isNonRetryableSidecarError,
  resolveRerankSidecarUrl,
  sidecarRerankBatch,
} from './sidecar-reranker';

const okResponse = (scores: number[]) =>
  new Response(JSON.stringify({ scores, runtime: 'node-onnx-worker', modelRev: 'm' }), { status: 200 });

/** Deterministic clock + no real sleeping. */
const fakeTiming = () => {
  let t = 0;
  return { now: () => (t += 10), sleepFn: async () => {}, advance: (ms: number) => (t += ms) };
};

describe('resolveRerankSidecarUrl', () => {
  it('reads the SAME env var as the embedder — one sidecar serves both wires', () => {
    expect(RERANK_SIDECAR_URL_ENV).toBe('PAPERCUSP_EMBED_SIDECAR_URL');
    expect(resolveRerankSidecarUrl({})).toBeNull();
    expect(resolveRerankSidecarUrl({ [RERANK_SIDECAR_URL_ENV]: '  ' })).toBeNull();
    expect(resolveRerankSidecarUrl({ [RERANK_SIDECAR_URL_ENV]: 'http://127.0.0.1:3384/' })).toBe(
      'http://127.0.0.1:3384',
    );
  });
});

describe('sidecarRerankBatch', () => {
  it('posts the wire shape to /rerank and returns the scores', async () => {
    const fetchFn = vi.fn(async () => okResponse([0.9, 0.1]));
    const res = await sidecarRerankBatch('http://x/', {
      model: 'rerank',
      query: 'q',
      texts: ['a', 'b'],
      fetchFn: fetchFn as unknown as typeof fetch,
    });
    expect(res.scores).toEqual([0.9, 0.1]);
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('http://x/rerank');
    expect(JSON.parse(init.body as string)).toEqual({ model: 'rerank', query: 'q', texts: ['a', 'b'] });
  });

  it('truncates oversized query and texts client-side rather than earning a deterministic 400', async () => {
    const fetchFn = vi.fn(async () => okResponse([1]));
    await sidecarRerankBatch('http://x', {
      model: 'rerank',
      query: 'q'.repeat(SIDECAR_MAX_TEXT_CHARS + 50),
      texts: ['t'.repeat(SIDECAR_MAX_TEXT_CHARS + 50)],
      fetchFn: fetchFn as unknown as typeof fetch,
    });
    const body = JSON.parse((fetchFn.mock.calls[0] as unknown as [string, RequestInit])[1].body as string);
    expect(body.query).toHaveLength(SIDECAR_MAX_TEXT_CHARS);
    expect(body.texts[0]).toHaveLength(SIDECAR_MAX_TEXT_CHARS);
  });

  it('throws a status-carrying error on non-2xx', async () => {
    const fetchFn = vi.fn(async () => new Response('nope', { status: 400 }));
    await expect(
      sidecarRerankBatch('http://x', {
        model: 'rerank',
        query: 'q',
        texts: ['a'],
        fetchFn: fetchFn as unknown as typeof fetch,
      }),
    ).rejects.toBeInstanceOf(SidecarRerankHttpError);
  });

  it('rejects a score/text length mismatch — misaligned scores rank on garbage', async () => {
    const fetchFn = vi.fn(async () => okResponse([0.5]));
    await expect(
      sidecarRerankBatch('http://x', {
        model: 'rerank',
        query: 'q',
        texts: ['a', 'b'],
        fetchFn: fetchFn as unknown as typeof fetch,
      }),
    ).rejects.toThrow(/bad_shape/);
  });
});

describe('isNonRetryableSidecarError', () => {
  it('separates a deterministic 4xx rejection from a transient failure', () => {
    expect(isNonRetryableSidecarError(new SidecarRerankHttpError(400, ''))).toBe(true);
    expect(isNonRetryableSidecarError(new SidecarRerankHttpError(404, ''))).toBe(true);
    expect(isNonRetryableSidecarError(new SidecarRerankHttpError(500, ''))).toBe(false);
    expect(isNonRetryableSidecarError(new Error('ECONNREFUSED'))).toBe(false);
  });
});

describe('buildSidecarFirstReranker — no sidecar configured', () => {
  it('uses the in-process engine as the SOLE engine', async () => {
    const localScorer = vi.fn(async (_q: string, texts: string[]) => texts.map((t) => t.length));
    const score = buildSidecarFirstReranker({ url: null, localScorer });
    expect(await score('q', ['aa', 'b'])).toEqual([2, 1]);
    expect(localScorer).toHaveBeenCalledTimes(1);
  });
});

describe('buildSidecarFirstReranker — sidecar REQUIRED', () => {
  it('serves from the sidecar and never touches the in-process engine', async () => {
    const localScorer = vi.fn(async () => [999]);
    const fetchFn = vi.fn(async () => okResponse([0.7]));
    const score = buildSidecarFirstReranker({
      url: 'http://x',
      localScorer,
      fetchFn: fetchFn as unknown as typeof fetch,
    });
    expect(await score('q', ['a'])).toEqual([0.7]);
    expect(localScorer).not.toHaveBeenCalled();
  });

  it('retries a TRANSIENT failure within the budget, then succeeds', async () => {
    const t = fakeTiming();
    let calls = 0;
    const fetchFn = vi.fn(async () => {
      calls++;
      if (calls < 3) throw new Error('ECONNREFUSED');
      return okResponse([0.5]);
    });
    const transitions: string[] = [];
    const score = buildSidecarFirstReranker({
      url: 'http://x',
      fetchFn: fetchFn as unknown as typeof fetch,
      now: t.now,
      sleepFn: t.sleepFn,
      onTransition: (state) => transitions.push(state),
    });
    expect(await score('q', ['a'])).toEqual([0.5]);
    expect(calls).toBe(3);
    // Transition-only logging: one 'down', one 'up' — not one line per attempt.
    expect(transitions).toEqual(['down', 'up']);
  });

  it('THROWS sidecar_required_unavailable rather than failing over in-process', async () => {
    const localScorer = vi.fn(async () => [999]);
    const t = fakeTiming();
    const fetchFn = vi.fn(async () => {
      throw new Error('ECONNREFUSED');
    });
    const score = buildSidecarFirstReranker({
      url: 'http://x',
      localScorer,
      fetchFn: fetchFn as unknown as typeof fetch,
      now: t.now,
      sleepFn: t.sleepFn,
      onTransition: () => {},
    });
    await expect(score('q', ['a'])).rejects.toThrow(/sidecar_required_unavailable/);
    // THE contract: no silent in-process failover.
    expect(localScorer).not.toHaveBeenCalled();
  });

  it('does NOT retry a deterministic 4xx — the same payload can never succeed', async () => {
    const t = fakeTiming();
    const fetchFn = vi.fn(async () => new Response('bad shape', { status: 400 }));
    const transitions: string[] = [];
    const score = buildSidecarFirstReranker({
      url: 'http://x',
      fetchFn: fetchFn as unknown as typeof fetch,
      now: t.now,
      sleepFn: t.sleepFn,
      onTransition: (state) => transitions.push(state),
    });
    await expect(score('q', ['a'])).rejects.toThrow(/sidecar_rejected_request/);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    // 'rejected', never 'down' — the sidecar is up and correctly refusing.
    expect(transitions).toEqual(['rejected']);
  });

  it('stops attempting once the TOTAL budget is spent, regardless of maxAttempts', async () => {
    const t = fakeTiming();
    const fetchFn = vi.fn(async () => {
      t.advance(9_000); // each attempt eats most of the 15s budget
      throw new Error('timeout');
    });
    const score = buildSidecarFirstReranker({
      url: 'http://x',
      fetchFn: fetchFn as unknown as typeof fetch,
      now: t.now,
      sleepFn: t.sleepFn,
      maxAttempts: 10,
      onTransition: () => {},
    });
    await expect(score('q', ['a'])).rejects.toThrow(/sidecar_required_unavailable/);
    expect(fetchFn.mock.calls.length).toBeLessThan(10);
  });

  it('serializes concurrent batches and sheds an impossible queued call before HTTP dispatch', async () => {
    let clock = 0;
    const fetchFn = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(init?.body as string) as { texts: string[] };
      clock += 400;
      return okResponse(body.texts.map((text) => text.length));
    });
    const transitions: string[] = [];
    const score = buildSidecarFirstReranker({
      url: 'http://x',
      timeoutMs: 5_000,
      maxAttempts: 1,
      now: () => clock,
      fetchFn: fetchFn as unknown as typeof fetch,
      onTransition: (state) => transitions.push(state),
    });

    const first = score('q1', ['a'], { deadline: 5_000 });
    const queued = score('q2', ['b'], { deadline: 450 });

    await expect(first).resolves.toEqual([1]);
    await expect(queued).rejects.toThrow(/deadline exceeded/);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    // Queue admission, not sidecar health, caused the shed: do not flap the
    // transition state or make the benchmark think the sidecar went down.
    expect(transitions).toEqual([]);
  });

  it('short-circuits an empty candidate list without a wire call', async () => {
    const fetchFn = vi.fn(async () => okResponse([]));
    const score = buildSidecarFirstReranker({ url: 'http://x', fetchFn: fetchFn as unknown as typeof fetch });
    expect(await score('q', [])).toEqual([]);
    expect(fetchFn).not.toHaveBeenCalled();
  });
});

describe('the fail-safe seam converts a sidecar throw into retrieval order', () => {
  it('rerank() degrades instead of propagating sidecar_required_unavailable', async () => {
    const { rerank } = await import('./index');
    const scorer = buildSidecarFirstReranker({
      url: 'http://x',
      fetchFn: (async () => {
        throw new Error('ECONNREFUSED');
      }) as unknown as typeof fetch,
      now: fakeTiming().now,
      sleepFn: async () => {},
      onTransition: () => {},
    });
    const docs = [
      { id: 'a', text: 'alpha', row: 'A' },
      { id: 'b', text: 'beta', row: 'B' },
    ];
    const out = await rerank('q', docs, { engine: 'local', scorer });
    expect(out.map((r) => r.row)).toEqual(['A', 'B']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
  });

  it('an injected scorer with the wrong number of scores fails safe rather than misaligning', async () => {
    const { rerank } = await import('./index');
    const docs = [
      { id: 'a', text: 'alpha', row: 'A' },
      { id: 'b', text: 'beta', row: 'B' },
    ];
    const out = await rerank('q', docs, { engine: 'local', scorer: async () => [0.9] });
    expect(out.map((r) => r.row)).toEqual(['A', 'B']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
  });

  it('a healthy sidecar scorer reorders through the normal shaping path', async () => {
    const { rerank } = await import('./index');
    const docs = [
      { id: 'a', text: 'alpha', row: 'A' },
      { id: 'b', text: 'beta', row: 'B' },
    ];
    const out = await rerank('q', docs, { engine: 'local', scorer: async () => [0.1, 0.9] });
    expect(out.map((r) => r.row)).toEqual(['B', 'A']);
    expect(out.every((r) => r.reranked === true)).toBe(true);
  });
});
