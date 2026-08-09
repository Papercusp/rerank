/**
 * @papercusp/rerank — the fail-safe contract is the whole point of this lib:
 * a rerank outage must degrade to retrieval order, never break search. These
 * tests pin every passthrough trigger (missing key / blank query / empty
 * docs / API error / empty API result), the happy-path reordering, the
 * topN/minScore shaping on BOTH paths, the zerank-2 instruction XML
 * embedding, and the out-of-range-index filter.
 *
 * Seam: the `zeroentropy` SDK constructor (module-mocked). Everything else
 * runs real. Note the module caches its client PER API KEY, so tests that
 * need a fresh constructor call use distinct keys.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const h = vi.hoisted(() => ({
  rerankCalls: [] as Array<Record<string, unknown>>,
  // Each test overrides; default echoes input order with descending scores.
  impl: async (_req: Record<string, unknown>): Promise<{ results: Array<{ index: number; relevance_score: number }> }> => ({
    results: [],
  }),
  constructed: [] as Array<{ apiKey: string }>,
}));

vi.mock('zeroentropy', () => ({
  default: class FakeZeroEntropy {
    models: { rerank: (req: Record<string, unknown>) => Promise<unknown> };
    constructor(opts: { apiKey: string }) {
      h.constructed.push(opts);
      this.models = {
        rerank: async (req: Record<string, unknown>) => {
          h.rerankCalls.push(req);
          return h.impl(req);
        },
      };
    }
  },
}));

import { buildSidecarFirstReranker, rerank, rerankAvailable, type RerankDoc } from './index';
import { _setLoaderForTest } from './local-engine';

const docs: Array<RerankDoc<string>> = [
  { id: 'a', text: 'alpha doc', row: 'A' },
  { id: 'b', text: 'beta doc', row: 'B' },
  { id: 'c', text: 'gamma doc', row: 'C' },
];

let keySeq = 0;
/** Distinct key per use so the module-level client cache never aliases tests. */
const freshKey = () => `test-key-${++keySeq}`;

beforeEach(() => {
  h.rerankCalls.length = 0;
  h.constructed.length = 0;
  h.impl = async () => ({ results: [] });
  delete process.env.ZEROENTROPY_API_KEY;
});

afterEach(() => {
  delete process.env.ZEROENTROPY_API_KEY;
});

describe('rerankAvailable', () => {
  it('false with no key anywhere; true via arg or env', () => {
    expect(rerankAvailable()).toBe(false);
    expect(rerankAvailable('k')).toBe(true);
    process.env.ZEROENTROPY_API_KEY = 'env-k';
    expect(rerankAvailable()).toBe(true);
  });

  it('the local engine needs no credential, so it is always available', () => {
    expect(rerankAvailable({ engine: 'local' })).toBe(true);
    expect(rerankAvailable({ engine: 'zeroentropy' })).toBe(false);
  });
});

describe('engine dispatch', () => {
  /** Fake cross-encoder: scores by the doc text, so ordering is assertable. */
  const loaderScoring = (byText: Record<string, number>) => async () => ({
    tokenizer: ((_text: string[], o: { text_pair: string[] }) => ({ __texts: o.text_pair })) as never,
    model: (async (inputs: Record<string, unknown>) => {
      const texts = inputs.__texts as string[];
      return { logits: { dims: [texts.length, 1], data: texts.map((t) => byText[t] ?? 0) } };
    }) as never,
  });

  afterEach(() => _setLoaderForTest(null));

  it('engine:"local" reorders without any API key or API call', async () => {
    _setLoaderForTest(loaderScoring({ 'alpha doc': -3, 'beta doc': 5, 'gamma doc': 1 }));

    const out = await rerank('query', docs, { engine: 'local' });

    expect(out.map((r) => r.row)).toEqual(['B', 'C', 'A']);
    expect(out.every((r) => r.reranked === true)).toBe(true);
    expect(out.every((r) => r.score > 0 && r.score < 1)).toBe(true);
    expect(h.rerankCalls).toHaveLength(0);
    expect(h.constructed).toHaveLength(0);
  });

  it('engine:"local" degrades to retrieval order when the model cannot load', async () => {
    _setLoaderForTest(async () => {
      throw new Error('no onnx runtime');
    });

    const out = await rerank('query', docs, { engine: 'local' });

    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.reranked === false && r.score === 0)).toBe(true);
  });

  /**
   * RECURRENCE GUARD for EI-20005411672741677 — the `timeoutMs` bound must be
   * ENFORCEABLE against an in-process engine, not merely declared.
   *
   * Every bound in this lib is `Promise.race` + `setTimeout`, and a timer cannot
   * preempt a stage that blocks the event loop. The local cross-encoder runs
   * in-process on the main thread wherever no embed sidecar is configured (the
   * desktop install — the shipping target), so before the fix NONE of these
   * bounds could fire there: measured on the sibling corpus path, a 2,000ms
   * bound was overrun to 2,473ms while still reporting success.
   *
   * The fake model below blocks the loop exactly the way real ONNX inference
   * does, so the assertion is on the OBSERVABLE contract rather than on
   * wall-clock: if the timer cannot fire, scoring runs to completion and the
   * call returns a REORDERED result (reranked: true). Enforcement is therefore
   * visible as the degrade — a behavioural difference, not a timing threshold,
   * so it cannot go flaky under fleet load.
   */
  describe('the timeout bound survives a loop-blocking in-process engine', () => {
    const BLOCK_MS = 40;

    /** A cross-encoder whose forward pass BLOCKS the event loop, as ONNX does. */
    const blockingLoader = (counter: { batches: number }) => async () => ({
      tokenizer: ((_t: string[], o: { text_pair: string[] }) => ({ __texts: o.text_pair })) as never,
      model: (async (inputs: Record<string, unknown>) => {
        counter.batches += 1;
        const until = Date.now() + BLOCK_MS;
        while (Date.now() < until) {
          /* busy-wait: hold the thread, exactly like a synchronous forward pass */
        }
        const texts = inputs.__texts as string[];
        return { logits: { dims: [texts.length, 1], data: texts.map(() => 1) } };
      }) as never,
    });

    it('degrades to retrieval order instead of running every batch to completion', async () => {
      const counter = { batches: 0 };
      _setLoaderForTest(blockingLoader(counter));

      // 4 docs at one pair per pass = 4 blocking batches ≈ 160ms of held thread,
      // against a 50ms bound. A bound that cannot fire scores all four and wins
      // the race; an enforceable one degrades partway through.
      const many: Array<RerankDoc<string>> = ['A', 'B', 'C', 'D'].map((row) => ({
        id: row,
        text: `${row} doc`,
        row,
      }));

      const startedAt = Date.now();
      const out = await rerank('query', many, { engine: 'local', batchSize: 1, timeoutMs: 50 });
      const elapsed = Date.now() - startedAt;
      const batchesAtReturn = counter.batches;

      // THE GUARD: the bound won, so the caller got the fail-safe passthrough.
      expect(out.every((r) => r.reranked === false && r.score === 0)).toBe(true);
      expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C', 'D']);

      // ...and it won BEFORE the engine had chewed through every batch, which is
      // what distinguishes a real bound from one that merely reports lateness.
      expect(batchesAtReturn).toBeLessThan(4);

      // Wall-clock is the weakest of the three (a loaded box can stretch it), so
      // it is asserted only against the full 4-batch cost it must beat.
      expect(elapsed).toBeLessThan(BLOCK_MS * 4);
    });

    /**
     * The branch above is NOT the one production takes. The operator always
     * injects a scorer (`buildSidecarAwareReranker`), so `rerank()` goes through
     * `opts.scorer` — and with no sidecar configured THAT scorer is the
     * in-process cross-encoder. This pins the deadline actually reaching it,
     * because a fix that stopped at the un-injected branch would pass every
     * other test here while leaving the shipped desktop path unbounded.
     */
    it('carries the deadline into the in-process fallback behind a sidecar-first scorer', async () => {
      const counter = { batches: 0 };
      _setLoaderForTest(blockingLoader(counter));

      // url: null ⇒ "no sidecar configured" ⇒ the in-process engine is the sole
      // engine. This is the desktop install's exact wiring.
      const scorer = buildSidecarFirstReranker({ url: null });

      await expect(
        scorer('query', ['a doc', 'b doc'], { deadline: Date.now() - 1 }),
      ).rejects.toThrow(/deadline exceeded/);
      expect(counter.batches).toBe(0); // refused before holding the thread at all

      // ...and with headroom it scores normally through the same seam.
      const scores = await scorer('query', ['a doc', 'b doc'], { deadline: Date.now() + 10_000 });
      expect(scores).toHaveLength(2);
    });

    it('still scores normally when the work fits inside the bound', async () => {
      const counter = { batches: 0 };
      _setLoaderForTest(blockingLoader(counter));

      // One batch of 40ms against a 4s bound: the yield must not cost correctness.
      const out = await rerank('query', docs, { engine: 'local', timeoutMs: 4_000 });

      expect(out.every((r) => r.reranked === true)).toBe(true);
      expect(out).toHaveLength(3);
      expect(counter.batches).toBe(1);
    });
  });

  it('topN and minScore shape the local result the same way as the hosted one', async () => {
    _setLoaderForTest(loaderScoring({ 'alpha doc': 5, 'beta doc': 4, 'gamma doc': -5 }));

    const out = await rerank('query', docs, { engine: 'local', minScore: 0.5, topN: 1 });

    expect(out.map((r) => r.row)).toEqual(['A']);
  });

  it('defaults to the hosted engine, so an existing caller is unchanged', async () => {
    h.impl = async () => ({ results: [{ index: 1, relevance_score: 0.8 }] });
    const out = await rerank('query', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['B']);
    expect(h.rerankCalls).toHaveLength(1);
  });
});

describe('tie-breaking', () => {
  it('equal scores keep retrieval order (the first-stage ranking is the tiebreak)', async () => {
    h.impl = async () => ({
      results: [
        { index: 2, relevance_score: 0.5 },
        { index: 0, relevance_score: 0.5 },
        { index: 1, relevance_score: 0.5 },
      ],
    });
    const out = await rerank('query', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
  });
});

describe('passthrough triggers (fail-safe contract)', () => {
  it('missing key → original order, score 0, reranked=false, no API call', async () => {
    const out = await rerank('query', docs);
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.score === 0 && r.reranked === false)).toBe(true);
    expect(h.rerankCalls).toHaveLength(0);
  });

  it('blank/whitespace query → passthrough without an API call', async () => {
    const out = await rerank('   ', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(h.rerankCalls).toHaveLength(0);
  });

  it('empty docs → empty result, no API call', async () => {
    const out = await rerank('query', [], { apiKey: freshKey() });
    expect(out).toEqual([]);
    expect(h.rerankCalls).toHaveLength(0);
  });

  it('API error → passthrough in original order (outage degrades, never throws)', async () => {
    h.impl = async () => { throw new Error('boom'); };
    const out = await rerank('query', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
  });

  it('API returns no usable results → passthrough rather than dropping everything', async () => {
    h.impl = async () => ({ results: [] });
    const out = await rerank('query', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
  });

  it('passthrough respects topN', async () => {
    const out = await rerank('query', docs, { topN: 2 });
    expect(out.map((r) => r.row)).toEqual(['A', 'B']);
  });
});

describe('scoring timeout (bounded degradation)', () => {
  /** A scorer that takes `ms` before returning index-aligned scores for `docs`. */
  const slowScorer =
    (ms: number, scores: number[] = [0.1, 0.9, 0.5]) =>
    async () => {
      await new Promise((r) => setTimeout(r, ms));
      return scores;
    };

  it('a scorer slower than timeoutMs degrades to retrieval order instead of stalling', async () => {
    const started = Date.now();
    const out = await rerank('query', docs, { engine: 'local', scorer: slowScorer(2_000), timeoutMs: 25 });
    const elapsed = Date.now() - started;

    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
    // The property that makes this a LATENCY guard and not just a fallback:
    // the call is bounded by the timeout, NOT by how slow the scorer was.
    expect(elapsed).toBeLessThan(1_000);
  });

  it('a scorer inside the budget still reranks — the guard never fires early', async () => {
    const out = await rerank('query', docs, { engine: 'local', scorer: slowScorer(5), timeoutMs: 5_000 });
    expect(out.map((r) => r.row)).toEqual(['B', 'C', 'A']);
    expect(out.every((r) => r.reranked === true)).toBe(true);
  });

  it('timeoutMs 0 disables the bound (a batch caller may legitimately want to wait)', async () => {
    const out = await rerank('query', docs, { engine: 'local', scorer: slowScorer(40), timeoutMs: 0 });
    expect(out.map((r) => r.row)).toEqual(['B', 'C', 'A']);
    expect(out.every((r) => r.reranked === true)).toBe(true);
  });

  it('degrades on timeout even when the scorer NEVER settles (the hang case)', async () => {
    const out = await rerank('query', docs, {
      engine: 'local',
      scorer: () => new Promise<number[]>(() => {}),
      timeoutMs: 25,
    });
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
  });

  it('a scorer that REJECTS after the timeout does not raise an unhandled rejection', async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (err: unknown) => unhandled.push(err);
    process.on('unhandledRejection', onUnhandled);
    try {
      const out = await rerank('query', docs, {
        engine: 'local',
        scorer: async () => {
          await new Promise((r) => setTimeout(r, 30));
          throw new Error('late sidecar failure');
        },
        timeoutMs: 10,
      });
      expect(out.every((r) => r.reranked === false)).toBe(true);
      // Let the loser reject and any unhandled-rejection report land.
      await new Promise((r) => setTimeout(r, 80));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});

describe('reranked path', () => {
  it('reorders by the API result and carries calibrated scores', async () => {
    h.impl = async () => ({
      results: [
        { index: 2, relevance_score: 0.9 },
        { index: 0, relevance_score: 0.5 },
        { index: 1, relevance_score: 0.1 },
      ],
    });
    const out = await rerank('query', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['C', 'A', 'B']);
    expect(out.map((r) => r.score)).toEqual([0.9, 0.5, 0.1]);
    expect(out.every((r) => r.reranked === true)).toBe(true);
  });

  it('filters out-of-range indices from the API instead of crashing', async () => {
    h.impl = async () => ({
      results: [
        { index: 7, relevance_score: 0.99 },
        { index: 1, relevance_score: 0.6 },
      ],
    });
    const out = await rerank('query', docs, { apiKey: freshKey() });
    expect(out.map((r) => r.row)).toEqual(['B']);
  });

  it('applies minScore then topN on the reranked list', async () => {
    h.impl = async () => ({
      results: [
        { index: 0, relevance_score: 0.9 },
        { index: 1, relevance_score: 0.7 },
        { index: 2, relevance_score: 0.2 },
      ],
    });
    const out = await rerank('query', docs, { apiKey: freshKey(), minScore: 0.5, topN: 1 });
    expect(out.map((r) => r.row)).toEqual(['A']);
    expect(out[0].score).toBe(0.9);
  });

  it('minScore filtering out every result → passthrough (the empty guard runs after shaping)', async () => {
    // The ordered.length guard runs AFTER minScore/topN shaping, so an
    // all-below-threshold result degrades to retrieval order rather than
    // returning an empty list.
    h.impl = async () => ({
      results: [
        { index: 0, relevance_score: 0.1 },
        { index: 1, relevance_score: 0.2 },
      ],
    });
    const out = await rerank('query', docs, { apiKey: freshKey(), minScore: 0.5 });
    expect(out.map((r) => r.row)).toEqual(['A', 'B', 'C']);
    expect(out.every((r) => r.reranked === false)).toBe(true);
  });

  it('embeds the instruction via zerank-2 XML tags; plain query goes through verbatim', async () => {
    h.impl = async () => ({ results: [{ index: 0, relevance_score: 1 }] });
    await rerank('find shoes', docs, { apiKey: freshKey(), instruction: 'rank products above parts' });
    expect(h.rerankCalls[0].query).toBe('<query>find shoes</query>\n<instruction>rank products above parts</instruction>');
    await rerank('find shoes', docs, { apiKey: freshKey() });
    expect(h.rerankCalls[1].query).toBe('find shoes');
  });

  it('sends doc texts and the default model; honors a model override', async () => {
    h.impl = async () => ({ results: [{ index: 0, relevance_score: 1 }] });
    await rerank('q', docs, { apiKey: freshKey() });
    expect(h.rerankCalls[0].model).toBe('zerank-2');
    expect(h.rerankCalls[0].documents).toEqual(['alpha doc', 'beta doc', 'gamma doc']);
    await rerank('q', docs, { apiKey: freshKey(), model: 'zerank-1' });
    expect(h.rerankCalls[1].model).toBe('zerank-1');
  });

  it('reads the API key from the env when no arg is given, and reuses the cached client per key', async () => {
    h.impl = async () => ({ results: [{ index: 0, relevance_score: 1 }] });
    process.env.ZEROENTROPY_API_KEY = 'env-key-cache-test';
    await rerank('q', docs);
    await rerank('q', docs);
    expect(h.constructed.filter((c) => c.apiKey === 'env-key-cache-test')).toHaveLength(1);
    expect(h.rerankCalls).toHaveLength(2);
  });
});
