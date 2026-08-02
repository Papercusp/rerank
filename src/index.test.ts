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

import { rerank, rerankAvailable, type RerankDoc } from './index';
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
