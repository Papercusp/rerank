import { describe, it, expect, vi, beforeEach } from 'vitest';

/**
 * T2.6 — rerank() fail-safe + filtering, complementing rerank.spec.ts
 * (timeout / success / API-error / no-key). Covers empty query/docs passthrough,
 * minScore/topN filtering, the "nothing usable → fall back" guard, and that it
 * never throws.
 */

const rerankMock = vi.fn();
vi.mock('zeroentropy', () => ({
  default: class {
    models = { rerank: (...args: unknown[]) => rerankMock(...args) };
  },
}));

import { rerank } from './index';

const docs = [
  { id: 'a', text: 'alpha', row: { id: 'a' } },
  { id: 'b', text: 'beta', row: { id: 'b' } },
];

beforeEach(() => {
  rerankMock.mockReset();
  process.env.ZEROENTROPY_API_KEY = 'test-key';
});

describe('rerank() passthrough conditions', () => {
  it('does not call the API for an empty/whitespace query', async () => {
    const res = await rerank('   ', docs, {});
    expect(rerankMock).not.toHaveBeenCalled();
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
    expect(res.every((r) => !r.reranked)).toBe(true);
  });

  it('returns [] for empty docs without calling the API', async () => {
    const res = await rerank('q', [], {});
    expect(rerankMock).not.toHaveBeenCalled();
    expect(res).toEqual([]);
  });

  it('applies topN even on the passthrough path', async () => {
    delete process.env.ZEROENTROPY_API_KEY;
    const res = await rerank('q', docs, { topN: 1 });
    expect(res.map((r) => r.row.id)).toEqual(['a']);
  });
});

describe('rerank() filtering on a successful response', () => {
  it('drops results below minScore', async () => {
    rerankMock.mockResolvedValue({
      results: [
        { index: 0, relevance_score: 0.9 },
        { index: 1, relevance_score: 0.2 },
      ],
    });
    const res = await rerank('q', docs, { minScore: 0.5 });
    expect(res.map((r) => r.row.id)).toEqual(['a']);
    expect(res[0].reranked).toBe(true);
  });

  it('truncates to topN (preserving the API best-first order)', async () => {
    rerankMock.mockResolvedValue({
      results: [
        { index: 1, relevance_score: 0.9 },
        { index: 0, relevance_score: 0.8 },
      ],
    });
    const res = await rerank('q', docs, { topN: 1 });
    expect(res.map((r) => r.row.id)).toEqual(['b']);
  });

  it('falls back to original order when minScore filters everything out', async () => {
    rerankMock.mockResolvedValue({
      results: [
        { index: 0, relevance_score: 0.1 },
        { index: 1, relevance_score: 0.2 },
      ],
    });
    const res = await rerank('q', docs, { minScore: 0.95 });
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
    expect(res.every((r) => !r.reranked)).toBe(true);
  });

  it('falls back when the API returns no usable results, ignoring out-of-range indices', async () => {
    rerankMock.mockResolvedValue({ results: [{ index: 99, relevance_score: 0.9 }] });
    const res = await rerank('q', docs, {});
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
    expect(res.every((r) => !r.reranked)).toBe(true);
  });
});

describe('rerank() never throws', () => {
  it('returns passthrough when the SDK throws synchronously', async () => {
    rerankMock.mockImplementation(() => {
      throw new Error('sync boom');
    });
    const res = await rerank('q', docs, {});
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
  });

  it('returns passthrough when the SDK resolves a malformed payload', async () => {
    rerankMock.mockResolvedValue({ unexpected: true });
    const res = await rerank('q', docs, {});
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
  });
});
