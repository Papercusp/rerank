import { describe, it, expect, vi, beforeEach } from 'vitest';

// Control zerank latency by mocking the ZeroEntropy SDK. `import ZeroEntropy
// from 'zeroentropy'` resolves to this default-exported class; each instance's
// models.rerank delegates to the shared mock so tests can make it resolve,
// reject, or hang.
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

describe('rerank() timeout + fail-safe', () => {
  it('falls back to retrieval order when the rerank call hangs past timeoutMs', async () => {
    rerankMock.mockImplementation(() => new Promise(() => {})); // never settles
    const res = await rerank('q', docs, { timeoutMs: 50 });
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']); // original order preserved
    expect(res.every((r) => !r.reranked)).toBe(true); // marked as passthrough
  });

  it('returns the reranked order when the call resolves within the cap', async () => {
    rerankMock.mockResolvedValue({
      results: [
        { index: 1, relevance_score: 0.9 },
        { index: 0, relevance_score: 0.1 },
      ],
    });
    const res = await rerank('q', docs, { timeoutMs: 1000 });
    expect(res.map((r) => r.row.id)).toEqual(['b', 'a']); // reordered by score
    expect(res.every((r) => r.reranked)).toBe(true);
  });

  it('falls back to retrieval order on an API error', async () => {
    rerankMock.mockRejectedValue(new Error('boom'));
    const res = await rerank('q', docs, { timeoutMs: 1000 });
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
    expect(res.every((r) => !r.reranked)).toBe(true);
  });

  it('does not call the API when there is no key (pure passthrough)', async () => {
    delete process.env.ZEROENTROPY_API_KEY;
    const res = await rerank('q', docs, {});
    expect(rerankMock).not.toHaveBeenCalled();
    expect(res.map((r) => r.row.id)).toEqual(['a', 'b']);
  });
});
