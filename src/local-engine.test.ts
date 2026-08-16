/**
 * The local ONNX cross-encoder engine.
 *
 * Seam: `_setLoaderForTest` swaps the model loader, so every one of these runs
 * the REAL engine logic (pair construction, batching, logit→score collapse,
 * cache keying, fail-safe conversion) with a fake tokenizer/model in place of
 * ONNX. Nothing here downloads a model or runs inference — that belongs to the
 * live verification item, not to a unit test.
 *
 * What is actually load-bearing here: `localEngineScores` must NEVER throw and
 * must return `null` on any failure, because that null is what makes a rerank
 * outage degrade to retrieval order instead of breaking search.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  DEFAULT_MAX_LENGTH,
  LOCAL_RERANKER_MODEL,
  _setLoaderForTest,
  loadCrossEncoder,
  localEngineScores,
  scoreCrossEncoder,
} from './local-engine';

interface TokenizeCall {
  text: string[];
  text_pair: string[];
  padding: boolean;
  truncation: boolean;
  max_length: number;
}

const calls = {
  loads: [] as Array<{ model: string; dtype: string }>,
  tokenize: [] as TokenizeCall[],
  forward: [] as number[],
};

/**
 * Fake model. `logitsFor` maps a batch of doc texts to the flat logits the ONNX
 * graph would emit; `numLabels` shapes `dims` the way transformers.js does.
 */
function fakeLoader(opts: {
  logitsFor?: (texts: string[]) => number[];
  numLabels?: number;
  loadError?: Error;
  dimsOverride?: number[];
}) {
  return async (model: string, dtype: string) => {
    calls.loads.push({ model, dtype });
    if (opts.loadError) throw opts.loadError;
    const numLabels = opts.numLabels ?? 1;
    return {
      tokenizer: ((text: string[], o: Omit<TokenizeCall, 'text'>) => {
        calls.tokenize.push({ text, ...o });
        return { __texts: o.text_pair };
      }) as never,
      model: (async (inputs: Record<string, unknown>) => {
        const texts = inputs.__texts as string[];
        calls.forward.push(texts.length);
        const data = opts.logitsFor
          ? opts.logitsFor(texts)
          : texts.map((_, i) => i).flatMap((i) => new Array(numLabels).fill(i));
        return { logits: { dims: opts.dimsOverride ?? [texts.length, numLabels], data } };
      }) as never,
    };
  };
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

beforeEach(() => {
  calls.loads.length = 0;
  calls.tokenize.length = 0;
  calls.forward.length = 0;
});

afterEach(() => {
  _setLoaderForTest(null);
});

describe('scoreCrossEncoder', () => {
  it('pairs the query against every doc and returns index-aligned sigmoid scores', async () => {
    _setLoaderForTest(fakeLoader({ logitsFor: (t) => t.map((x) => (x === 'relevant' ? 4 : -4)) }));

    const scores = await scoreCrossEncoder('q', ['junk', 'relevant', 'junk']);

    expect(scores).toEqual([sigmoid(-4), sigmoid(4), sigmoid(-4)]);
    // The cross-encoder scores a PAIR — the query is repeated per doc, and the
    // docs go in as `text_pair`, never concatenated into one string.
    expect(calls.tokenize[0].text).toEqual(['q', 'q', 'q']);
    expect(calls.tokenize[0].text_pair).toEqual(['junk', 'relevant', 'junk']);
    expect(calls.tokenize[0].truncation).toBe(true);
    expect(calls.tokenize[0].padding).toBe(true);
    expect(calls.tokenize[0].max_length).toBe(DEFAULT_MAX_LENGTH);
  });

  it('empty input short-circuits — no model load, no forward pass', async () => {
    _setLoaderForTest(fakeLoader({}));
    expect(await scoreCrossEncoder('q', [])).toEqual([]);
    expect(calls.loads).toHaveLength(0);
  });

  it('batches, and the concatenated scores stay index-aligned across batches', async () => {
    _setLoaderForTest(fakeLoader({ logitsFor: (t) => t.map((x) => Number(x)) }));

    const texts = ['0', '1', '2', '3', '4'];
    const scores = await scoreCrossEncoder('q', texts, { batchSize: 2 });

    expect(calls.forward).toEqual([2, 2, 1]);
    expect(scores).toEqual([0, 1, 2, 3, 4].map(sigmoid));
  });

  it('a two-label head is collapsed by the positive-vs-negative margin, not read raw', async () => {
    _setLoaderForTest(
      // rows: [neg, pos] → doc A clearly relevant, doc B clearly not.
      fakeLoader({ numLabels: 2, logitsFor: () => [-1, 3, 2, -2] }),
    );

    const scores = await scoreCrossEncoder('q', ['A', 'B']);

    expect(scores).toEqual([sigmoid(3 - -1), sigmoid(-2 - 2)]);
    expect(scores[0]).toBeGreaterThan(scores[1]);
  });

  it('a 1-D [batch] logits export is read as one logit per pair', async () => {
    _setLoaderForTest(fakeLoader({ dimsOverride: [2], logitsFor: () => [1, -1] }));
    expect(await scoreCrossEncoder('q', ['A', 'B'])).toEqual([sigmoid(1), sigmoid(-1)]);
  });

  it('throws when the logits shape disagrees with the batch size', async () => {
    // A silent mis-shape would misalign EVERY score with its document — worse
    // than an error, because the ordering would look plausible and be wrong.
    _setLoaderForTest(fakeLoader({ logitsFor: () => [1] }));
    await expect(scoreCrossEncoder('q', ['A', 'B'])).rejects.toThrow(/logits length/);
  });
});

describe('model cache', () => {
  it('loads once per (model, dtype) and reuses the warm model', async () => {
    _setLoaderForTest(fakeLoader({}));

    await scoreCrossEncoder('q', ['a']);
    await scoreCrossEncoder('q', ['b']);

    expect(calls.loads).toHaveLength(1);
    expect(calls.forward).toHaveLength(2);
  });

  it('keys on dtype and model separately', async () => {
    _setLoaderForTest(fakeLoader({}));

    await scoreCrossEncoder('q', ['a']);
    await scoreCrossEncoder('q', ['a'], { dtype: 'fp32' });
    await scoreCrossEncoder('q', ['a'], { model: 'other/model' });

    expect(calls.loads).toHaveLength(3);
    expect(calls.loads.map((l) => l.dtype)).toEqual(['q8', 'fp32', 'q8']);
    expect(calls.loads.map((l) => l.model)).toEqual([
      LOCAL_RERANKER_MODEL,
      LOCAL_RERANKER_MODEL,
      'other/model',
    ]);
  });

  it('a failed load is NOT cached — a transient failure must not poison the process', async () => {
    // The whole point: a cold network or a half-downloaded model would
    // otherwise disable reranking for this process's entire lifetime.
    _setLoaderForTest(fakeLoader({ loadError: new Error('cold network') }));
    await expect(loadCrossEncoder()).rejects.toThrow('cold network');

    _setLoaderForTest(fakeLoader({}));
    await expect(scoreCrossEncoder('q', ['a'])).resolves.toHaveLength(1);
  });
});

describe('localEngineScores — the fail-safe boundary', () => {
  it('returns scores on the happy path', async () => {
    _setLoaderForTest(fakeLoader({ logitsFor: () => [2] }));
    expect(await localEngineScores('q', ['a'])).toEqual([sigmoid(2)]);
  });

  it('returns null (never throws) when the model cannot load', async () => {
    _setLoaderForTest(fakeLoader({ loadError: new Error('no onnx runtime') }));
    expect(await localEngineScores('q', ['a'])).toBeNull();
  });

  it('returns null (never throws) when inference produces a bad shape', async () => {
    _setLoaderForTest(fakeLoader({ logitsFor: () => [1, 2, 3] }));
    expect(await localEngineScores('q', ['a'])).toBeNull();
  });
});
