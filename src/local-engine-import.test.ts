import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.doUnmock('@huggingface/transformers');
  vi.resetModules();
});

describe('the default optional transformer loader in a module runner', () => {
  it('detects and loads the installed module without a loader override or model download', async () => {
    const tokenizer = vi.fn(() => ({}));
    const model = vi.fn(async () => ({ logits: { dims: [1, 1], data: [2] } }));
    const tokenizeFactory = vi.fn(async () => tokenizer);
    const modelFactory = vi.fn(async () => model);
    vi.doMock('@huggingface/transformers', () => ({
      AutoTokenizer: { from_pretrained: tokenizeFactory },
      AutoModelForSequenceClassification: { from_pretrained: modelFactory },
    }));
    const engine = await import('./local-engine');
    expect(await engine.localRerankAvailable()).toBe(true);
    const loaded = await engine.loadCrossEncoder({ model: 'fixture-model', device: 'cpu', dtype: 'q8' });
    expect(loaded.tokenizer).toBe(tokenizer);
    expect(loaded.model).toBe(model);
    expect(tokenizeFactory).toHaveBeenCalledWith('fixture-model');
    expect(modelFactory).toHaveBeenCalledWith('fixture-model', {
      device: 'cpu', dtype: 'q8', session_options: engine.ORT_SESSION_OPTIONS,
    });
  });

  it('preserves a genuine optional-module resolution failure', async () => {
    const missing = Object.assign(new Error('Cannot find optional transformer module'), { code: 'ERR_MODULE_NOT_FOUND' });
    vi.doMock('@huggingface/transformers', () => { throw missing; });
    const engine = await import('./local-engine');
    expect(await engine.localRerankAvailable()).toBe(false);
    await expect(engine.loadCrossEncoder({ model: 'missing-model', device: 'cpu', dtype: 'q8' }))
      .rejects.toMatchObject({ cause: missing });
  });
});
