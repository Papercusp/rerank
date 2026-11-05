/** Deterministic token encoding for fake models; no model download or inference. */
export function withTestTokenEncoding<T extends object>(tokenize: T) {
  return Object.assign(tokenize, {
    encode(text: string, opts: { text_pair?: string; add_special_tokens?: boolean } = {}) {
      const ids = (value: string) => Array.from(value, c => c.codePointAt(0)! + 4);
      return opts.add_special_tokens === false ? ids(text) : [1, ...ids(text), 2, ...ids(opts.text_pair ?? ''), 3];
    },
    decode: (ids: number[]) => String.fromCodePoint(...ids.map(id => id - 4)),
  });
}
