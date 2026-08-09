/**
 * Worker script for local-reranker-worker.ts. Runs the ONNX cross-encoder in a
 * dedicated worker_threads thread so a forward pass does not hold the Node.js
 * main event loop.
 *
 * WHY THIS EXISTS (EI-20005411672741677, finished by WI-37555). A cross-encoder
 * forward pass is CPU-bound and holds whatever thread it runs on. While it runs
 * on the MAIN thread nothing else in the process progresses — including the
 * `setTimeout` behind every one of this lib's wall-clock bounds, so a timer
 * cannot preempt it. `scoreCrossEncoder` yields between batches to make those
 * bounds real, but that only bounds the overrun to ONE batch (~720ms at the
 * default 16 pairs). Moving the model here makes the caller's `Promise.race`
 * bound EXACT, because the main thread is free to service its own timers.
 *
 * Protocol (main → worker):
 *   { kind: 'score', id: number, query: string, texts: string[],
 *     model: string, dtype: string, device: string,
 *     maxLength: number, batchSize: number, deadline?: number }
 *
 * Protocol (worker → main):
 *   { kind: 'ready' }                                   on init complete
 *   { kind: 'score_ok',  id: number, scores: number[] } on success
 *   { kind: 'score_err', id: number, error: string }    on failure
 *
 * Plain ESM .mjs because a worker_threads spawn does not go through any
 * TypeScript transform. That means the two constants below and `scoreFromLogits`
 * are DUPLICATED from local-engine.ts and must be kept in sync — the same
 * deliberate trade the local embedder's worker script already makes.
 */

import { parentPort } from 'node:worker_threads';

/** Mirrored from local-engine.ts — see WI-3792. ONNX Runtime otherwise grows a
 *  spin-waiting thread pool sized to EVERY core, per loading process. */
const ORT_SESSION_OPTIONS = { intraOpNumThreads: 4, interOpNumThreads: 1 };

/** One warm model per (id, dtype, device) — loading is seconds, scoring is
 *  milliseconds, so this cache is the difference between usable and unusable. */
const modelsByKey = new Map();

async function getModel(model, dtype, device) {
  const key = `${model}::${dtype}::${device}`;
  let loaded = modelsByKey.get(key);
  if (!loaded) {
    // Dynamic import keeps the worker spawn cheap when @huggingface/transformers
    // is not installed — the package only loads on the first score call.
    loaded = import('@huggingface/transformers').then(async (t) => {
      const [tokenizer, classifier] = await Promise.all([
        t.AutoTokenizer.from_pretrained(model),
        t.AutoModelForSequenceClassification.from_pretrained(model, {
          dtype,
          device,
          session_options: ORT_SESSION_OPTIONS,
        }),
      ]);
      return { tokenizer, model: classifier };
    });
    modelsByKey.set(key, loaded);
    // De-cache a failed load so a transient failure (half-downloaded model, cold
    // network) does not poison this worker for the rest of its life.
    loaded.catch(() => modelsByKey.delete(key));
  }
  return loaded;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Mirrored from local-engine.ts. 1 logit → sigmoid; N logits → softmax over
 *  the 0/1 pair, which is `sigmoid(positive - negative)`. Both monotonic in the
 *  logit, so ordering is unaffected by the squashing. */
function scoreFromLogits(row) {
  if (row.length === 1) return sigmoid(row[0]);
  const [negative, positive] = row;
  return sigmoid(positive - negative);
}

async function score(msg) {
  const { query, texts, model, dtype, device, maxLength, batchSize, deadline } = msg;
  if (texts.length === 0) return [];

  const { tokenizer, model: classifier } = await getModel(model, dtype, device);
  const size = Math.max(1, batchSize);
  const scores = [];

  for (let start = 0; start < texts.length; start += size) {
    // The deadline is re-checked here as well as on the main thread. Off-thread
    // work is not cancellable by the caller abandoning its Promise.race, so
    // without this the worker would keep burning a core scoring a result nobody
    // will ever read — on exactly the host (no sidecar ⇒ in-process) least able
    // to spare one.
    if (deadline !== undefined && Date.now() >= deadline) {
      throw new Error(
        `rerank: scoring deadline exceeded after ${start} of ${texts.length} pairs — degrading to retrieval order`,
      );
    }

    const batch = texts.slice(start, start + size);
    const inputs = tokenizer(new Array(batch.length).fill(query), {
      text_pair: batch,
      padding: true,
      truncation: true,
      max_length: maxLength,
    });
    const { logits } = await classifier(inputs);

    const numLabels = logits.dims.length > 1 ? logits.dims[logits.dims.length - 1] : 1;
    if (logits.data.length !== batch.length * numLabels) {
      throw new Error(
        `rerank: logits length ${logits.data.length} != ${batch.length} pairs x ${numLabels} labels`,
      );
    }
    for (let i = 0; i < batch.length; i++) {
      const row = [];
      for (let l = 0; l < numLabels; l++) row.push(Number(logits.data[i * numLabels + l]));
      scores.push(scoreFromLogits(row));
    }
  }
  return scores;
}

parentPort.on('message', async (msg) => {
  if (!msg || typeof msg !== 'object') return;
  if (msg.kind !== 'score') return;
  try {
    parentPort.postMessage({ kind: 'score_ok', id: msg.id, scores: await score(msg) });
  } catch (err) {
    parentPort.postMessage({
      kind: 'score_err',
      id: msg.id,
      error: err && err.message ? err.message : String(err),
    });
  }
});

// Ready as soon as the handler is installed; the model loads lazily on first use.
parentPort.postMessage({ kind: 'ready' });
