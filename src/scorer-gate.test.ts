/**
 * Admission control for the shared in-process scorer (WI-37676).
 *
 * Every shedding assertion here is paired with a control that a
 * shed-everything implementation would fail — "refuse every call" satisfies the
 * headline property while destroying the feature, and for a search stage that
 * means silently returning retrieval order for every query forever. The two
 * directions are called out on each pair.
 */

import { afterEach, describe, expect, it } from 'vitest';

import {
  RerankDeadlineError,
  _resetScorerGateForTest,
  _setScorerGateCalibrationForTest,
  _setScorerGateConcurrencyForTest,
  getScorerGateState,
  isDeadlineFailure,
  runInScorerGate,
} from './scorer-gate';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

afterEach(() => _resetScorerGateForTest());

describe('scorer admission gate', () => {
  it('serializes concurrent callers instead of letting them contend', async () => {
    _setScorerGateConcurrencyForTest(1);
    let inFlight = 0;
    let peak = 0;

    const job = async () => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await sleep(15);
      inFlight--;
    };

    await Promise.all(Array.from({ length: 5 }, () => runInScorerGate(10, undefined, job)));

    // Measured: concurrency buys zero throughput on this scorer, so overlapping
    // callers only spread the same completion time across everyone.
    expect(peak).toBe(1);
    expect(getScorerGateState().active).toBe(0);
    expect(getScorerGateState().queued).toBe(0);
  });

  it('sheds a call that provably cannot meet its deadline, WITHOUT running it', async () => {
    _setScorerGateConcurrencyForTest(1);
    _setScorerGateCalibrationForTest(50); // 50ms per pair

    let ran = false;
    // 100 pairs x 50ms = 5000ms of work against 200ms of budget.
    const attempt = runInScorerGate(100, Date.now() + 200, async () => {
      ran = true;
      return 'scored';
    });

    await expect(attempt).rejects.toBeInstanceOf(RerankDeadlineError);
    // Not running it is the entire point: the alternative is spending the whole
    // budget to arrive at the same passthrough several seconds later.
    expect(ran).toBe(false);
  });

  it('CONTROL: admits a call that comfortably fits its budget', async () => {
    _setScorerGateConcurrencyForTest(1);
    _setScorerGateCalibrationForTest(1); // 1ms per pair

    // 24 pairs x 1ms = 24ms against 4000ms — the ordinary case. A gate that
    // sheds this would return retrieval order for every query forever while
    // still passing the shedding test above.
    await expect(runInScorerGate(24, Date.now() + 4000, async () => 'scored')).resolves.toBe(
      'scored',
    );
  });

  it('CONTROL: never sheds before it has measured anything', async () => {
    _setScorerGateConcurrencyForTest(1);
    _resetScorerGateForTest(); // msPerPair === null
    expect(getScorerGateState().msPerPair).toBeNull();

    // An impossible-looking budget, but with no calibration the gate has no
    // basis for the verdict. Shedding from a guess would be a manufactured
    // failure — and it would fire hardest at process start, when nothing has
    // been measured yet and every first search would degrade.
    await expect(runInScorerGate(1000, Date.now() + 1, async () => 'scored')).resolves.toBe(
      'scored',
    );
  });

  it('sheds at the head of the queue when the wait made the call undeliverable', async () => {
    _setScorerGateConcurrencyForTest(1);
    _setScorerGateCalibrationForTest(10);

    let secondRan = false;
    // Admitted first and holds the gate for 400ms; its own budget is fine.
    const first = runInScorerGate(5, Date.now() + 10_000, async () => {
      await sleep(400);
      return 'first';
    });
    // Queued behind it. On ARRIVAL the estimate is (5 active + 5 mine) x 10ms =
    // 100ms against a 250ms budget, so it is admitted to the queue rather than
    // refused — the arrival check alone would let it through. It is the RE-CHECK
    // at the head, ~400ms later with the budget long gone, that catches it.
    const second = runInScorerGate(5, Date.now() + 250, async () => {
      secondRan = true;
      return 'second';
    });

    await expect(first).resolves.toBe('first');
    await expect(second).rejects.toBeInstanceOf(RerankDeadlineError);
    expect(secondRan).toBe(false);
  });

  it('drops the first observation, which pays the one-time model load', async () => {
    _setScorerGateConcurrencyForTest(1);
    _resetScorerGateForTest();

    // A cold first call is dominated by loading a ~150M model, not by per-pair
    // cost. Seeding the estimator from it would inflate it by an order of
    // magnitude and shed everything that followed.
    await runInScorerGate(1, undefined, async () => {
      await sleep(40);
    });
    expect(getScorerGateState().observations).toBe(1);
    expect(getScorerGateState().msPerPair).toBeNull();

    await runInScorerGate(1, undefined, async () => {
      await sleep(10);
    });
    expect(getScorerGateState().msPerPair).not.toBeNull();
    // Calibrated from the WARM call, so nowhere near the 40ms cold sample.
    expect(getScorerGateState().msPerPair!).toBeLessThan(30);
  });

  it('releases the gate when the job throws, rather than wedging every later call', async () => {
    _setScorerGateConcurrencyForTest(1);

    await expect(
      runInScorerGate(5, undefined, async () => {
        throw new Error('engine exploded');
      }),
    ).rejects.toThrow('engine exploded');

    expect(getScorerGateState().active).toBe(0);
    // A gate that leaked its slot on the failure path would deadlock the whole
    // search stage after one engine error — worse than the bug it fixes.
    await expect(runInScorerGate(5, undefined, async () => 'ok')).resolves.toBe('ok');
  });

  it('honours a concurrency override for hosts where scoring does parallelize', async () => {
    _setScorerGateConcurrencyForTest(3);
    let inFlight = 0;
    let peak = 0;

    await Promise.all(
      Array.from({ length: 6 }, () =>
        runInScorerGate(1, undefined, async () => {
          inFlight++;
          peak = Math.max(peak, inFlight);
          await sleep(15);
          inFlight--;
        }),
      ),
    );

    expect(peak).toBe(3);
  });

  it('classifies a shed as a DEADLINE failure, not an engine outage', () => {
    // `scoreCrossEncoder` uses this to decide whether to retry inline. Retrying
    // a shed inline would burn the budget that was just proven insufficient, on
    // the main thread — the opposite of the fix.
    expect(isDeadlineFailure(new RerankDeadlineError('rerank: scoring deadline exceeded …'))).toBe(
      true,
    );
    // The worker's throw crosses a thread boundary and arrives as a plain Error.
    expect(isDeadlineFailure(new Error('rerank: scoring deadline exceeded after 3 of 24 pairs'))).toBe(
      true,
    );
    // CONTROL: a genuine engine failure must NOT be read as a budget verdict,
    // or an outage would be reported to callers as "too slow".
    expect(isDeadlineFailure(new Error('onnxruntime: failed to load model'))).toBe(false);
  });
});
