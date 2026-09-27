# 3. Enforce the latency budget by shedding quality, not by timing out

- Status: accepted
- Date: 2026-09-27

## Context

The service promises a 50 ms p99. Something has to happen when a request is not going to make it, and
there are three families of answer.

**Time out.** Start the work, watch a deadline, abandon the request when it expires. This is what most
HTTP stacks give you for free. For a cascade it is close to the worst option: the caller has already
waited the whole budget and gets nothing, the server has burned the CPU anyway, and under load every
request pays full cost and then fails — the classic congestion collapse where throughput falls as
offered load rises.

**Shed load.** Reject a fraction of requests at the door so the rest are fast. Correct, and cheap to
implement, but it turns a latency problem into an availability problem. A recommendation request is
not a payment; a slightly worse list of items is obviously better than an error page.

**Shed quality.** Notice before each stage that the planned work does not fit, and do less work.
A cascade is unusually well suited to this because its stages have a natural quality/cost dial each:
`efSearch` trades ANN recall for graph traversal, the rerank depth trades precision for rows scored,
and reranking itself can be dropped entirely in favour of the retrieval order, which is *already a
relevance ranking* — just a less precise one. Nothing about the response becomes invalid.

There is a second question underneath: how does a request know it is in trouble? Wall-clock elapsed
time is not enough, because the expensive stage has not run yet — by the time the clock says the
budget is gone, the money is already spent. The prediction has to happen before the stage.

And a third: what does "in trouble" even mean under concurrency? An early version measured only the
queue wait (the time between the request reaching the event loop and the handler starting in a worker
thread) and still blew p99 at 2x load. The reason is that most of the waiting under real overload
happens *outside* the process — in the accept backlog and in clients' connection pools — where nothing
can be measured from inside. What can be measured is how many accepted requests are in flight.

## Decision

Implement a per-stage budget controller (`serving/budget.py`) that predicts, plans and degrades:

1. **Predict.** Each stage declares how many units of work it plans to do (nodes the HNSW graph will
   visit, candidate rows the ranker will score). A stage's cost is modelled as `fixed + rate * units`,
   where both terms are exponentially weighted moving averages over observed (units, seconds) pairs,
   multiplied by a safety factor so a prediction is pessimistic rather than average. The split is
   fitted online: a sample at the stage's minimum units updates `fixed`, anything larger updates
   `rate`. That is enough for the two-stage shape here and needs no matrix algebra on the request path.
2. **Account for what has already been lost.** `BudgetController.start(spent_seconds, contention)`
   takes the measured queue wait, so a request that waited 30 ms of a 50 ms budget begins with 20 ms.
3. **Account for sharing.** With `C` requests in flight over `N` worker slots, each should expect
   roughly `max(1, C/N)` times its solo cost, so predictions are multiplied by that factor. Under load
   this makes the expensive stage look as expensive as it really is, the ladder engages, responses get
   cheaper, and the backlog drains.
4. **Walk a ladder, cheapest loss of quality first.** Halve `efSearch` (then the candidate count) until
   stage 1 fits; then reduce the rerank depth towards a floor; then skip reranking and return the ANN
   order. Every step is recorded as a reason, the response carries `degraded=True` and the reasons, and
   `degraded_responses_total{reason}` counts them.
5. **Never interrupt.** A stage that has started runs to completion. The controller bounds *planned*
   work only.

The ladder's order follows the measured cost of each step, not intuition: lowering `efSearch` from 768
to 256 costs 0.046 recall@500 (0.9771 to 0.9313) and saves 0.7 ms, while dropping the reranker changes
the ranking itself. Quality is given up in the order that buys the most time per unit of quality lost.

## Consequences

- **Measured.** At 50 RPS on the reference CPU the service holds p50 12.7 ms and p99 20.5 ms with 0
  errors and 0% degraded responses (`bench/results/load.json`; an earlier run taken while the machine
  was also running the test suite measured p99 48.7 ms, which is the box and not the code). At 2x it
  still keeps up, so it correctly does not degrade. At 4x it holds 200.0 of 200 RPS with 0 errors and
  0.1% of responses degraded.
- The invariant CI enforces is the one this decision is really about, and it is deliberately not "does
  it degrade at 2x": **whenever throughput falls behind the offered rate, at least 10% of responses
  must be degraded** (`min_degraded_rate_when_behind`). Falling behind while still planning full work is
  precisely the failure this design exists to prevent, and it is machine-independent in a way an
  absolute p99 is not. It is asserted only for a phase where the *server* fell behind: when the load
  generator's own connection pool is the queue, client-observed latency runs orders of magnitude above
  the server's own view of the same requests at the median or the tail, and the phase is reported as
  harness-limited instead.
- A caller can tell a slow-but-complete response from a fast-but-degraded one, because the flag and
  the reasons are in the body and in the metrics. A dashboard that only watched latency would see the
  4x phase as healthy.
- **The budget can still be overshot.** One pathological call — an unexpectedly slow feature lookup, a
  GC pause inside a stage — is not interrupted. The measured 4x client p99 of 124.7 ms against a
  server-side 38.3 ms is the honest shape of that: the controller keeps *throughput* and *errors*
  intact, and the tail degrades gracefully rather than being clamped.
- Cost estimates are per process and start from documented defaults measured on the reference CPU. A
  fresh process plans with those guesses for the first few requests; a restart resets any drift, which
  is a useful property when an estimator has been polluted by an outlier.
- The estimators add mutable shared state to a hot path. They are behind a lock, the lock is held only
  for arithmetic, and `budget_plan_request` measures the whole plan-a-request path at a p50 of 6.1 us —
  about 0.04% of the budget it protects.
- The two-term cost model is crude. It assumes cost is affine in one unit count per stage, which is
  true enough for HNSW (visits scale with `max(efSearch, k)`) and for a tree ensemble (linear in rows),
  and would not be for a stage whose cost depended on data-dependent branching.

## Alternatives considered

- **A deadline plus cancellation.** Rejected above: the caller pays full latency for nothing. It also
  needs cancellable stages, which numpy, faiss and LightGBM calls are not.
- **Admission control only (a concurrency limiter).** Keeps latency honest by rejecting requests. It is
  complementary rather than alternative — a real deployment would want both — but on its own it turns
  every overload into 503s for a service where a cheaper answer is available.
- **A fixed degradation schedule keyed on a load metric** ("above 80 RPS, skip reranking"). Simpler, and
  wrong in both directions: it degrades a fast machine that did not need to, and misses a slow one that
  did. The point of predicting from measured cost is that the thresholds calibrate themselves.
- **One global budget instead of per-stage allowances.** Tried first. Without per-stage allowances a
  cheap retrieval stage can consume the whole budget by requesting a huge `efSearch`, and the ranking
  stage then has no basis to plan against. Per-stage budgets that sum to less than the total give each
  stage a claim it can be held to.
- **Predicting from the previous request's latency** instead of a fitted cost model. Cheaper, but it
  cannot answer the actual question, which is "how long will *this* amount of work take" — and the
  amount of work is exactly what the controller is choosing.
