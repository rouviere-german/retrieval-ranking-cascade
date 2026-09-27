# 5. Train the reranker on IPS-weighted clicks rather than raw clicks or LightGBM's unbiased lambdarank

- Status: accepted
- Date: 2026-09-27

## Context

The reranker's labels are clicks, and clicks are not relevance. Under the position-based model a click
is `P(examine | rank) * relevance`, so with an examination exponent of 1 an item at rank 10 is examined
a tenth as often as one at rank 1. Anything trained on raw clicks therefore learns the logging policy's
ordering — which in this world is mostly popularity — and calls it relevance.

The measurement is unambiguous. Mean relevance per rank in the log, estimated naively from clicks,
falls from 0.6008 at rank 1 to 0.0273 at rank 10, while the true relevance of the items shown at those
ranks only falls from 0.6023 to 0.2783. Most of that apparent decay is examination, not quality. A
lambdarank objective fed those labels spends its gradient on reproducing it.

Three options were available, and the choice needed data rather than argument.

1. **Raw clicks.** Do nothing. Cheapest, and the baseline any correction has to beat.
2. **IPS row weights.** Multiply each shown row's contribution by the clipped inverse of its logged
   position's examination propensity, so the objective estimates NDCG over relevance rather than over
   clicks (Joachims, Swaminathan and Schnabel, "Unbiased Learning-to-Rank with Biased Feedback",
   WSDM 2017). Requires propensities, which this world provides through a randomized slice.
3. **LightGBM's unbiased lambdarank.** Hand the logged positions to `Dataset(position=...)` and let
   LightGBM fit position factors jointly with the model (after Hu et al., "Unbiased LambdaMART",
   WWW 2019). Attractive because it needs no propensity estimation at all.

## Decision

Use IPS row weights by default (`RankerConfig.ips_weights = True`). Keep the other two as ablations
rather than deleting them, because the comparison is part of what the repo is for.

Propensities come from result randomization: a 5% slice of sessions shows the same candidate slate in
uniformly random order, so within that slice rank is independent of relevance and `CTR(r) / CTR(1)`
estimates `p_r / p_1`. Fixing `p_1 = 1` identifies the curve up to the one scale IPS does not need.
Ranks with too few randomized impressions inherit the nearest reliable rank rather than a noisy ratio,
and weights are clipped at 20 — clipping trades a little bias for a large variance reduction, which
matters because `1/p` grows quickly down the page. Estimation **raises** rather than degrading when the
randomized slice cannot identify rank 1, because that is a data problem the caller has to see.

Rows for items the tower retrieved but that were never shown keep weight 1: they were never examined,
so there is no propensity to invert.

Serving scores never include any of this. The correction exists to make the training objective estimate
relevance; at serving time the model already does.

## Consequences

Measured with `uv run python bench/bench_quality.py --check --ablations`, one training run per row, on
the `small` world:

| labels | corpus NDCG@10 | slate NDCG truth | slate NDCG naive | `tower_score` gain share |
|---|---:|---:|---:|---:|
| raw clicks | 0.5672 | 0.8237 | 0.6933 | 0.135 |
| **IPS weights** | **0.6054** | 0.8544 | 0.6908 | 0.225 |
| LightGBM unbiased | 0.4709 | **0.8722** | 0.5396 | 0.140 |

- IPS weighting is worth **+0.038 corpus NDCG** over raw clicks (0.5672 to 0.6054), and +0.031 on the
  slate metric. It also raises the share of the model's total gain attributed to `tower_score` from
  0.135 to 0.225, which is the mechanism made visible: with the position bias divided out, the
  relevance-carrying feature earns more of the splits and the popularity-carrying features earn fewer.
- **LightGBM's unbiased lambdarank wins the slate metric by the largest margin of the three and loses
  the corpus metric by an even larger one** (0.4709, worse than raw clicks). The two metrics disagree
  because they ask different questions: the slate protocol only ever scores items the logging policy
  chose, and a model that has absorbed a position correction fitted *on that same policy's slates* is
  very good at reordering them and has learned something that does not generalize to the catalogue.
  Had only the slate protocol been measured, this would have looked like the best option. Measuring two
  protocols is what made the choice decidable, and this row is the strongest argument in the repo for
  reporting full-corpus metrics next to slate metrics.
- The default costs a dependency on the randomized slice. Removing the randomization would make IPS
  unavailable; the slice is 5% of sessions shown in a random order, which is a real (if small) cost in
  production quality that a team would have to accept.
- A wrong propensity estimate becomes a wrong weight, silently. Two guards: the estimator refuses to
  run when rank 1 is unidentified, and the bench reports the recovered exponent against the truth
  (1.024 against 1.0) so a broken estimator shows up as a metric, not as a slow quality drift.
- Clipping at 20 leaves residual bias at the bottom of the page. The bench reports the remaining
  per-rank error (0.0373 against a naive 0.2513) rather than claiming the correction is exact.
- **The correction does not make the reranker beat the tower on the full corpus** (0.6054 against
  0.6205). It closes a large part of the gap that raw clicks leave, and the remaining limitation is the
  training distribution rather than the label treatment — the `exact_rank` upper bound of 0.6264 shows
  the model itself, applied to every item, barely improves on the tower. That is recorded in the model
  card's Caveats and in the README's Non-goals.
- Both alternatives stay in the code as `RankerConfig` flags with an ablation that exercises them, so
  the claim above is re-measured on every quality bench run rather than being a note about something
  that was once tried.

## Alternatives considered

Covered above. Two further options were considered and dropped:

- **A learned click model (e.g. regression EM / DBN) instead of randomization.** Avoids the cost of a
  randomized slice by estimating propensities from the logs' natural variation. It needs enough
  position variation for the same (user, item) pair to be identifiable, and it makes the propensities
  depend on a second model whose own bias would then need measuring. The randomized slice is a much
  cleaner estimand and this world can afford it.
- **Dropping click labels entirely and training on the generator's ground-truth relevance.** Would
  produce better numbers and destroy the point: no real system has the labels, and the whole bias
  apparatus would become untestable. Ground truth is used only to *grade* estimators, never to fit one.
