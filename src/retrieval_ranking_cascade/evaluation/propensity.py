"""Position-bias propensities estimated from result randomization, and IPS click estimators.

Under the position-based click model ``P(click | u, i, rank) = P(examine | rank) * rel(u, i)``, a
naive click-through rate at rank ``r`` measures ``p_r * E[rel]`` and therefore understates the
relevance of anything shown low on the page. Inverse-propensity scoring divides each observed
click by ``p_r`` to recover an unbiased estimate of the relevance term (Joachims, Swaminathan and
Schnabel, "Unbiased Learning-to-Rank with Biased Feedback", WSDM 2017).

Where the propensities come from. A small slice of sessions shows the same candidate slate in a
uniformly random order (the RandTop-n intervention). In that slice rank is independent of
relevance, so ``CTR(rank) / CTR(rank 1)`` is a consistent estimate of ``p_rank / p_1``; fixing
``p_1 = 1`` makes the estimator identified up to that scale, which is all IPS needs. Ranks with
too few randomized impressions fall back to the nearest well-estimated rank rather than to a
noisy ratio, and weights are clipped at ``max_weight`` - clipping trades a little bias for a large
variance reduction, which matters because 1/p grows quickly down the page.

Everything here is estimated from the log only. The true propensities live in the generator's
world and are used solely to grade the estimator in tests and benches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.data.generator import Interactions

__all__ = [
    "PropensityModel",
    "estimate_eta",
    "estimate_propensities",
    "ips_estimate",
    "position_click_rates",
    "snips_estimate",
]

FArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PropensityModel:
    """Examination propensities by 1-based rank, normalized so ``p[0] == 1``."""

    propensities: FArray
    impressions: npt.NDArray[np.int64]
    clicks: npt.NDArray[np.int64]
    max_weight: float

    def __post_init__(self) -> None:
        if self.propensities.ndim != 1 or self.propensities.size == 0:
            raise ValueError("propensities must be a non-empty 1-D array")
        if np.any(self.propensities <= 0.0) or not np.all(np.isfinite(self.propensities)):
            raise ValueError("propensities must be finite and positive")
        if self.max_weight < 1.0:
            raise ValueError("max_weight must be >= 1")

    @property
    def slots(self) -> int:
        return int(self.propensities.shape[0])

    def propensity(self, positions: npt.ArrayLike) -> FArray:
        """Propensity for 1-based positions; positions beyond the slate use the last rank."""
        index = np.asarray(positions, dtype=np.int64) - 1
        if np.any(index < 0):
            raise ValueError("positions are 1-based")
        return self.propensities[np.minimum(index, self.slots - 1)]

    def weight(self, positions: npt.ArrayLike) -> FArray:
        """Clipped inverse propensity weights ``min(1 / p_r, max_weight)``."""
        return np.minimum(1.0 / self.propensity(positions), self.max_weight)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "propensities": [float(x) for x in self.propensities],
            "impressions": [int(x) for x in self.impressions],
            "clicks": [int(x) for x in self.clicks],
        }


def position_click_rates(
    positions: npt.ArrayLike, clicked: npt.ArrayLike, slots: int
) -> tuple[FArray, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per-rank click rate, impression count and click count for 1-based positions."""
    if slots < 1:
        raise ValueError("slots must be >= 1")
    index = np.asarray(positions, dtype=np.int64) - 1
    clicks_in = np.asarray(clicked, dtype=np.int64)
    if index.shape != clicks_in.shape:
        raise ValueError("positions and clicked must have the same shape")
    if index.size and (index.min() < 0 or index.max() >= slots):
        raise ValueError("positions must be within 1..slots")
    impressions = np.bincount(index, minlength=slots).astype(np.int64)
    clicks = np.bincount(index, weights=clicks_in, minlength=slots).astype(np.int64)
    rates = np.divide(
        clicks.astype(np.float64),
        impressions.astype(np.float64),
        out=np.full(slots, np.nan),
        where=impressions > 0,
    )
    return rates, impressions, clicks


def estimate_propensities(
    interactions: Interactions,
    *,
    slots: int,
    min_impressions: int = 200,
    max_weight: float = 20.0,
) -> PropensityModel:
    """Estimate ``p_r / p_1`` from the randomized slice of ``interactions``.

    Raises when the randomized slice cannot identify rank 1, which is the only rank the whole
    scale depends on; that is a data problem the caller must see, not something to paper over.
    """
    randomized = interactions.select(interactions.randomized)
    rates, impressions, clicks = position_click_rates(
        randomized.position, randomized.clicked, slots
    )
    reliable = (impressions >= min_impressions) & np.isfinite(rates) & (rates > 0.0)
    if not reliable[0]:
        raise ValueError(
            "the randomized slice has too few impressions (or no clicks) at rank 1 to "
            "identify the propensity scale"
        )
    known = np.flatnonzero(reliable)
    # Ranks without a reliable estimate inherit the nearest reliable rank's ratio.
    nearest = known[np.abs(known[None, :] - np.arange(slots)[:, None]).argmin(axis=1)]
    ratios = rates[nearest] / rates[0]
    return PropensityModel(
        propensities=np.clip(ratios, 1.0 / max_weight, 1.0),
        impressions=impressions,
        clicks=clicks,
        max_weight=max_weight,
    )


def estimate_eta(model: PropensityModel) -> float:
    """Least-squares exponent of ``p_r = r ** -eta`` fitted to the estimated propensities."""
    ranks = np.arange(1, model.slots + 1, dtype=np.float64)
    usable = ranks > 1.0
    if usable.sum() < 2:
        raise ValueError("at least two ranks beyond the first are needed to fit eta")
    x = np.log(ranks[usable])
    y = np.log(model.propensities[usable])
    return float(-np.dot(x, y) / np.dot(x, x))


def ips_estimate(clicked: npt.ArrayLike, positions: npt.ArrayLike, model: PropensityModel) -> float:
    """Inverse-propensity estimate of the mean relevance of the observed impressions."""
    clicks = np.asarray(clicked, dtype=np.float64)
    if clicks.size == 0:
        return 0.0
    return float(np.mean(clicks * model.weight(positions)))


def snips_estimate(
    clicked: npt.ArrayLike, positions: npt.ArrayLike, model: PropensityModel
) -> float:
    """Self-normalized IPS: the same ratio divided by the mean weight.

    After Swaminathan and Joachims, "The Self-Normalized Estimator for Counterfactual Learning",
    NeurIPS 2015. Self-normalization removes the scale error that clipping and an imperfect
    propensity estimate introduce, at the cost of a small bias; it is the more stable of the two
    on small slices.
    """
    clicks = np.asarray(clicked, dtype=np.float64)
    if clicks.size == 0:
        return 0.0
    weights = model.weight(positions)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(np.dot(clicks, weights) / total)
