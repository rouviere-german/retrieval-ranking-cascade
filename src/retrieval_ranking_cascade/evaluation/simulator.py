"""Feedback simulator: turn served lists back into clicks using the world's click model.

Offline there is no traffic, so the only way to close the loop - serve a list, observe clicks,
report CTR - is to let the same position-based click model that produced the training log judge
the new lists. ``P(click) = P(examine | rank) * rel(u, i)`` with the world's true relevance.

The simulated CTR of a *new* policy is not a naive replay of logged clicks: the items are the ones
the policy actually chose, so a better ranker genuinely earns more clicks. It is still a
simulation, not an A/B test: it assumes the click model, and the click model is an assumption. The
number belongs next to the ground-truth relevance metrics, never on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retrieval_ranking_cascade.data.generator import World

__all__ = ["FeedbackSimulator", "SimulationResult"]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Outcome of simulating clicks over a batch of served lists."""

    impressions: int
    clicks: int
    users: int
    mean_relevance: float
    position_ctr: tuple[float, ...]

    @property
    def ctr(self) -> float:
        return float(self.clicks / self.impressions) if self.impressions else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "users": self.users,
            "ctr": round(self.ctr, 6),
            "mean_relevance": round(self.mean_relevance, 6),
            "position_ctr": [round(value, 6) for value in self.position_ctr],
        }


class FeedbackSimulator:
    """Draws clicks for served lists from the world's position-based click model."""

    def __init__(self, world: World, *, seed: int = 0) -> None:
        self._world = world
        self._rng = np.random.default_rng(seed)

    def simulate_list(
        self, user_id: int, item_ids: Sequence[int] | npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int8]:
        """Click outcome per served position (1-based positions, in list order)."""
        items = np.asarray(item_ids, dtype=np.int64)
        if items.size == 0:
            return np.zeros(0, dtype=np.int8)
        examine = self._world.examination(np.arange(1, items.shape[0] + 1))
        relevance = self._world.relevance(np.full(items.shape, user_id), items)
        draws = self._rng.random(items.shape[0])
        return (draws < examine * relevance).astype(np.int8)

    def simulate(self, served: Sequence[tuple[int, Sequence[int]]]) -> SimulationResult:
        """Simulate every (user, item list) pair and aggregate CTR by position."""
        if not served:
            return SimulationResult(0, 0, 0, 0.0, ())
        width = max(len(items) for _, items in served)
        clicks_by_position = np.zeros(width, dtype=np.int64)
        impressions_by_position = np.zeros(width, dtype=np.int64)
        relevance_total = 0.0
        impressions = 0
        for user_id, items in served:
            outcome = self.simulate_list(user_id, items)
            length = outcome.shape[0]
            if length == 0:
                continue
            clicks_by_position[:length] += outcome
            impressions_by_position[:length] += 1
            relevance_total += float(
                self._world.relevance(
                    np.full(length, user_id), np.asarray(items, dtype=np.int64)
                ).sum()
            )
            impressions += length
        position_ctr = tuple(
            float(clicks / impressions_at) if impressions_at else 0.0
            for clicks, impressions_at in zip(
                clicks_by_position, impressions_by_position, strict=True
            )
        )
        total_clicks = int(clicks_by_position.sum())
        return SimulationResult(
            impressions=impressions,
            clicks=total_clicks,
            users=len(served),
            mean_relevance=relevance_total / impressions if impressions else 0.0,
            position_ctr=position_ctr,
        )
