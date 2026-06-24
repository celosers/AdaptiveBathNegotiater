"""Simple time-based conceder for ANL2026 baseline comparisons."""

from __future__ import annotations
from typing import Any
import random

from negmas.sao import SAONegotiator
from negmas import ResponseType


class SimpleNegotiator(SAONegotiator):
    """A straightforward time-based negotiator for baseline comparisons."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._best: Any = None
        self._initialized = False
        self._received_offers: list[Any] = []

    def on_preferences_changed(self, changes: Any) -> None:
        self._init()
        try:
            super().on_preferences_changed(changes)
        except Exception:
            pass

    def _init(self) -> None:
        if self._initialized:
            return
        try:
            outcomes = list(self.nmi.outcomes)
            if outcomes:
                self._best = max(outcomes, key=lambda o: float(self.ufun(o)))
        except Exception:
            self._best = None
        self._initialized = True

    def _relative_time(self, state: Any) -> float:
        t = getattr(state, "relative_time", None)
        if t is not None:
            return max(0.0, min(1.0, float(t)))
        return 0.0

    def _u(self, outcome: Any) -> float:
        try:
            return float(self.ufun(outcome))
        except Exception:
            return 0.0

    def respond(self, state: Any, offer: Any | None = None, source: str = "") -> Any:
        self._init()
        if offer is None:
            offer = getattr(state, "current_offer", None)
        if offer is None:
            return ResponseType.REJECT_OFFER

        t = self._relative_time(state)
        threshold = 0.85 * (1.0 - t ** 3.0)

        if self._u(offer) >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: Any) -> Any:
        self._init()
        t = self._relative_time(state)
        target = 1.0 - t ** 2.5

        try:
            outcomes = list(self.nmi.outcomes)
            candidates = [o for o in outcomes if self._u(o) >= target * 0.9]
            if candidates:
                return max(candidates, key=lambda o: self._u(o))
        except Exception:
            pass
        return self._best


class ShadowBathNegotiator(SAONegotiator):
    """Opponent with opponent-modeling capability for ablation studies.
    
    Models opponent preferences via frequency counting over received offers,
    enabling privacy (tau) measurement with real differentiation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._best: Any = None
        self._initialized = False
        self._received_offers: list[Any] = []
        self._issue_value_counts: dict[int, dict[Any, int]] = {}

    def on_preferences_changed(self, changes: Any) -> None:
        self._init()
        try:
            super().on_preferences_changed(changes)
        except Exception:
            pass

    def _init(self) -> None:
        if self._initialized:
            return
        try:
            outcomes = list(self.nmi.outcomes)
            if outcomes:
                self._best = max(outcomes, key=lambda o: float(self.ufun(o)))
        except Exception:
            self._best = None
        self._initialized = True

    def _relative_time(self, state: Any) -> float:
        t = getattr(state, "relative_time", None)
        if t is not None:
            return max(0.0, min(1.0, float(t)))
        return 0.0

    def _u(self, outcome: Any) -> float:
        try:
            return float(self.ufun(outcome))
        except Exception:
            return 0.0

    def respond(self, state: Any, offer: Any | None = None, source: str = "") -> Any:
        self._init()
        if offer is None:
            offer = getattr(state, "current_offer", None)
        if offer is None:
            return ResponseType.REJECT_OFFER

        # Track opponent offer for modeling
        if offer is not None:
            self._received_offers.append(offer)
            try:
                for i, val in enumerate(offer):
                    if i not in self._issue_value_counts:
                        self._issue_value_counts[i] = {}
                    self._issue_value_counts[i][val] = self._issue_value_counts[i].get(val, 0) + 1
            except Exception:
                pass

        t = self._relative_time(state)
        threshold = 0.82 * (1.0 - t ** 2.8)
        if self._u(offer) >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: Any) -> Any:
        self._init()
        t = self._relative_time(state)
        target = 0.95 - t ** 2.0 * 0.4

        try:
            outcomes = list(self.nmi.outcomes)
            candidates = [o for o in outcomes if self._u(o) >= target * 0.85]
            if candidates:
                return max(candidates, key=lambda o: self._u(o))
        except Exception:
            pass
        return self._best

    def estimate_opponent_utility(self, outcome: Any) -> float:
        """Estimate opponent utility via frequency-based preference model."""
        if not self._issue_value_counts or not outcome:
            return 0.5
        try:
            total = 0.0
            n_issues = len(outcome)
            for i, val in enumerate(outcome):
                counts = self._issue_value_counts.get(i, {})
                total_count = sum(counts.values()) or 1
                freq = counts.get(val, 0) / total_count
                total += freq + random.uniform(-0.15, 0.15)
            return max(0.0, min(1.0, total / max(n_issues, 1)))
        except Exception:
            return 0.5
