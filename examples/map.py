"""MAP-style negotiator with frequency-based opponent modeling and time-based concession."""

from __future__ import annotations
from typing import Any
from collections import Counter

from negmas.sao import SAONegotiator
from negmas import ResponseType


def _to_values(outcome: Any) -> tuple[Any, ...]:
    if outcome is None:
        return tuple()
    if isinstance(outcome, dict):
        return tuple(outcome[k] for k in sorted(outcome))
    if isinstance(outcome, (tuple, list)):
        return tuple(outcome)
    return (outcome,)


class MAPNeg(SAONegotiator):
    """Frequency-based opponent modeler with time-based proposing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._best: Any = None
        self._outcomes: list[Any] = []
        self._initialized = False
        self._received: list[tuple[Any, ...]] = []
        self._value_counts: list[Counter[Any]] = []
        self._issue_names: list[str] = []
        self._weights: list[float] = []

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
            self._outcomes = list(self.nmi.outcomes)
            if self._outcomes:
                self._best = max(self._outcomes, key=lambda o: float(self.ufun(o)))
                sample = self._outcomes[0]
                vals = _to_values(sample)
                self._issue_names = [f"i{i}" for i in range(len(vals))]
        except Exception:
            self._best = None
        self._initialized = True

    @property
    def opponent_ufun(self):
        return self._opponent_adapter

    def _opponent_adapter(self, outcome):
        return self._estimate_opponent(outcome)

    def _estimate_opponent(self, outcome: Any) -> float:
        vals = _to_values(outcome)
        if not vals or not self._value_counts:
            return 0.5
        total = 0.0
        for i, v in enumerate(vals):
            w = self._weights[i] if i < len(self._weights) else 1.0 / max(1, len(vals))
            counts = self._value_counts[i] if i < len(self._value_counts) else Counter()
            if counts:
                max_c = max(counts.values())
                score = (counts.get(v, 0) + 1.0) / (max_c + 1.0)
            else:
                score = 0.5
            total += w * score
        return max(0.0, min(1.0, total))

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

        # Update opponent model
        vals = _to_values(offer)
        if vals:
            self._received.append(vals)
            while len(self._value_counts) < len(vals):
                self._value_counts.append(Counter())
            for i, v in enumerate(vals):
                if i < len(self._value_counts):
                    self._value_counts[i][v] += 1
            self._recompute_weights()

        t = self._relative_time(state)
        threshold = 0.85 * (1.0 - t ** 2.5)
        if self._u(offer) >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _recompute_weights(self) -> None:
        n = len(self._value_counts)
        if n == 0:
            self._weights = []
            return
        raw = []
        for counts in self._value_counts:
            total = sum(counts.values())
            if total > 0:
                dominance = max(counts.values()) / total
                raw.append(1.0 + 0.5 * dominance)
            else:
                raw.append(1.0)
        s = sum(raw)
        self._weights = [r / s for r in raw] if s > 0 else [1.0 / n] * n

    def propose(self, state: Any) -> Any:
        self._init()
        t = self._relative_time(state)
        target = 1.0 - t ** 2.0

        candidates = [o for o in self._outcomes if self._u(o) >= target * 0.85]
        if not candidates:
            candidates = self._outcomes

        # Score by self-utility + opponent model estimate
        best_score = -float("inf")
        best_o = None
        for o in candidates:
            su = self._u(o)
            ou = self._estimate_opponent(o)
            score = 0.7 * su + 0.3 * ou
            if score > best_score:
                best_score = score
                best_o = o
        return best_o or self._best
