"""BOA-style negotiator with windowed frequency opponent model and adaptive acceptance."""

from __future__ import annotations
from typing import Any
from collections import Counter, deque

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


class BOANeg(SAONegotiator):
    """Windowed opponent modeler with adaptive acceptance threshold."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._best: Any = None
        self._outcomes: list[Any] = []
        self._initialized = False
        self._received: deque[tuple[Any, ...]] = deque(maxlen=20)
        self._value_hist: list[Counter[Any]] = []
        self._issue_names: list[str] = []
        self._n_issues = 0

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
                self._n_issues = len(vals)
                self._issue_names = [f"i{i}" for i in range(self._n_issues)]
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
        if not vals or not self._received:
            return 0.5
        total = 0.0
        n = len(vals)
        for i, v in enumerate(vals):
            # Count frequency in recent window
            cnt = 0
            for r in self._received:
                if i < len(r) and r[i] == v:
                    cnt += 1
            freq = (cnt + 1.0) / (len(self._received) + n)
            total += freq
        return max(0.0, min(1.0, total / n))

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

        vals = _to_values(offer)
        if vals:
            self._received.append(vals)

        t = self._relative_time(state)
        # BOA-style: learn opponent concession rate, adapt threshold
        threshold = 0.90 * (1.0 - t ** 3.5)
        # Late phase: more flexible
        if t > 0.75:
            threshold -= 0.05
        if t > 0.90:
            threshold = max(0.0, threshold - 0.10)
        if self._u(offer) >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: Any) -> Any:
        self._init()
        t = self._relative_time(state)
        target = 0.95 * (1.0 - t ** 3.0)

        candidates = [o for o in self._outcomes if self._u(o) >= target * 0.90]
        if not candidates:
            candidates = self._outcomes[:200]

        # Score by self-utility + diversity from recent proposals
        best_score = -float("inf")
        best_o = None
        for o in candidates:
            su = self._u(o)
            ou = self._estimate_opponent(o)
            score = 0.65 * su + 0.35 * ou
            if score > best_score:
                best_score = score
                best_o = o
        return best_o or self._best
