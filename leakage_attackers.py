"""
Leakage attackers for preference concealment evaluation.

Each attacker is an offline model: given a sequence of bids from agent A,
it estimates A's utility function. The estimated model is then compared
against A's true utility function via Kendall tau and issue-weight MAE.

Attackers:
  1. ClassicFrequencyAttacker  — per-issue value frequency counting
  2. RethinkingFrequencyAttacker — DFM-based, chi-square concession detection
  3. BayesianAttacker — Dirichlet-multinomial belief over issue weights + values
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Sequence

import numpy as np
from scipy.stats import kendalltau, chi2

EPS = 1e-9


# =========================================================================
# Helpers
# =========================================================================

def _outcome_to_values(outcome: Any) -> tuple[Any, ...]:
    if outcome is None:
        return tuple()
    if isinstance(outcome, dict):
        return tuple(outcome[k] for k in sorted(outcome))
    if isinstance(outcome, (tuple, list)):
        return tuple(outcome)
    return (outcome,)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize_positive(values: list[float]) -> list[float]:
    if not values:
        return []
    values = [max(0.0, v) for v in values]
    s = sum(values)
    if s <= EPS:
        f = 1.0 / len(values)
        return [f for _ in values]
    return [v / s for v in values]


def _critical_chi_square_95(df: int) -> float:
    table = {
        1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
        6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
    }
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    return chi2.ppf(0.95, df)


def _extract_issue_space(bids: list[tuple[Any, ...]]) -> tuple[int, list[list[Any]]]:
    """Infer number of issues and possible values from bid sequences."""
    n_issues = 0
    for bid in bids:
        if len(bid) > n_issues:
            n_issues = len(bid)
    values: list[set[Any]] = [set() for _ in range(n_issues)]
    for bid in bids:
        for i, v in enumerate(bid):
            if i < n_issues:
                values[i].add(v)
    return n_issues, [sorted(vs) for vs in values]


# =========================================================================
# 1. Classic Frequency Attacker
# =========================================================================

class ClassicFrequencyAttacker:
    """Estimates preferences by counting per-issue value frequencies.

    The simplest and most common opponent modeling approach: values that
    appear more often in the bid sequence are assumed to be more preferred.
    """

    def __init__(self, stability_bonus: float = 0.05):
        self.stability_bonus = stability_bonus
        self._n_issues = 0
        self._value_counts: list[Counter[Any]] = []
        self._issue_weights: list[float] = []
        self._value_domains: list[list[Any]] = []
        self._fitted = False

    def fit(self, bids: list[tuple[Any, ...]]) -> "ClassicFrequencyAttacker":
        if not bids:
            return self
        self._n_issues, self._value_domains = _extract_issue_space(bids)
        self._value_counts = [Counter() for _ in range(self._n_issues)]

        # Count value frequencies
        for bid in bids:
            for i, v in enumerate(bid):
                if i < self._n_issues:
                    self._value_counts[i][v] += 1

        # Stability scoring: values that appear consecutively get a bonus
        stability_scores = [0.0] * self._n_issues
        for i in range(self._n_issues):
            for idx in range(1, len(bids)):
                prev = bids[idx - 1]
                curr = bids[idx]
                if i < len(prev) and i < len(curr) and prev[i] == curr[i]:
                    stability_scores[i] += self.stability_bonus

        # Issue weights: higher frequency dominance + stability = more important
        raw_weights = []
        for i, counts in enumerate(self._value_counts):
            total = sum(counts.values())
            dominance = (max(counts.values()) / total) if total > 0 else 0.0
            raw_weights.append(1.0 + stability_scores[i] + 0.35 * dominance)
        self._issue_weights = _normalize_positive(raw_weights)
        self._fitted = True
        return self

    def predict_utility(self, outcome: Any) -> float:
        values = _outcome_to_values(outcome)
        if not self._fitted or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self._issue_weights[i] if i < len(self._issue_weights) else 0.0
            counts = self._value_counts[i] if i < len(self._value_counts) else Counter()
            max_count = max(counts.values()) if counts else 1
            score = (counts.get(v, 0) + 1.0) / (max_count + 1.0)
            total += w * score
        return _clip(total, 0.0, 1.0)

    def predict_issue_weights(self) -> list[float]:
        return list(self._issue_weights)

    def __call__(self, outcome: Any) -> float:
        return self.predict_utility(outcome)


# =========================================================================
# 2. Rethinking Frequency Attacker (DFM-based)
# =========================================================================

class RethinkingFrequencyAttacker:
    """DFM-inspired attacker based on Tunali et al. (2017).

    Uses sliding windows with chi-square tests to detect which issues the
    agent is conceding on vs. holding firm.  Issues that remain unchanged
    across concession windows are inferred to be truly important.
    """

    def __init__(
        self,
        window_size: int = 4,
        alpha: float = 0.18,
        beta: float = 1.6,
        gamma: float = 0.70,
    ):
        self.window_size = max(2, int(window_size))
        self.alpha = max(0.0, alpha)
        self.beta = max(0.1, beta)
        self.gamma = _clip(gamma, 0.05, 1.0)
        self._n_issues = 0
        self._value_domains: list[list[Any]] = []
        self._value_counts: list[Counter[Any]] = []
        self._issue_weights: list[float] = []
        self._fitted = False

    def fit(self, bids: list[tuple[Any, ...]], relative_times: list[float] | None = None) -> "RethinkingFrequencyAttacker":
        if not bids:
            return self
        self._n_issues, self._value_domains = _extract_issue_space(bids)
        self._value_counts = [Counter() for _ in range(self._n_issues)]
        self._issue_weights = [1.0 / self._n_issues] * self._n_issues

        # Count all values
        for bid in bids:
            for i, v in enumerate(bid):
                if i < self._n_issues:
                    self._value_counts[i][v] += 1

        # DFM sliding window analysis
        k = self.window_size
        for window_start in range(0, max(1, len(bids) - 2 * k), k):
            if window_start + 2 * k > len(bids):
                break
            prev_window = bids[window_start:window_start + k]
            curr_window = bids[window_start + k:window_start + 2 * k]

            unchanged_issues: list[int] = []
            concession_detected = False

            for i in range(self._n_issues):
                prev_dist = self._smoothed_distribution(prev_window, i)
                curr_dist = self._smoothed_distribution(curr_window, i)
                if not prev_dist or not curr_dist:
                    continue
                df = max(1, len(set(prev_dist) | set(curr_dist)) - 1)
                stat = self._chi_square_stat(prev_dist, curr_dist, k)
                if stat <= _critical_chi_square_95(df):
                    unchanged_issues.append(i)
                else:
                    prev_eu = sum(self._value_score(i, v) * p for v, p in prev_dist.items())
                    curr_eu = sum(self._value_score(i, v) * p for v, p in curr_dist.items())
                    if curr_eu + 1e-5 < prev_eu:
                        concession_detected = True

            if concession_detected and 0 < len(unchanged_issues) < self._n_issues:
                rt = relative_times[window_start + 2 * k] if relative_times else (window_start + 2 * k) / len(bids)
                delta = self.alpha * max(0.0, 1.0 - rt ** self.beta)
                raw = list(self._issue_weights)
                for i in unchanged_issues:
                    raw[i] += delta
                self._issue_weights = _normalize_positive(raw)

        self._fitted = True
        return self

    def _smoothed_distribution(self, window: list[tuple[Any, ...]], issue_idx: int) -> dict[Any, float]:
        domain = self._value_domains[issue_idx] if issue_idx < len(self._value_domains) else []
        if not domain:
            return {}
        counts: Counter[Any] = Counter()
        for bid in window:
            if issue_idx < len(bid):
                counts[bid[issue_idx]] += 1
        denom = len(window) + len(domain)
        return {v: (1.0 + counts.get(v, 0)) / max(EPS, denom) for v in domain}

    def _chi_square_stat(self, prev: dict[Any, float], curr: dict[Any, float], k: int) -> float:
        stat = 0.0
        for v in set(prev) | set(curr):
            expected = max(EPS, prev.get(v, 0.0) * k)
            observed = curr.get(v, 0.0) * k
            stat += (observed - expected) ** 2 / expected
        return stat

    def _value_score(self, issue_idx: int, value: Any) -> float:
        counts = self._value_counts[issue_idx] if issue_idx < len(self._value_counts) else Counter()
        domain = self._value_domains[issue_idx] if issue_idx < len(self._value_domains) else []
        if not domain:
            return 0.5
        max_count = max((counts.get(v, 0) for v in domain), default=0)
        numerator = (1.0 + counts.get(value, 0)) ** self.gamma
        denominator = (1.0 + max_count) ** self.gamma
        return _clip(numerator / max(EPS, denominator), 0.0, 1.0)

    def predict_utility(self, outcome: Any) -> float:
        values = _outcome_to_values(outcome)
        if not self._fitted or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self._issue_weights[i] if i < len(self._issue_weights) else 0.0
            total += w * self._value_score(i, v)
        return _clip(total, 0.0, 1.0)

    def predict_issue_weights(self) -> list[float]:
        return list(self._issue_weights)

    def __call__(self, outcome: Any) -> float:
        return self.predict_utility(outcome)


# =========================================================================
# 3. Bayesian Attacker
# =========================================================================

class BayesianAttacker:
    """Bayesian preference learner with Dirichlet-multinomial conjugate model.

    Maintains a Dirichlet prior over issue weights and per-issue categorical
    distributions over values.  Each bid is treated as a draw from the agent's
    latent utility-maximizing distribution, updating the posterior via Bayes.

    After observing a sequence of bids, the posterior mean gives the best
    estimate of the agent's utility function.
    """

    def __init__(self, prior_strength: float = 2.0):
        self.prior_strength = max(0.5, prior_strength)
        self._n_issues = 0
        self._value_domains: list[list[Any]] = []
        # Dirichlet posterior parameters for issue weights
        self._weight_alpha: list[float] = []
        # Per-issue Dirichlet posterior for value preferences
        self._value_alpha: list[dict[Any, float]] = []
        self._fitted = False

    def fit(self, bids: list[tuple[Any, ...]]) -> "BayesianAttacker":
        if not bids:
            return self
        self._n_issues, self._value_domains = _extract_issue_space(bids)

        # Initialize Dirichlet priors
        self._weight_alpha = [self.prior_strength] * self._n_issues
        self._value_alpha = []
        for i in range(self._n_issues):
            domain = self._value_domains[i]
            self._value_alpha.append({v: self.prior_strength for v in domain})

        # Bayesian update: each bid is treated as evidence
        for bid in bids:
            # Per-issue value updates: observed value gets +1 pseudo-count
            for i, v in enumerate(bid):
                if i >= self._n_issues:
                    continue
                if v in self._value_alpha[i]:
                    self._value_alpha[i][v] += 1.0
                else:
                    self._value_alpha[i][v] = self.prior_strength + 1.0

            # Issue weight update: allocate weight based on how "extreme"
            # the bid's values are relative to the prior
            value_scores = []
            for i, v in enumerate(bid):
                if i >= self._n_issues:
                    continue
                domain = self._value_domains[i]
                alpha = self._value_alpha[i]
                prior_mass = sum(alpha.get(dv, self.prior_strength) for dv in domain)
                p = alpha.get(v, self.prior_strength) / max(EPS, prior_mass)
                value_scores.append(p)

            if value_scores:
                # Issues where the agent picks high-probability values gain weight
                mean_score = sum(value_scores) / len(value_scores)
                for i, s in enumerate(value_scores):
                    if i < self._n_issues and s > mean_score:
                        self._weight_alpha[i] += 0.3

        # Compute posterior means
        weight_sum = sum(self._weight_alpha)
        self._issue_weights = [a / weight_sum for a in self._weight_alpha]
        self._fitted = True
        return self

    def predict_utility(self, outcome: Any) -> float:
        values = _outcome_to_values(outcome)
        if not self._fitted or not values:
            return 0.5
        total = 0.0
        for i, v in enumerate(values):
            w = self._issue_weights[i] if i < len(self._issue_weights) else 0.0
            if i < len(self._value_alpha):
                alpha = self._value_alpha[i]
                total_mass = sum(alpha.values())
                val_mass = alpha.get(v, self.prior_strength)
                score = val_mass / max(EPS, total_mass)
            else:
                score = 0.5
            total += w * score
        return _clip(total, 0.0, 1.0)

    def predict_issue_weights(self) -> list[float]:
        return list(self._issue_weights)

    def __call__(self, outcome: Any) -> float:
        return self.predict_utility(outcome)


# =========================================================================
# Evaluation metrics
# =========================================================================

def evaluate_kendall_tau(
    attacker,
    true_ufun,
    outcomes: Sequence[Any],
) -> float:
    """Kendall tau between attacker's estimated ranking and true ranking.

    Returns tau in [-1, 1]. Higher = more accurate attacker = more leakage.
    """
    if len(outcomes) < 2:
        return 0.0
    estimated = np.array([attacker(o) for o in outcomes], dtype=float)
    true = np.array([float(true_ufun(o)) for o in outcomes], dtype=float)
    tau, _ = kendalltau(estimated, true)
    return float(tau if not math.isnan(tau) else 0.0)


def evaluate_issue_weight_mae(
    attacker,
    true_ufun,
    n_issues: int,
    issue_domains: list[list[Any]] | None = None,
) -> float:
    """Mean absolute error between estimated and true issue weights.

    True weights are estimated by fitting a linear model to the utility
    function over the outcome space.
    """
    est_weights = attacker.predict_issue_weights()
    if not est_weights:
        return 1.0

    true_weights = _estimate_true_issue_weights(true_ufun, n_issues, issue_domains)
    errors = [abs(est_weights[i] - true_weights[i]) for i in range(n_issues)]
    return float(np.mean(errors))


def _estimate_true_issue_weights(
    ufun,
    n_issues: int,
    issue_domains: list[list[Any]] | None = None,
) -> list[float]:
    """Estimate true issue weights by measuring utility variance per issue."""
    if issue_domains is None:
        return [1.0 / n_issues] * n_issues

    # For each issue, measure how much utility changes when that issue varies
    # while holding others fixed at a reference value
    importances = []
    for i in range(n_issues):
        domain = issue_domains[i] if i < len(issue_domains) else []
        if len(domain) < 2:
            importances.append(0.0)
            continue

        # Build reference outcome using first value of each issue
        ref_values = [dom[0] for dom in issue_domains[:n_issues]]
        utilities = []
        for v in domain:
            test_values = list(ref_values)
            test_values[i] = v
            # Try both dict and tuple representations
            try:
                u = float(ufun(tuple(test_values)))
            except Exception:
                try:
                    u = float(ufun(dict(enumerate(test_values))))
                except Exception:
                    u = 0.0
            utilities.append(u)
        importances.append(max(utilities) - min(utilities))

    return _normalize_positive(importances)


def evaluate_all(
    attacker,
    true_ufun,
    outcomes: Sequence[Any],
    n_issues: int,
    issue_domains: list[list[Any]] | None = None,
) -> dict[str, float]:
    """Convenience: run both metrics and return as dict."""
    return {
        "kendall_tau": evaluate_kendall_tau(attacker, true_ufun, outcomes),
        "issue_weight_mae": evaluate_issue_weight_mae(attacker, true_ufun, n_issues, issue_domains),
    }
