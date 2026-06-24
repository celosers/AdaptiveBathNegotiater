"""
Analysis script for concealment ablation experiments.

Reads traces_all.csv from experiment_runner.py, trains attacker models
on bid traces, and computes all evaluation metrics with statistical tests.

Usage:
    python analyze_results.py                          # full analysis
    python analyze_results.py --quick                  # skip attacker eval, just utility/agreement
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import wilcoxon
# Holm-Bonferroni implemented inline to avoid statsmodels dependency

from negmas.inout import Scenario

from leakage_attackers import (
    ClassicFrequencyAttacker,
    RethinkingFrequencyAttacker,
    BayesianAttacker,
    evaluate_kendall_tau,
    evaluate_issue_weight_mae,
    _outcome_to_values,
    _extract_issue_space,
)


# =========================================================================
# Configuration
# =========================================================================

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "experiment_results"
TRACES_FILE = RESULTS_DIR / "traces_all.csv"
OUTPUT_DIR = RESULTS_DIR / "analysis"
ANALYSIS_JSON = OUTPUT_DIR / "results.json"

CONFIGS = ["OFF", "J-only", "N-only", "B-only", "FULL", "Random", "OM"]

METRICS = [
    "self_utility",
    "agreement_rate",
    "kendall_tau_cf",
    "kendall_tau_rf",
    "kendall_tau_bay",
    "issue_weight_mae_cf",
    "issue_weight_mae_rf",
    "issue_weight_mae_bay",
]

EPS = 1e-9


# =========================================================================
# Data loading
# =========================================================================

@dataclass
class NegotiationResult:
    domain: str
    opponent: str
    config: str
    seed: int
    agreement_reached: bool
    final_self_utility: float
    our_bids: list[tuple[Any, ...]]
    relative_times: list[float]


def load_traces(csv_path: str = str(TRACES_FILE)) -> list[NegotiationResult]:
    """Load and group trace rows into per-negotiation result objects."""
    if not Path(csv_path).exists():
        print(f"[ERROR] Traces file not found: {csv_path}")
        return []

    # Group rows by (domain, opponent, config, seed)
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["domain"], row["opponent"], row["config"], int(row["seed"]))
            groups[key].append(row)

    results = []
    for (domain, opponent, config, seed), rows in groups.items():
        our_bids = []
        relative_times = []
        agreement = False
        final_u = 0.0

        for r in rows:
            if r["our_offer"] and r["our_offer"].strip():
                our_bids.append(r["our_offer"])
                relative_times.append(float(r.get("relative_time", 0)))
            if r.get("agreement_reached", "").lower() in ("true", "1", "yes"):
                agreement = True
            fu = float(r.get("final_self_utility", 0))
            if fu > 0:
                final_u = fu

        results.append(NegotiationResult(
            domain=domain,
            opponent=opponent,
            config=config,
            seed=seed,
            agreement_reached=agreement,
            final_self_utility=final_u,
            our_bids=our_bids,
            relative_times=relative_times,
        ))

    return results


# =========================================================================
# Scenario cache
# =========================================================================

def _parse_bid_string(bid_str: str) -> tuple[Any, ...] | None:
    """Parse a bid string like '(v4, v2, v4)' into a tuple."""
    s = bid_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip().strip("'\"") for p in s.split(",")]
    if parts:
        return tuple(parts)
    return None


def _infer_domain_from_bids(bids: list[str]) -> tuple[int, list[list[str]]]:
    """Infer issue space from bid strings."""
    parsed = []
    for b in bids:
        p = _parse_bid_string(b)
        if p:
            parsed.append(p)
    return _extract_issue_space(parsed)


# =========================================================================
# Attacker evaluation
# =========================================================================

def evaluate_attackers_on_result(
    result: NegotiationResult,
    scenario: Scenario,
    attacker_classes: list[tuple[str, type]],
) -> dict[str, float]:
    """Train all attackers on one negotiation trace and compute metrics."""
    metrics = {}

    # Parse bids from string to tuples
    parsed_bids = []
    for b_str in result.our_bids:
        p = _parse_bid_string(b_str)
        if p:
            parsed_bids.append(p)
    if not parsed_bids:
        return metrics

    ufun = scenario.ufuns[0]  # our agent's true utility function
    n_issues, domains = _infer_domain_from_bids(result.our_bids)

    # Get all outcomes from scenario for tau computation
    try:
        os_obj = scenario.outcome_space
        all_outcomes = list(os_obj.enumerate_or_sample(5000))
    except Exception:
        all_outcomes = []

    for name, cls in attacker_classes:
        attacker = cls()
        attacker.fit(parsed_bids)

        tau = 0.0
        mae = 0.0
        if all_outcomes and len(all_outcomes) >= 2:
            tau = evaluate_kendall_tau(attacker, ufun, all_outcomes)
            mae = evaluate_issue_weight_mae(attacker, ufun, n_issues, domains)

        metrics[f"kendall_tau_{name}"] = tau
        metrics[f"issue_weight_mae_{name}"] = mae

    return metrics


# =========================================================================
# Aggregation and statistics
# =========================================================================

def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper)."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    n = len(arr)
    if n < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    z = 1.96  # 95% CI
    return mean, mean - z * se, mean + z * se


def wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    if len(values_a) < 3 or len(values_b) < 3:
        return 0.0, 1.0
    # Align to same length
    n = min(len(values_a), len(values_b))
    try:
        stat, p = wilcoxon(values_a[:n], values_b[:n], zero_method="wilcox")
        return float(stat), float(p)
    except Exception:
        return 0.0, 1.0


def run_analysis(results: list[NegotiationResult], quick: bool = False) -> dict:
    """Main analysis: aggregate metrics per config, run statistical tests."""

    # Group by config
    config_results: dict[str, list[NegotiationResult]] = defaultdict(list)
    for r in results:
        config_results[r.config].append(r)

    print(f"Loaded {len(results)} negotiation results across {len(config_results)} configs")

    # Load scenarios once per domain
    scenarios: dict[str, Scenario] = {}
    domains = sorted({r.domain for r in results})
    for domain in domains:
        path = SCENARIOS_DIR / domain
        if path.is_dir():
            s = Scenario.load(path, ignore_discount=True)
            if s is not None:
                scenarios[domain] = s
                print(f"  Loaded scenario: {domain}")
            else:
                print(f"  [WARN] Failed to load scenario: {domain}")
        else:
            print(f"  [WARN] Scenario dir not found: {domain}")

    # Attacker classes
    attacker_classes = [
        ("cf", ClassicFrequencyAttacker),
        ("rf", RethinkingFrequencyAttacker),
        ("bay", BayesianAttacker),
    ]

    # Compute per-result metrics
    metric_by_config: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_by_domain_config: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_by_opp_config: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_evaluated = 0

    for r in results:
        # Basic metrics (always computed)
        metric_by_config[r.config]["self_utility"].append(r.final_self_utility)
        metric_by_config[r.config]["agreement_rate"].append(1.0 if r.agreement_reached else 0.0)
        metric_by_domain_config[(r.domain, r.config)]["self_utility"].append(r.final_self_utility)
        metric_by_domain_config[(r.domain, r.config)]["agreement_rate"].append(1.0 if r.agreement_reached else 0.0)
        metric_by_opp_config[(r.opponent, r.config)]["self_utility"].append(r.final_self_utility)
        metric_by_opp_config[(r.opponent, r.config)]["agreement_rate"].append(1.0 if r.agreement_reached else 0.0)

        # Attacker metrics (slow, skip in quick mode)
        if not quick and r.our_bids:
            scenario = scenarios.get(r.domain)
            if scenario is not None:
                att_metrics = evaluate_attackers_on_result(r, scenario, attacker_classes)
                for k, v in att_metrics.items():
                    metric_by_config[r.config][k].append(v)
                    metric_by_domain_config[(r.domain, r.config)][k].append(v)
                    metric_by_opp_config[(r.opponent, r.config)][k].append(v)
                n_evaluated += 1

        if (n_evaluated > 0 and n_evaluated % 100 == 0):
            print(f"  Attacker evaluation: {n_evaluated} done...")

    if not quick:
        print(f"  Attacker evaluation: {n_evaluated} total")

    # Aggregate: mean ± 95% CI per config per metric
    all_metrics = METRICS if not quick else ["self_utility", "agreement_rate"]
    aggregated = {}
    for config in CONFIGS:
        if config not in metric_by_config:
            continue
        aggregated[config] = {}
        for metric in all_metrics:
            vals = metric_by_config[config].get(metric, [])
            m, lo, hi = mean_ci(vals)
            aggregated[config][metric] = {
                "mean": round(m, 6),
                "ci_lower": round(lo, 6),
                "ci_upper": round(hi, 6),
                "n": len(vals),
            }

    # Statistical tests: FULL vs OFF, FULL vs Random for each metric
    stats_results = []
    for comparison, cfg_a, cfg_b in [
        ("FULL vs OFF", "FULL", "OFF"),
        ("FULL vs Random", "FULL", "Random"),
    ]:
        if cfg_a not in metric_by_config or cfg_b not in metric_by_config:
            continue
        for metric in all_metrics:
            vals_a = metric_by_config[cfg_a].get(metric, [])
            vals_b = metric_by_config[cfg_b].get(metric, [])
            if len(vals_a) >= 3 and len(vals_b) >= 3:
                stat, p = wilcoxon_test(vals_a, vals_b)
                stats_results.append({
                    "comparison": comparison,
                    "metric": metric,
                    "wilcoxon_stat": round(stat, 4),
                    "p_value": round(p, 6),
                })

    # Holm-Bonferroni correction (implemented inline)
    if stats_results:
        pvals = [s["p_value"] for s in stats_results]
        # Holm step-down: sort p-values, compare to alpha/(n-rank+1)
        n = len(pvals)
        sorted_indices = np.argsort(pvals)
        corrected = np.ones(n)
        for rank, idx in enumerate(sorted_indices):
            adjusted = min(1.0, pvals[idx] * (n - rank))
            corrected[idx] = adjusted
            # Ensure monotonicity
            if rank > 0:
                corrected[idx] = max(corrected[idx], corrected[sorted_indices[rank - 1]])
        for i, s in enumerate(stats_results):
            s["p_corrected"] = round(float(corrected[i]), 6)
            s["significant"] = bool(corrected[i] < 0.05)

    output = {
        "n_results": len(results),
        "configs_aggregated": aggregated,
        "statistical_tests": stats_results,
    }

    # Per-domain aggregation
    domains = sorted(set(r.domain for r in results))
    output["per_domain"] = {}
    for domain in domains:
        output["per_domain"][domain] = {}
        for config in CONFIGS:
            key = (domain, config)
            if key not in metric_by_domain_config:
                continue
            output["per_domain"][domain][config] = {}
            for metric in all_metrics:
                vals = metric_by_domain_config[key].get(metric, [])
                if not vals:
                    continue
                m, lo, hi = mean_ci(vals)
                output["per_domain"][domain][config][metric] = {
                    "mean": round(m, 6),
                    "ci_lower": round(lo, 6),
                    "ci_upper": round(hi, 6),
                    "n": len(vals),
                }

    # Per-opponent aggregation
    opponents = sorted(set(r.opponent for r in results))
    output["per_opponent"] = {}
    for opp in opponents:
        output["per_opponent"][opp] = {}
        for config in CONFIGS:
            key = (opp, config)
            if key not in metric_by_opp_config:
                continue
            output["per_opponent"][opp][config] = {}
            for metric in all_metrics:
                vals = metric_by_opp_config[key].get(metric, [])
                if not vals:
                    continue
                m, lo, hi = mean_ci(vals)
                output["per_opponent"][opp][config][metric] = {
                    "mean": round(m, 6),
                    "ci_lower": round(lo, 6),
                    "ci_upper": round(hi, 6),
                    "n": len(vals),
                }

    return output


# =========================================================================
# Output formatting
# =========================================================================

def print_table(aggregated: dict, metrics: list[str], configs: list[str]) -> None:
    """Print a formatted results table."""
    header = ["Metric"] + configs
    col_widths = [28] + [max(22, len(c)) for c in configs]

    def fmt_row(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))

    print(fmt_row(header))
    print("-" * sum(col_widths))

    for metric in metrics:
        row = [metric]
        for config in configs:
            info = aggregated.get(config, {}).get(metric, {})
            if info and info.get("n", 0) > 0:
                row.append(f"{info['mean']:.4f} ± {info['ci_upper'] - info['mean']:.4f}")
            else:
                row.append("—")
        print(fmt_row(row))


def print_stats(stats: list[dict]) -> None:
    """Print statistical test results."""
    if not stats:
        print("\nNo statistical tests available.")
        return
    print(f"\n{'Comparison':<20} {'Metric':<22} {'p':<10} {'p (corr)':<10} {'Sig':<6}")
    print("-" * 70)
    for s in stats:
        sig = "YES" if s["significant"] else "no"
        print(f"{s['comparison']:<20} {s['metric']:<22} {s['p_value']:<10.4f} {s['p_corrected']:<10.4f} {sig:<6}")


# =========================================================================
# Main
# =========================================================================

def main():
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("Concealment Ablation Analysis")
    print("=" * 60)

    if not TRACES_FILE.exists():
        print(f"\n[ERROR] Traces file not found: {TRACES_FILE}")
        print("Run experiment_runner.py first.")
        sys.exit(1)

    print(f"\nLoading traces from {TRACES_FILE}...")
    results = load_traces()
    if not results:
        print("[ERROR] No results loaded.")
        sys.exit(1)

    print(f"\nRunning analysis ({'quick mode' if quick else 'full mode'})...")
    output = run_analysis(results, quick=quick)

    # Print tables
    configs = [c for c in CONFIGS if c in output["configs_aggregated"]]
    if not configs:
        print("[ERROR] No configs found in data.")
        sys.exit(1)

    core_metrics = ["self_utility", "agreement_rate"]
    if not quick:
        core_metrics += ["kendall_tau_cf", "kendall_tau_rf", "kendall_tau_bay",
                         "issue_weight_mae_cf", "issue_weight_mae_rf", "issue_weight_mae_bay"]

    print(f"\n--- Per-Config Results ---\n")
    print_table(output["configs_aggregated"], core_metrics, configs)

    print(f"\n--- Statistical Tests ---")
    print_stats(output["statistical_tests"])

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {ANALYSIS_JSON}")


if __name__ == "__main__":
    main()
