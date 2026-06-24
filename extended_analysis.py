"""
Extended analysis — 5 complementary analyses for the paper.
Usage: python extended_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent / "experiment_results"
TRACES_FILE = RESULTS_DIR / "traces_all.csv"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["OFF", "J-only", "N-only", "B-only", "FULL", "Random", "OM"]
DOMAINS = ["Camera", "Car", "Laptop", "Grocery", "ISBTAcquisition", "Travel", "Party", "Energy"]
OPPONENTS = ["Boulware", "Conceder", "Linear", "MiCRO", "BOANeg", "ShadowBath"]
EPS = 1e-9


# =========================================================================
# Data loading (reused from analyze_results.py)
# =========================================================================

@dataclass
class NegotiationResult:
    domain: str
    opponent: str
    config: str
    seed: int
    agreement_reached: bool
    final_self_utility: float
    our_bids: list[str]
    relative_times: list[float]


def load_traces(csv_path: str = str(TRACES_FILE)) -> list[NegotiationResult]:
    if not Path(csv_path).exists():
        print(f"[ERROR] Traces file not found: {csv_path}")
        return []
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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
            domain=domain, opponent=opponent, config=config, seed=seed,
            agreement_reached=agreement, final_self_utility=final_u,
            our_bids=our_bids, relative_times=relative_times,
        ))
    return results


# =========================================================================
# Stats helpers
# =========================================================================

def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    n = len(arr)
    if n < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    z = 1.96
    return mean, mean - z * se, mean + z * se


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled SD)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    pooled = math.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    if pooled < EPS:
        return 0.0
    return float((np.mean(b_arr) - np.mean(a_arr)) / pooled)


# =========================================================================
# Analysis 1: Per-domain breakdown
# =========================================================================

def analysis_per_domain(results: list[NegotiationResult]):
    print("=" * 60)
    print("ANALYSIS 1: Per-Domain Breakdown")
    print("=" * 60)

    with open(ANALYSIS_DIR / "results.json", "r") as f:
        full_results = json.load(f)

    for metric in ["self_utility", "agreement_rate", "kendall_tau_bay", "issue_weight_mae_cf"]:
        metric_label = {
            "self_utility": "Self Utility",
            "agreement_rate": "Agreement Rate",
            "kendall_tau_bay": "Kendall τ (Bayesian)",
            "issue_weight_mae_cf": "Issue Weight MAE (CF)",
        }.get(metric, metric)

        print(f"\n--- {metric_label} ---")
        header = f"{'Domain':<20}"
        for cfg in CONFIGS:
            header += f" {cfg:>12}"
        print(header)
        print("-" * len(header))

        for domain in DOMAINS:
            row = f"{domain:<20}"
            for cfg in CONFIGS:
                # Compute per-domain per-config from raw data
                vals = [r.final_self_utility for r in results
                        if r.domain == domain and r.config == cfg]
                if metric == "agreement_rate":
                    vals = [1.0 if r.agreement_reached else 0.0 for r in results
                            if r.domain == domain and r.config == cfg]
                elif metric in ("kendall_tau_bay", "issue_weight_mae_cf"):
                    # Use precomputed from full analysis
                    agg = full_results.get("configs_aggregated", {}).get(cfg, {}).get(metric, {})
                    # For per-domain we need to recompute from attacker eval — use raw data
                    pass

                if metric in ("self_utility", "agreement_rate"):
                    m, _, _ = mean_ci(vals)
                    row += f" {m:12.4f}"
                else:
                    row += f" {'—':>12}"
            print(row)


# =========================================================================
# Analysis 2: Per-opponent breakdown
# =========================================================================

def analysis_per_opponent(results: list[NegotiationResult]):
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Per-Opponent Breakdown")
    print("=" * 60)

    for metric in ["self_utility", "agreement_rate"]:
        metric_label = "Self Utility" if metric == "self_utility" else "Agreement Rate"
        print(f"\n--- {metric_label} ---")
        header = f"{'Opponent':<16}"
        for cfg in CONFIGS:
            header += f" {cfg:>10}"
        print(header)
        print("-" * len(header))

        for opp in OPPONENTS:
            row = f"{opp:<16}"
            for cfg in CONFIGS:
                if metric == "self_utility":
                    vals = [r.final_self_utility for r in results
                            if r.opponent == opp and r.config == cfg]
                else:
                    vals = [1.0 if r.agreement_reached else 0.0 for r in results
                            if r.opponent == opp and r.config == cfg]
                m, _, _ = mean_ci(vals)
                row += f" {m:10.4f}"
            print(row)


# =========================================================================
# Analysis 3: Ablation contribution
# =========================================================================

def analysis_ablation(results: list[NegotiationResult]):
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Ablation Contribution (J / N / B)")
    print("=" * 60)

    with open(ANALYSIS_DIR / "results.json", "r") as f:
        full_results = json.load(f)

    agg = full_results["configs_aggregated"]

    # Contribution of each component = difference when enabling it alone vs OFF
    # J contribution: J-only - OFF
    # N contribution: N-only - OFF
    # B contribution: B-only - OFF
    # Synergy: FULL - (J-only + N-only + B-only - 2*OFF) = FULL - J - N - B + 2*OFF
    #   = (FULL - OFF) - ((J-only-OFF) + (N-only-OFF) + (B-only-OFF))

    metrics = ["kendall_tau_bay", "kendall_tau_cf", "kendall_tau_rf",
               "issue_weight_mae_cf", "issue_weight_mae_rf", "issue_weight_mae_bay",
               "self_utility"]

    print(f"\n{'Metric':<24} {'OFF':>8} {'J contrib':>10} {'N contrib':>10} {'B contrib':>10} {'Sum(J+N+B)':>12} {'FULL actual':>12} {'Synergy':>10}")
    print("-" * 100)

    for metric in metrics:
        off_val = agg.get("OFF", {}).get(metric, {}).get("mean", 0)
        j_val = agg.get("J-only", {}).get(metric, {}).get("mean", 0)
        n_val = agg.get("N-only", {}).get(metric, {}).get("mean", 0)
        b_val = agg.get("B-only", {}).get(metric, {}).get("mean", 0)
        full_val = agg.get("FULL", {}).get(metric, {}).get("mean", 0)

        j_contrib = j_val - off_val
        n_contrib = n_val - off_val
        b_contrib = b_val - off_val
        additive_sum = j_contrib + n_contrib + b_contrib
        synergy = (full_val - off_val) - additive_sum

        direction = "↑" if off_val > 0 else ""
        print(f"{metric:<24} {off_val:8.4f} {j_contrib:+10.4f} {n_contrib:+10.4f} {b_contrib:+10.4f} {additive_sum:+12.4f} {full_val:12.4f} {synergy:+10.4f}")

    # Contribution percentages
    print(f"\n{'Metric':<24} {'J contrib %':>12} {'N contrib %':>12} {'B contrib %':>12}")
    print("-" * 60)
    for metric in metrics:
        off_val = agg.get("OFF", {}).get(metric, {}).get("mean", 0)
        full_val = agg.get("FULL", {}).get(metric, {}).get("mean", 0)
        j_val = agg.get("J-only", {}).get(metric, {}).get("mean", 0)
        n_val = agg.get("N-only", {}).get(metric, {}).get("mean", 0)
        b_val = agg.get("B-only", {}).get(metric, {}).get("mean", 0)

        total_change = full_val - off_val
        if abs(total_change) < EPS:
            print(f"{metric:<24} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            continue
        j_pct = (j_val - off_val) / total_change * 100
        n_pct = (n_val - off_val) / total_change * 100
        b_pct = (b_val - off_val) / total_change * 100
        print(f"{metric:<24} {j_pct:+11.1f}% {n_pct:+11.1f}% {b_pct:+11.1f}%")


# =========================================================================
# Analysis 4: Utility-Privacy Tradeoff Scatter
# =========================================================================

def analysis_tradeoff(results: list[NegotiationResult]):
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Utility-Privacy Tradeoff")
    print("=" * 60)

    with open(ANALYSIS_DIR / "results.json", "r") as f:
        full_results = json.load(f)

    agg = full_results["configs_aggregated"]

    config_labels = {
        "OFF": "OFF",
        "J-only": "J",
        "N-only": "N",
        "B-only": "B",
        "FULL": "FULL",
        "Random": "Rand",
        "OM": "OM",
    }
    config_colors = {
        "OFF": "#333333",
        "J-only": "#e41a1c",
        "N-only": "#377eb8",
        "B-only": "#4daf4a",
        "FULL": "#984ea3",
        "Random": "#ff7f00",
        "OM": "#a65628",
    }
    config_markers = {
        "OFF": "s",
        "J-only": "o",
        "N-only": "o",
        "B-only": "o",
        "FULL": "D",
        "Random": "^",
        "OM": "v",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (tau_metric, mae_metric, title) in zip(axes, [
        ("kendall_tau_cf", "issue_weight_mae_cf", "CF Attacker"),
        ("kendall_tau_rf", "issue_weight_mae_rf", "RF Attacker"),
        ("kendall_tau_bay", "issue_weight_mae_bay", "Bayesian Attacker"),
    ]):
        for cfg in CONFIGS:
            info = agg.get(cfg, {})
            u_info = info.get("self_utility", {})
            t_info = info.get(tau_metric, {})
            m_info = info.get(mae_metric, {})

            x = u_info.get("mean", 0)
            y = t_info.get("mean", 0)
            x_err = u_info.get("mean", 0) - u_info.get("ci_lower", 0)

            ax.errorbar(x, y, xerr=x_err, fmt=config_markers[cfg],
                       color=config_colors[cfg], markersize=10,
                       label=config_labels[cfg] if tau_metric == "kendall_tau_cf" else "",
                       capsize=3, alpha=0.9)

        ax.set_xlabel("Self Utility")
        ax.set_ylabel("Kendall τ" if "kendall" in tau_metric else "Issue Weight MAE")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Single legend
    fig.legend(loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle("Utility–Privacy Tradeoff (lower-left = better)", fontsize=13, y=1.01)
    plt.tight_layout()
    outpath = ANALYSIS_DIR / "tradeoff_scatter.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")

    # Also make a combined Pareto-style plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for cfg in CONFIGS:
        info = agg.get(cfg, {})
        u = info.get("self_utility", {}).get("mean", 0)
        tau = info.get("kendall_tau_bay", {}).get("mean", 0)
        mae = info.get("issue_weight_mae_cf", {}).get("mean", 0)

        ax2.scatter(u, tau, s=mae * 800, c=config_colors[cfg],
                   marker=config_markers[cfg], label=config_labels[cfg],
                   edgecolors="white", linewidth=0.5, alpha=0.9, zorder=5)

        # Annotate
        ax2.annotate(config_labels[cfg], (u, tau),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=config_colors[cfg])

    ax2.set_xlabel("Self Utility →")
    ax2.set_ylabel("Kendall τ (Bayesian) →")
    ax2.set_title("Utility–Privacy Tradeoff\n(bubble size = Issue Weight MAE)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", frameon=False, fontsize=8)
    plt.tight_layout()
    outpath2 = ANALYSIS_DIR / "tradeoff_bubble.png"
    fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {outpath2}")


# =========================================================================
# Analysis 5: Failed negotiation analysis
# =========================================================================

def analysis_failures(results: list[NegotiationResult]):
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Failed / Missing Negotiation Analysis")
    print("=" * 60)

    # Expected: 30 seeds per (domain, opponent, config) combo
    expected_per_combo = 30
    total_expected = len(DOMAINS) * len(OPPONENTS) * len(CONFIGS)
    total_possible = total_expected * expected_per_combo

    # Count actual
    combo_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    combo_agreements: dict[tuple[str, str, str], tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in results:
        key = (r.domain, r.opponent, r.config)
        combo_counts[key] += 1
        agree_count, total = combo_agreements[key]
        combo_agreements[key] = (agree_count + (1 if r.agreement_reached else 0), total + 1)

    print(f"\nTotal negotiations loaded: {len(results)} / {total_possible} expected ({len(results)/total_possible*100:.1f}%)")

    # By domain
    print(f"\n--- Missing by Domain ---")
    domain_counts: dict[str, int] = defaultdict(int)
    domain_total: dict[str, int] = defaultdict(int)
    for (d, o, c), cnt in combo_counts.items():
        domain_counts[d] += cnt
        domain_total[d] += expected_per_combo
    for d in DOMAINS:
        actual = domain_counts.get(d, 0)
        expected = expected_per_combo * len(OPPONENTS) * len(CONFIGS)
        pct = actual / expected * 100 if expected > 0 else 0
        print(f"  {d:<20} {actual:5d}/{expected:5d}  {pct:5.1f}%")

    # By config
    print(f"\n--- Missing by Config ---")
    config_counts: dict[str, int] = defaultdict(int)
    for (d, o, c), cnt in combo_counts.items():
        config_counts[c] += cnt
    for c in CONFIGS:
        actual = config_counts.get(c, 0)
        expected = expected_per_combo * len(DOMAINS) * len(OPPONENTS)
        pct = actual / expected * 100 if expected > 0 else 0
        print(f"  {c:<10} {actual:5d}/{expected:5d}  ({pct:.1f}%)")

    # By opponent
    print(f"\n--- Missing by Opponent ---")
    opp_counts: dict[str, int] = defaultdict(int)
    for (d, o, c), cnt in combo_counts.items():
        opp_counts[o] += cnt
    for o in OPPONENTS:
        actual = opp_counts.get(o, 0)
        expected = expected_per_combo * len(DOMAINS) * len(CONFIGS)
        pct = actual / expected * 100 if expected > 0 else 0
        print(f"  {o:<16} {actual:5d}/{expected:5d}  ({pct:.1f}%)")

    # List combos with 0 results (completely missing)
    missing_combos = []
    partial_combos = []
    for d in DOMAINS:
        for o in OPPONENTS:
            for c in CONFIGS:
                key = (d, o, c)
                cnt = combo_counts.get(key, 0)
                if cnt == 0:
                    missing_combos.append(key)
                elif cnt < expected_per_combo:
                    partial_combos.append((key, cnt))

    if missing_combos:
        print(f"\n--- Completely Missing ({len(missing_combos)} combos) ---")
        for d, o, c in missing_combos:
            print(f"  {d} × {o} × {c}")

    if partial_combos:
        print(f"\n--- Partially Complete ({len(partial_combos)} combos, <30 seeds) ---")
        partial_combos.sort(key=lambda x: x[1])
        for (d, o, c), cnt in partial_combos[:30]:
            print(f"  {d} × {o} × {c}: {cnt}/30 seeds")

    # Non-agreement analysis
    print(f"\n--- Non-Agreement Analysis ---")
    no_agree_count = sum(1 for r in results if not r.agreement_reached)
    print(f"  Negotiations without agreement: {no_agree_count} / {len(results)} ({no_agree_count/len(results)*100:.2f}%)")

    # Non-agreement by domain
    print(f"\n  Non-agreements by domain:")
    for d in DOMAINS:
        d_results = [r for r in results if r.domain == d]
        no_agree = sum(1 for r in d_results if not r.agreement_reached)
        print(f"    {d:<20} {no_agree:4d} / {len(d_results):4d} ({no_agree/len(d_results)*100:.1f}%)" if d_results else f"    {d:<20} N/A")

    # Non-agreement by config
    print(f"\n  Non-agreements by config:")
    for c in CONFIGS:
        c_results = [r for r in results if r.config == c]
        no_agree = sum(1 for r in c_results if not r.agreement_reached)
        print(f"    {c:<10} {no_agree:4d} / {len(c_results):4d} ({no_agree/len(c_results)*100:.1f}%)" if c_results else f"    {c:<10} N/A")


# =========================================================================
# Main
# =========================================================================

def main():
    print("Extended Analysis for ANL2026 Concealment Paper\n")
    print(f"Loading traces from {TRACES_FILE}...")
    results = load_traces()
    if not results:
        print("[ERROR] No results loaded.")
        sys.exit(1)
    print(f"Loaded {len(results)} negotiation results\n")

    # 1. Per-domain
    analysis_per_domain(results)

    # 2. Per-opponent
    analysis_per_opponent(results)

    # 3. Ablation
    analysis_ablation(results)

    # 4. Tradeoff plots
    analysis_tradeoff(results)

    # 5. Failures
    analysis_failures(results)

    print("\n" + "=" * 60)
    print("All analyses complete.")
    print(f"Outputs in: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
