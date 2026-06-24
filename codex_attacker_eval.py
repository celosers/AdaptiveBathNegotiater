r"""
Codex Task: Attacker Evaluation on Filtered Traces
===================================================
Input:  experiment_results/traces_5opp.csv (7,650 negotiations, 5 opponents, NO ShadowBath)
Output: experiment_results/analysis/raw_attacker_metrics.json

This is the SLOW part (~30-60 min). Do NOT aggregate — just compute per-negotiation
metrics and dump raw. The main agent will handle aggregation, statistics, and paper tables.

RUN:
    cd C:\Users\32546\Desktop\adaptive
    .\.venv\Scripts\python.exe codex_attacker_eval.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from negmas.inout import Scenario
from leakage_attackers import (
    ClassicFrequencyAttacker,
    RethinkingFrequencyAttacker,
    BayesianAttacker,
    evaluate_kendall_tau,
    evaluate_issue_weight_mae,
)


# =========================================================================
# Config
# =========================================================================
BASE = Path(__file__).parent
SCENARIOS_DIR = BASE / "scenarios"
TRACES_FILE = BASE / "experiment_results" / "traces_5opp.csv"
OUTPUT_DIR = BASE / "experiment_results" / "analysis"
OUTPUT_FILE = OUTPUT_DIR / "raw_attacker_metrics.json"


def _parse_bid_string(bid_str: str) -> tuple[Any, ...] | None:
    s = bid_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip().strip("'\"") for p in s.split(",")]
    return tuple(parts) if parts else None


def _extract_issue_space(bids: list[tuple[Any, ...]]) -> tuple[int, list[list[Any]]]:
    """Infer issue space from bid tuples."""
    if not bids or not bids[0]:
        return 0, []
    n = len(bids[0])
    domains: list[list[Any]] = [[] for _ in range(n)]
    for b in bids:
        for i, v in enumerate(b):
            if v not in domains[i]:
                domains[i].append(v)
    return n, domains


def load_traces_grouped(csv_path: str) -> list[dict]:
    """Load trace CSV, group by (domain, opponent, config, seed). Returns list of dicts."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["domain"], row["opponent"], row["config"], int(row["seed"]))
            groups[key].append(row)

    results = []
    for (domain, opponent, config, seed), rows in groups.items():
        our_bids = []
        agreement = False
        final_u = 0.0
        opponent_utility = 0.0

        for r in rows:
            if r["our_offer"] and r["our_offer"].strip():
                our_bids.append(r["our_offer"])
            if r.get("agreement_reached", "").lower() in ("true", "1", "yes"):
                agreement = True
            fu = float(r.get("final_self_utility", 0))
            if fu > 0:
                final_u = fu
            ou = float(r.get("opponent_utility", 0))
            if ou > 0:
                opponent_utility = max(opponent_utility, ou)

        results.append({
            "domain": domain,
            "opponent": opponent,
            "config": config,
            "seed": seed,
            "agreement": agreement,
            "self_utility": final_u,
            "opponent_utility": opponent_utility,
            "social_welfare": final_u + opponent_utility,
            "our_bids": our_bids,
            "n_bids": len(our_bids),
        })

    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("Codex: Attacker Evaluation (CF + RF + Bayesian)")
    print(f"Input: {TRACES_FILE}")
    print("=" * 60)

    # --- Load ---
    print("\n[1] Loading traces...")
    results = load_traces_grouped(str(TRACES_FILE))
    print(f"  Loaded {len(results)} negotiations")

    # Quick stats
    from collections import Counter
    config_cnt = Counter(r["config"] for r in results)
    domain_cnt = Counter(r["domain"] for r in results)
    opp_cnt = Counter(r["opponent"] for r in results)
    print(f"  Configs: {dict(config_cnt)}")
    print(f"  Domains: {dict(domain_cnt)}")
    print(f"  Opponents: {dict(opp_cnt)}")

    # --- Load scenarios ---
    print("\n[2] Loading scenarios...")
    scenarios = {}
    for domain in sorted(domain_cnt.keys()):
        path = SCENARIOS_DIR / domain
        if path.is_dir():
            s = Scenario.load(path, ignore_discount=True)
            if s is not None:
                scenarios[domain] = s
                print(f"  Loaded: {domain}")
            else:
                print(f"  [WARN] Could not load: {domain}")
        else:
            print(f"  [WARN] Not found: {domain}")

    # --- Attacker evaluation ---
    attacker_classes = [
        ("cf", ClassicFrequencyAttacker),
        ("rf", RethinkingFrequencyAttacker),
        ("bay", BayesianAttacker),
    ]

    print(f"\n[3] Evaluating attackers on {len(results)} negotiations...")
    print("    (This is the slow part. ETA ~30-60 min)")

    raw_metrics = []
    n_eval = 0
    skip_count = 0
    last_report = t0

    for i, r in enumerate(results):
        domain = r["domain"]
        scenario = scenarios.get(domain)

        metrics = {
            "domain": domain,
            "opponent": r["opponent"],
            "config": r["config"],
            "seed": r["seed"],
            "agreement": r["agreement"],
            "self_utility": r["self_utility"],
            "opponent_utility": r["opponent_utility"],
            "social_welfare": r["social_welfare"],
            "n_bids": r["n_bids"],
        }

        if scenario is None or not r["our_bids"]:
            skip_count += 1
            raw_metrics.append(metrics)
            continue

        # Parse bids
        parsed_bids = []
        for b_str in r["our_bids"]:
            p = _parse_bid_string(b_str)
            if p:
                parsed_bids.append(p)

        if not parsed_bids:
            skip_count += 1
            raw_metrics.append(metrics)
            continue

        ufun = scenario.ufuns[0]
        n_issues, issue_domains = _extract_issue_space(parsed_bids)

        # Get all outcomes for tau computation
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
                try:
                    tau = evaluate_kendall_tau(attacker, ufun, all_outcomes)
                except Exception:
                    tau = 0.0
                try:
                    mae = evaluate_issue_weight_mae(attacker, ufun, n_issues, issue_domains)
                except Exception:
                    mae = 0.0

            metrics[f"tau_{name}"] = round(tau, 6)
            metrics[f"mae_{name}"] = round(mae, 6)

        raw_metrics.append(metrics)
        n_eval += 1

        # Progress every 2 minutes or every 500 evaluations
        now = time.time()
        if n_eval % 500 == 0 or (now - last_report) > 120:
            elapsed = now - t0
            rate = n_eval / elapsed if elapsed > 0 else 0
            remaining = (len(results) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(results)}] {elapsed/60:.0f}min elapsed, "
                  f"~{remaining:.0f}min remaining, {skip_count} skipped")
            last_report = now

    elapsed = time.time() - t0
    print(f"\n  Done! Evaluated {n_eval}, skipped {skip_count}")
    print(f"  Total time: {elapsed/60:.1f} minutes")

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(raw_metrics)} records to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.0f} KB")
    print(f"\nHand this file back to the main agent for aggregation & paper filling.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
