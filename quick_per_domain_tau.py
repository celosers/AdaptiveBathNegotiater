"""
Quick per-domain tau computation — Bayesian attacker only.
Reuses analyze_results.py infrastructure.
"""
import sys, json, csv, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from negmas.inout import Scenario

# Reuse analysis infrastructure
from analyze_results import (
    load_traces, NegotiationResult,
    _parse_bid_string, _infer_domain_from_bids,
)
from leakage_attackers import BayesianAttacker, evaluate_kendall_tau

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
TRACES_FILE = Path(__file__).parent / "experiment_results" / "traces_all.csv"
OUTPUT_FILE = Path(__file__).parent / "experiment_results" / "analysis" / "per_domain_tau.json"

DOMAINS = ["Camera", "Car", "Laptop", "Grocery", "ISBTAcquisition", "Travel", "Party", "Energy"]
CONFIGS = ["OFF", "J-only", "N-only", "B-only", "FULL", "Random"]

def load_scenario(domain):
    path = SCENARIOS_DIR / domain
    if not path.is_dir():
        return None
    try:
        return Scenario.load(path, ignore_discount=True)
    except Exception:
        return None

def mean_ci(vals):
    if not vals: return 0, 0, 0
    arr = np.array(vals, dtype=float)
    m = float(np.mean(arr))
    if len(arr) < 2: return m, m, m
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return m, m - 1.96*se, m + 1.96*se

print("Loading traces...")
results = load_traces(str(TRACES_FILE))
print(f"Loaded {len(results)} results")

print("Loading scenarios...")
scenarios = {}
for domain in DOMAINS:
    s = load_scenario(domain)
    if s is not None:
        scenarios[domain] = s
        print(f"  {domain}: OK")
    else:
        print(f"  {domain}: FAILED")

# Per (domain, config) metric lists
per_domain = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

n_eval = 0
for r in results:
    if r.config not in CONFIGS:
        continue
    if r.domain not in DOMAINS:
        continue

    # Basic metrics
    per_domain[r.domain][r.config]["self_utility"].append(r.final_self_utility)
    per_domain[r.domain][r.config]["agreement_rate"].append(1.0 if r.agreement_reached else 0.0)

    # Bayesian attacker
    if r.our_bids and r.domain in scenarios:
        parsed = []
        for b_str in r.our_bids:
            p = _parse_bid_string(b_str)
            if p:
                parsed.append(p)
        if parsed:
            try:
                attacker = BayesianAttacker()
                attacker.fit(parsed)
                ufun = scenarios[r.domain].ufuns[0]
                os_obj = scenarios[r.domain].outcome_space
                all_outcomes = list(os_obj.enumerate_or_sample(5000))
                if all_outcomes and len(all_outcomes) >= 2:
                    tau = evaluate_kendall_tau(attacker, ufun, all_outcomes)
                    per_domain[r.domain][r.config]["kendall_tau_bay"].append(tau)
                    n_eval += 1
            except Exception as e:
                pass

    if n_eval > 0 and n_eval % 500 == 0:
        print(f"  {n_eval} attacker evals done...")

print(f"  {n_eval} attacker evals total")

# Aggregate
output = {}
for domain in DOMAINS:
    output[domain] = {}
    for config in CONFIGS:
        if config not in per_domain[domain]:
            continue
        output[domain][config] = {}
        for metric in ["self_utility", "agreement_rate", "kendall_tau_bay"]:
            vals = per_domain[domain][config].get(metric, [])
            if not vals:
                continue
            m, lo, hi = mean_ci(vals)
            output[domain][config][metric] = {
                "mean": round(m, 6),
                "ci_lower": round(lo, 6),
                "ci_upper": round(hi, 6),
                "n": len(vals),
            }

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")

# Print summary table
print(f"\n{'Domain':<14} {'OFF τ_Bay':>10} {'FULL τ_Bay':>10} {'Δ':>8} {'OFF Util':>10} {'FULL Util':>10}")
print("-" * 64)
for domain in DOMAINS:
    off_tau = output[domain].get("OFF", {}).get("kendall_tau_bay", {}).get("mean", 0)
    full_tau = output[domain].get("FULL", {}).get("kendall_tau_bay", {}).get("mean", 0)
    off_util = output[domain].get("OFF", {}).get("self_utility", {}).get("mean", 0)
    full_util = output[domain].get("FULL", {}).get("self_utility", {}).get("mean", 0)
    d_tau = full_tau - off_tau
    print(f"{domain:<14} {off_tau:10.4f} {full_tau:10.4f} {d_tau:+8.4f} {off_util:10.4f} {full_util:10.4f}")
