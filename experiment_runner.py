"""
Batch experiment runner for ANL2026 concealment ablation study.

Runs negotiations across domain × opponent × config × seed combinations,
logging per-round bid traces to CSV for offline attacker evaluation.

Usage:
    python experiment_runner.py           # run all combinations
    python experiment_runner.py --dry-run # print what would run
    python experiment_runner.py --domain Camera --config OFF,FULL  # subset
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from negmas.helpers import instantiate, get_full_type_name
from negmas.inout import Scenario
from negmas.sao import SAOMechanism
from negmas.preferences import compare_ufuns

# Our agent
from adaptive_bath_agent import AdaptiveBathNegotiator

# =========================================================================
# Configuration
# =========================================================================

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# All linear additive domains available in scenarios/
# (todo calls for 8; add more when available)
DOMAINS = [
    "Camera",
    "Car",
    "Laptop",
    "Grocery",
    "ISBTAcquisition",
    "Travel",
    "Party",
    "Energy",
]

# 6 opponent classes (negmas built-in + custom examples)
OPPONENTS: list[tuple[str, str]] = [
    # (display name, full class path)
    ("Boulware", "negmas.sao.BoulwareTBNegotiator"),
    ("Conceder", "negmas.sao.ConcederTBNegotiator"),
    ("Linear", "negmas.sao.LinearTBNegotiator"),
    ("MiCRO", "negmas.sao.MiCRONegotiator"),
    ("BOANeg", "examples.boa.BOANeg"),
    ("ShadowBath", "examples.simple.ShadowBathNegotiator"),
]

# 5 concealment configs per the todo + 1 random perturbation baseline
# (Jitter, NoveltyOscillation, Bluff)
CONFIGS: list[tuple[str, bool, bool, bool]] = [
    ("OFF", False, False, False),
    ("J-only", True, False, False),
    ("N-only", False, True, False),
    ("B-only", False, False, True),
    ("FULL", True, True, True),
    ("Random", False, False, False),  # random perturbation baseline, no concealment
    ("OM", False, False, False),      # opponent-model-driven concealment
]

N_SEEDS = 30
N_STEPS = 100

# Output directory (write to WorkBuddy workspace to avoid sandbox restrictions)
OUTPUT_DIR = Path.home() / "WorkBuddy" / "2026-06-18-23-08-13" / "experiment_results"
PROGRESS_FILE = OUTPUT_DIR / "_progress.json"
TRACES_FILE = OUTPUT_DIR / "traces_all.csv"


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class TraceRow:
    domain: str
    opponent: str
    config: str
    seed: int
    round_number: int
    relative_time: float
    our_offer: str
    opponent_offer: str
    self_utility: float
    opponent_model_utility: float
    agreement_reached: bool
    final_self_utility: float


# =========================================================================
# Main runner
# =========================================================================

def load_scenario(name: str) -> Scenario | None:
    path = SCENARIOS_DIR / name
    if not path.is_dir():
        print(f"  [WARN] Scenario dir not found: {path}")
        return None
    s = Scenario.load(path, ignore_discount=True)
    return s


def run_single_negotiation(
    domain_name: str,
    opponent_name: str,
    opponent_class: str,
    config_name: str,
    config_tuple: tuple[bool, bool, bool],
    seed: int,
) -> list[TraceRow] | None:
    """Run one negotiation and return per-round trace rows."""
    enable_jitter, enable_novelty, enable_bluff = config_tuple

    scenario = load_scenario(domain_name)
    if scenario is None:
        return None

    ufuns = scenario.ufuns
    if len(ufuns) < 2:
        return None

    # Our agent (always uses ufuns[0])
    use_random = (config_name == "Random")
    use_om = (config_name == "OM")
    our_agent = AdaptiveBathNegotiator(
        ufun=ufuns[0],
        name="OurAgent",
        enable_jitter=enable_jitter if not (use_random or use_om) else False,
        enable_novelty_oscillation=enable_novelty if not (use_random or use_om) else False,
        enable_bluff=enable_bluff if not (use_random or use_om) else False,
        use_random_perturbation=use_random,
        use_om_driven=use_om,
        perturbation_rate=0.12,
        rng_seed=seed,
    )

    # Opponent (uses ufuns[1])
    opp = instantiate(opponent_class, ufun=ufuns[1], name=opponent_name)

    mechanism = SAOMechanism(
        n_steps=N_STEPS,
        outcome_space=scenario.outcome_space,
    )
    mechanism.add(our_agent)
    mechanism.add(opp)

    try:
        mechanism.run()
    except Exception:
        return None

    # Extract per-round trace
    try:
        df = mechanism.full_trace_with_utils_df()
    except Exception:
        return None

    # Identify utility columns (UUID-based, one per agent)
    util_cols = [c for c in df.columns if c not in
                 ('time', 'relative_time', 'step', 'negotiator', 'offer',
                  'responses', 'state', 'text', 'data')]
    if len(util_cols) < 2:
        return None

    our_id = our_agent.id
    opp_id = opp.id
    our_col = None
    opp_col = None
    for col in util_cols:
        if our_id in col:
            our_col = col
        elif opp_id in col:
            opp_col = col

    if our_col is None or opp_col is None:
        return None

    agreement = mechanism.agreement
    agreement_reached = agreement is not None

    # Identify our agent's rows in the trace
    our_name = our_agent.name
    opp_name_actual = opp.name if hasattr(opp, 'name') else opponent_name

    # Final self utility
    if agreement_reached and agreement is not None:
        try:
            final_self_u = float(ufuns[0](agreement))
        except Exception:
            final_self_u = float(our_agent.ufun.reserved_value)
    else:
        final_self_u = float(our_agent.ufun.reserved_value)

    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        step = int(row.get("step", idx))
        relative_time = float(row.get("relative_time", step / N_STEPS))
        negotiator = str(row.get("negotiator", ""))
        offer = str(row.get("offer", ""))

        is_our_turn = our_name in negotiator or our_id in negotiator

        if is_our_turn:
            our_offer = offer
            opponent_offer = ""
            self_u = float(row.get(our_col, 0.0))
            opp_model_u = float(row.get(opp_col, 0.0))
        else:
            our_offer = ""
            opponent_offer = offer
            # Opponent's perspective: opp_col is opponent's self-utility
            self_u = float(row.get(our_col, 0.0))
            opp_model_u = float(row.get(opp_col, 0.0))

        rows.append(TraceRow(
            domain=domain_name,
            opponent=opponent_name,
            config=config_name,
            seed=seed,
            round_number=step,
            relative_time=relative_time,
            our_offer=our_offer,
            opponent_offer=opponent_offer,
            self_utility=self_u,
            opponent_model_utility=opp_model_u,
            agreement_reached=agreement_reached,
            final_self_utility=final_self_u,
        ))

    return rows


def build_run_key(domain: str, opponent: str, config: str, seed: int) -> str:
    return f"{domain}|{opponent}|{config}|{seed}"


def save_traces(rows: list[TraceRow], append: bool = True) -> None:
    """Save trace rows to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not TRACES_FILE.exists()

    with open(TRACES_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "domain", "opponent", "config", "seed",
                "round_number", "relative_time",
                "our_offer", "opponent_offer",
                "our_utility", "opponent_utility",
                "agreement_reached", "final_self_utility",
            ])
        for r in rows:
            writer.writerow([
                r.domain, r.opponent, r.config, r.seed,
                r.round_number, r.relative_time,
                r.our_offer, r.opponent_offer,
                r.self_utility, r.opponent_model_utility,
                r.agreement_reached, r.final_self_utility,
            ])


def load_progress() -> set[str]:
    """Load set of completed run keys."""
    if not PROGRESS_FILE.exists():
        return set()
    try:
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    except Exception:
        return set()


def save_progress(completed: set[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": sorted(completed), "updated": datetime.now().isoformat()}, f)


def generate_combinations():
    """Yield all (domain, opponent_name, opponent_class, config_name, config_tuple, seed)."""
    for domain in DOMAINS:
        for opp_name, opp_class in OPPONENTS:
            for cfg_name, j, n, b in CONFIGS:
                for seed in range(N_SEEDS):
                    yield domain, opp_name, opp_class, cfg_name, (j, n, b), seed


def main():
    dry_run = "--dry-run" in sys.argv

    # Parse optional filters
    domain_filter = None
    config_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--domain="):
            domain_filter = set(arg.split("=", 1)[1].split(","))
        if arg.startswith("--config="):
            config_filter = set(arg.split("=", 1)[1].split(","))

    total = len(DOMAINS) * len(OPPONENTS) * len(CONFIGS) * N_SEEDS
    print(f"Experiment matrix: {len(DOMAINS)} domains × {len(OPPONENTS)} opponents × {len(CONFIGS)} configs × {N_SEEDS} seeds = {total} negotiations")
    print(f"Output: {OUTPUT_DIR}")

    if dry_run:
        print("\n[Dry run — will not execute]")
        n = 0
        for domain, opp_name, opp_class, cfg_name, cfg, seed in generate_combinations():
            if domain_filter and domain not in domain_filter:
                continue
            if config_filter and cfg_name not in config_filter:
                continue
            n += 1
        print(f"Would run: {n} negotiations")
        return

    # Resume support
    completed = load_progress()
    if completed:
        print(f"Resuming: {len(completed)} already completed, skipping")

    batch_rows: list[TraceRow] = []
    n_done = 0
    n_skipped = 0
    n_errors = 0
    start_time = time.time()

    for domain, opp_name, opp_class, cfg_name, cfg, seed in generate_combinations():
        if domain_filter and domain not in domain_filter:
            continue
        if config_filter and cfg_name not in config_filter:
            continue

        key = build_run_key(domain, opp_name, cfg_name, seed)
        if key in completed:
            n_skipped += 1
            continue

        rows = run_single_negotiation(domain, opp_name, opp_class, cfg_name, cfg, seed)

        if rows is None:
            n_errors += 1
            # Mark as completed anyway to avoid retrying broken combos
            completed.add(key)
        else:
            batch_rows.extend(rows)
            completed.add(key)
            n_done += 1

        # Flush to disk every 100 negotiations
        if len(batch_rows) >= 50000 or n_done % 100 == 0:
            save_traces(batch_rows, append=True)
            save_progress(completed)
            batch_rows.clear()
            elapsed = time.time() - start_time
            rate = n_done / max(1, elapsed)
            remaining_total = total - n_done - n_skipped - n_errors
            eta = remaining_total / max(1, rate)
            print(f"  [{n_done}/{total}] {rate:.1f}/s, err={n_errors}, ETA={eta/60:.0f}min", flush=True)

        # Prevent memory leaks from negmas
        if n_done % 20 == 0:
            gc.collect()

    # Final flush
    if batch_rows:
        save_traces(batch_rows, append=True)
    save_progress(completed)

    elapsed = time.time() - start_time
    print(f"\nDone. {n_done} new runs, {n_skipped} skipped, {n_errors} errors in {elapsed/60:.1f}min")
    print(f"Traces: {TRACES_FILE}")
    print(f"Progress: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
