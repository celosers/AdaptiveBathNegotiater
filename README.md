# AdaptiveBathNegotiator — ANL 2026

A preference-concealing automated negotiation agent for the Automated Negotiation League (ANL) 2026. Built on [NegMAS](https://negmas.readthedocs.io/).

## Quick Start

```bash
pip install -r requirements.txt
python main.py run
```

## Agent

`AdaptiveBathNegotiator` extends a monotonic time-dependent concession backbone with three concealment mechanisms:

- **Target Jitter** — perturbs aspiration to obscure concession patterns
- **Novelty Oscillation** — oscillates candidate scoring to reduce frequency leakage
- **Guarded Bluffing** — injects utility-bounded misleading bids

## Structure

```
├── adaptive_bath_agent.py   # Core agent
├── ceanl.py                 # ANL competition wrapper
├── main.py                  # CLI (Typer)
├── leakage_attackers.py     # Offline attacker models (CF, RF, Bayesian)
├── experiment_runner.py     # Batch experiment harness
├── exploitation_experiment.py # Exploitation loss evaluation
├── codex_attacker_eval.py   # Attacker metrics from bid traces
├── analyze_results.py       # Statistics aggregation
├── extended_analysis.py     # Visualizations
├── quick_per_domain_tau.py  # Per-domain analysis
├── requirements.txt
├── scenarios/               # 8 benchmark domains
├── examples/                # Opponent implementations
├── experiment_results/
│   └── analysis/            # Summary JSONs
└── paper/                   # LNCS manuscript
    ├── main.tex
    └── sn-bibliography.bib
```

## Domains

Camera, Car, Laptop, Grocery, ISBTAcquisition, Travel, Party, Energy

## Citation

```
@article{chen2026concealing,
  title={Concealing Preference Information in Automated Negotiation:
         A Multi-Stage Bidding Strategy Against Opponent Modeling},
  author={Chen, Long and Lv, Yichen and Fujita, Katsuhide and
          Chang, Shengbo and Wu, Zigao},
  journal={ANL 2026},
  year={2026}
}
```
