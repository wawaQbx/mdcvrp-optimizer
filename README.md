# MDCVRP: Multi-Depot Fleet Routing Optimization Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OR-Tools](https://img.shields.io/badge/Google_OR--Tools-Optimization-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal.svg)

## Overview

An end-to-end optimization engine for the Multi-Depot Capacitated Vehicle Routing Problem (MDCVRP), benchmarked against 200 instances from the INCOM 2024 and MIM 2025 supply chain challenge sets.

The solver integrates dynamic multi-depot modeling with Guided Local Search (GLS) metaheuristics via Google OR-Tools, achieving a 100% feasible solution rate under 30-second per-instance computation limits.

---

## Results

Evaluated against a Nearest-Neighbor Greedy baseline across all 200 instances:

- 13.39% reduction in total fleet distance (1.1M+ km saved)
- 123 fewer active vehicles required across the network
- Outperformed baseline in 93% of instances (186/200)

![Benchmark Dashboard](results/plots/comprehensive_benchmark_dashboard.png)
*3x2 benchmark dashboard generated directly from pipeline outputs via `visualizer.py`*

---

## Architecture

Three decoupled modules:

- **`data_loader.py`** — parses heterogeneous `.vrp` and `.yaml` instance files; computes Euclidean distance matrices via NumPy
- **`solver.py`** — configures OR-Tools constraint solver with capacity dimensions and multi-depot index mappings; runs GLS optimization within a strict time budget; writes results to CSV asynchronously via `ThreadPoolExecutor`
- **`visualizer.py`** — reads `benchmark_results.csv` and generates Seaborn/Matplotlib dashboards and per-summary charts

---

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the solver
```bash
python solver.py
```
Outputs:
- `results/benchmark_results.csv` — per-instance results
- `results/benchmark_summary.csv` — aggregated statistics
- `output_plots/` — per-instance route visualization (200 PNG files)

> Note: `output_plots/` is not committed to keep the repository lightweight. Run `solver.py` locally to regenerate, or download the pre-packaged archive (~70 MB) from [Releases](../../releases).

### 3. Generate visualizations
```bash
python visualizer.py
```
Reads `results/benchmark_results.csv` and outputs the dashboard and 6 summary charts to `results/plots/`.

> Note: `visualizer.py` reads from the current working directory by default. Run from inside `results/`, or adjust the file path to point at `results/benchmark_results.csv`.

---

## Repository Structure

```
.
├── data/                   # MDCVRP instance files (.vrp + .yaml)
├── data_loader.py
├── solver.py
├── visualizer.py
├── results/
│   ├── benchmark_results.csv
│   ├── benchmark_summary.csv
│   └── plots/
├── output_plots/           # per-instance route plots (generated locally, not committed)
├── requirements.txt
└── README.md
```

---

## Dataset

Instances are drawn from the INCOM 2024 and MIM 2025 SimMD challenge sets, derived from the Supply Chain Disruption Monitoring Dataset [^1].

---

## License

MIT

---

## References

[^1]: Almahri, S., Xu, L., & Brintrup, A. (2026). *Supply Chain Disruption Monitoring Dataset*. https://github.com/sara-almahri/supply-chain-disruption-monitoring

```bibtex
@misc{almahri2026disruption,
  title={Supply Chain Disruption Monitoring Dataset},
  author={Almahri, Sara and Xu, Liming and Brintrup, Alexandra},
  year={2026},
  howpublished={\url{https://github.com/sara-almahri/supply-chain-disruption-monitoring}}
}
```
