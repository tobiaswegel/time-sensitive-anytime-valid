This repository contains anonymized supplementary code used to generate the submission figures.

The code was run with Python 3.13.5, NumPy 2.3.5, Matplotlib 3.10.8, and pandas 2.2.3. The Python package pins are also listed in `requirements.txt`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Recreating Figures

Each script can be run directly with `python3`. Generated figures, CSV summaries, and JSON result files are intentionally ignored by Git.

To recreate the exact heatmap, run:

```bash
python3 optimal_bet_heatmap_gaussian.py --horizon 150 --mu0 0 --mu1 0.6 --sigma 1 --no-linear --max-t-plot 50 --time-scale 30 --exp-time-scale 15 --beta 5 --cdf-mode normal --color-limits panel --quantity lambda
```

The remaining scripts generate the other submission images:

```bash
python3 Bernoulli_EDO_vs_EDO_capped_vs_Bellman.py
python3 gauss_GRO_vs_EDO.py
python3 hard_deadline_Bernoulli.py
python3 literature_comparison.py
python3 power_plot.py
```
