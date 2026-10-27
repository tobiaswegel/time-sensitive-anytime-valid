# Time-sensitive anytime-valid testing

Code for Figures 1–6 of the paper. No datasets are needed.

## Reproduce all plots

Use Python 3.12 (tested with 3.12.7). From this repository's root, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python reproduce.py
```

The command uses the paper's simulation sizes and fixed seeds. It saves all plots,
numerical results, and logs in `results/`. On the tested Mac, the full run takes
about one minute. Figures 3 and 4 also have PDF versions.

| Figure | Plot | File in `results/` |
| --- | --- | --- |
| 1 | Gaussian optimal-action heatmaps | `figure_1.png` |
| 2 | Bernoulli hard deadlines versus GRO | `bernoulli_hard_deadline_vs_gro_stacked_small.png` |
| 3 | Bernoulli EDO, capped EDO, and Bellman | `bernoulli_edo_bellman_stacked_colorblind.png` |
| 4 | Gaussian GRO versus EDO | `gaussian_uncapped_gro_vs_approx_stacked.png` |
| 5 | Bernoulli power and bounds | `figure_5.png` |
| 6 | Comparison with horizon-aware baselines | `figure_6.png` |

To reproduce just one plot, for example Figure 3:

```bash
python reproduce.py --figures 3
```

`reproduce.py` contains the exact commands. Each plotting script contains its
settings and can also run independently. `results/run_manifest.json` records the
commands, Python version, package versions, and runtime of each requested figure.

Figure 1 uses logistic scale **5**, exponential scale **15**, and normal simulations,
which exactly reproduce the image in the supplied PDF. Appendix D.1 describes
different settings; see [verification notes](VERIFICATION.md). Fonts can affect
pixel-level rendering: the original figures use Times New Roman, with fallbacks
when it is unavailable.

## Check the calculations

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```
