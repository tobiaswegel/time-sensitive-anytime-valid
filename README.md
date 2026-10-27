# Time-sensitive anytime-valid testing

[![arXiv](https://img.shields.io/badge/arXiv-2605.06521-b31b1b.svg)](https://arxiv.org/abs/2605.06521)

This repository provides the code to reproduce the figures in the paper "Time-sensitive anytime-valid testing" by Eugenio Clerico, Tobias Wegel, Iskander Azangulov, and Patrick Rebeschini, published at NeurIPS 2026.

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
about one minute. Every plot is saved as PNG.

| Figure | Script | Outputs in `results/` |
| --- | --- | --- |
| 1 | `figure_1.py` | `figure_1.png` |
| 2 | `figure_2.py` | `figure_2.png` |
| 3 | `figure_3.py` | `figure_3.png` |
| 4 | `figure_4.py` | `figure_4.png` |
| 5 | `figure_5.py` | `figure_5.png` |
| 6 | `figure_6.py` | `figure_6.png` |

To reproduce just one plot, for example Figure 3:

```bash
python reproduce.py --figures 3
```

`reproduce.py` contains the exact commands. Each plotting script contains its
settings and can also run independently. `results/run_manifest.json` records the
commands, Python version, package versions, and runtime of each requested figure.

The plots use Times New Roman when available, with fallback fonts otherwise.

## Check the calculations

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{clerico2026time,
  title = {Time-sensitive anytime-valid testing},
  author = {Clerico, Eugenio and Wegel, Tobias and Azangulov, Iskander and Rebeschini, Patrick},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
