# Verification notes

Checked on 27 October 2026 using the supplied `time_for_coins-7.pdf` and the
untouched scripts in `time-sensitive-anytime-valid-5CD7/`.

## Reproduction checks

- Ran all six original scripts at full simulation sizes, then ran all six cleaned
  scripts with the same settings. Before the requested Figure 5 restyling, all six
  PNG images were pixel-identical between the original and cleaned code. All three CSV files were byte-identical; all five
  saved JSON result objects were equal.
- Extracted the embedded Figure 1 image from the paper: the reproduced image was
  pixel-identical, including the heatmaps and simulated CDFs.
- Checked the other figures against the paper and its experimental descriptions.
  Their embedded PDF images are not all pixel-identical to the supplied scripts'
  output, so exact raster reproduction of those PDF images is not established.
  The numerical results are preserved; Figure 5 now uses the requested matching
  plot style.
- Ran ten independent numerical checks: exhaustive binary-path calculations for
  constant and Doob strategies; the stated Neyman–Pearson boundary sizes; null
  rejection control at deadlines 10, 20, 25, and 100; fairness and overshoot control
  of capped bets; Bernoulli EDO and power-exponent equations; Gaussian quadrature
  moments and EDO formulas; and analytic one-step Bellman values away from grid
  discontinuities. Also checked the two edge-case fixes below.
- Checked formatting, imports, function docstrings, and installed dependencies.

The checked environment is Python 3.12.7 on macOS, NumPy 2.3.5, Matplotlib 3.10.8,
and pandas 2.2.3, with Times New Roman. `requirements-lock.txt` records all runtime
package versions. The original README reported Python 3.13.5; that interpreter was
not available for this check. Font availability and rendering libraries can affect
image pixels on other systems.

## Paper details to reconcile

1. **Appendix D.1:** the actual Figure 1 and original README use logistic scale 5,
   exponential scale 15, and normal observations for the simulated CDFs. Appendix
   D.1 instead states scales 2 and 10 and quadrature observations. `reproduce.py`
   follows the actual image. The solver still uses 41-node Gaussian quadrature.
2. **Appendix D.2:** the code stops when the conditional event probability reaches
   `P0(event) / alpha`, as specified by Proposition 2 and Appendix E. D.2's wording
   about a prefix forcing event membership describes the stronger threshold 1.
3. **Figure 1 wealth label:** the plotted coordinates are `log(u)`, where
   `u = alpha * W`; the inherited axis label says `log w`. The image is preserved
   for reproduction, but the paper should clarify that the wealth is rescaled.
4. **Figure 6:** the Taga curve is the Taga-inspired heuristic described in
   Appendix E, rather than the authors' full reinforcement-learning procedure.

## Numerical scope

Bellman solutions use finite wealth/action grids and interpolation, not an exact
continuous-state optimum. Interpolation smooths value discontinuities near the
rejection boundary. Monte Carlo curves use the original seeds, batch sizes, and
sample sizes; those settings are retained because changing batching can change
random draws. Null rejection estimates can fluctuate slightly above 0.05: for
example the Figure 3 Bellman estimate is 0.05046 from 100,000 null paths.

The Figure 5 estimate stops paths far below the boundary and bounds their remaining
contribution, which is recorded in its JSON output. Power-one values use the
proposition rather than an arbitrarily long simulation.

## Edge-case fixes

The exact Bernoulli calculation now returns zero power when a deadline is too short
for any nonempty rejection event at the requested level. The power simulation now
rejects nonpositive sample sizes, batch sizes, and step limits, and invalid
truncation tolerances. In particular, batch size zero previously caused an infinite
loop. These changes do not affect any of the paper's default runs.

## Figure naming and formats

The plotting scripts are now `figure_1.py` through `figure_6.py`. Every script saves
both `figure_N.png` and `figure_N.pdf`; numerical outputs use the same stem.
Figure 5 uses the smaller fonts, line widths, horizontal grid, and compact legend
of the other distribution plots. This styling change leaves its estimates and
bounds unchanged. All six figures were regenerated, all twelve image files were
checked, and the numerical tests were rerun after these changes.
