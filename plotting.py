"""Shared layout for the paper's rejection-time distributions."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils import aggregate_pmf_on_bins, fade_color, make_time_bins, tick_label_decimal


def plot_rejection_times(
    curves, colors, linestyles, plot_horizon, n_bins, style="bernoulli", ymax_bin=None
):
    """Plot binned rejection probabilities above CDFs in the paper's layout."""
    compact = style == "deadline"
    gaussian = style == "gaussian"
    tick_size = 9.0 if compact else 8.5
    label_size = 10.6 if compact else 10.2
    top_lw = 1.05 if compact else 1.15
    edge_lw = 0.75 if compact else 0.85
    cdf_lw = 1.55 if compact else 1.45
    face_blend = 0.80 if compact else 0.82
    edge_blend = 0.45 if compact else 0.50
    outer_margin = 0.06 if compact else 0.07
    inner_gap = 0.035 if compact else 0.040
    xtick_every = 1 if compact else 2

    times = np.arange(1, plot_horizon + 1)
    lefts, rights = make_time_bins(plot_horizon, n_bins)
    pmf_bins = [aggregate_pmf_on_bins(c["pmf"][:plot_horizon], lefts, rights) for c in curves]
    if ymax_bin is None:
        data_max = max(float(np.max(p)) for p in pmf_bins)
        if gaussian:
            ymax_bin = max(0.11, 1.08 * data_max)
        else:
            ymax_bin = math.ceil(10.0 * max(0.20, 1.08 * data_max)) / 10.0

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(4.15, 3.25) if compact else (6.2, 4.35),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    for j, (left, right) in enumerate(zip(lefts, rights)):
        if j % 2 == 0:
            ax1.axvspan(left, right, color="0.955", zorder=0)
        ax1.axvline(left, color="0.82", linewidth=0.45 if compact else 0.55, zorder=1)
    ax1.axvline(rights[-1], color="0.82", linewidth=0.45 if compact else 0.55, zorder=1)

    for j, (left, right) in enumerate(zip(lefts, rights)):
        width = right - left
        usable_width = width * (1.0 - 2.0 * outer_margin)
        gap = width * inner_gap
        bar_width = (usable_width - (len(curves) - 1) * gap) / len(curves)
        x_start = left + width * outer_margin
        for k, (pmf_bin, color, linestyle) in enumerate(zip(pmf_bins, colors, linestyles)):
            x0 = x_start + k * (bar_width + gap)
            y = pmf_bin[j]
            bar_style = (
                {"color": color, "alpha": 0.18}
                if gaussian
                else {"color": fade_color(color, blend=face_blend)}
            )
            ax1.bar(
                x=x0,
                height=y,
                width=bar_width,
                align="edge",
                edgecolor=fade_color(color, blend=edge_blend),
                linewidth=edge_lw,
                zorder=3,
                **bar_style,
            )
            ax1.hlines(y, x0, x0 + bar_width, color=color, linewidth=top_lw, zorder=4)
            cue_style = {"alpha": 0.85} if gaussian else {}
            ax1.vlines(
                x0 + 0.5 * bar_width,
                0.0,
                y,
                color=color,
                linewidth=0.9,
                linestyle=linestyle,
                zorder=4,
                **cue_style,
            )

    ax1.set_xlim(1, plot_horizon + 1)
    ax1.set_ylim(0.0, ymax_bin)
    ax1.grid(axis="y", alpha=0.18)
    top_ticks = [0.0, 0.05, 0.10] if gaussian else [0.0, 0.1, 0.2, 0.3, 0.4]
    top_ticks = [tick for tick in top_ticks if tick <= ymax_bin + 1e-12]
    ax1.set_yticks(top_ticks)
    tick_labels = (
        [r"$0$" if t == 0.0 else rf"${t:.2f}$" for t in top_ticks]
        if gaussian
        else [tick_label_decimal(t) for t in top_ticks]
    )
    ax1.set_yticklabels(tick_labels, fontsize=tick_size)
    ax1.tick_params(axis="y", length=2.5, width=0.7, pad=1)
    ax1.set_ylabel(r"$\pi_1^{\otimes}(\tau \in \mathrm{bin})$", fontsize=label_size)
    handles = [
        Line2D([0], [0], color=c, lw=top_lw + 0.9, linestyle=ls, label=curve["label"])
        for c, ls, curve in zip(colors, linestyles, curves)
    ]
    if compact:
        ax1.legend(
            handles=handles,
            loc="upper center",
            fontsize=8.4,
            ncol=3,
            frameon=True,
            handlelength=1.8,
            columnspacing=0.8,
            borderpad=0.22,
            bbox_to_anchor=(0.5, 1.03),
        )
    else:
        ax1.legend(
            handles=handles,
            loc="upper right",
            fontsize=8.2,
            ncol=2,
            frameon=True,
            handlelength=2.2,
            columnspacing=1.0,
            borderpad=0.25,
        )

    for curve, color, linestyle in zip(curves, colors, linestyles):
        ax2.step(
            times,
            curve["cdf"][:plot_horizon],
            where="post",
            color=color,
            linewidth=cdf_lw,
            linestyle=linestyle,
        )
    ax2.set_xlim(1, plot_horizon)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", alpha=0.18)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_yticklabels([r"$0$", r"$0.5$", r"$1$"], fontsize=tick_size)
    ax2.tick_params(axis="y", length=2.5, width=0.7, pad=1)
    ax2.set_ylabel(r"$\pi_1^{\otimes}(\tau \leq t)$", fontsize=label_size)

    centers = 0.5 * (lefts + rights - 1)
    bin_labels = [
        rf"${left}\!-\!{right - 1}$" if right - left > 1 else rf"${left}$"
        for left, right in zip(lefts, rights)
    ]
    ax1.set_xticks(centers[::xtick_every])
    ax1.set_xticklabels(
        bin_labels[::xtick_every],
        rotation=25 if compact else 30,
        ha="right",
        fontsize=tick_size,
    )
    ax1.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax1.set_xlabel("")
    if compact:
        ticks = np.arange(5, plot_horizon + 1, 5)
    else:
        ticks = np.arange(0, plot_horizon + 1, 20)
        ticks[0] = 1
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([rf"${int(t)}$" for t in ticks], fontsize=tick_size)
    ax2.tick_params(axis="x", length=2.5, width=0.7, pad=1)
    ax2.set_xlabel("rejection time", fontsize=label_size)
    fig.tight_layout(h_pad=0.28 if compact else 0.45)
    return fig
