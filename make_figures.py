#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the resolution-condition figures from the CSVs written by run_resolution.py."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"]


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Render resolution figures from CSV")
    ap.add_argument("-o", "--output_dir", type=Path, default=Path("output"))
    ap.add_argument("-p", "--pol_degs", type=int, nargs="+", default=[1, 2, 3, 4])
    return ap.parse_args()


def _load(path):
    """Load a CSV file into a 2D array, or None if it doesn't exist."""
    if not path.exists():
        return None
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return np.atleast_2d(a)


def _fit(k, h):
    """Log-log slope fit; returns (slope, intercept) or (nan, nan)."""
    m = (k > 0) & (h > 0)
    if m.sum() < 2:
        return np.nan, np.nan
    s, b = np.polyfit(np.log(k[m]), np.log(h[m]), 1)
    return float(s), float(b)


def make_png(p, crit, sweep, out_png):
    """Build the 3-panel resolution figure for one polynomial degree."""
    k, khc, hc = crit[:, 0], crit[:, 1], crit[:, 2]
    s_pred = -(p + 1) / p
    s_fit, b_fit = _fit(k, hc)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    ax[0].loglog(k, hc, "o", color=PAL[0], ms=7, label="data")
    kk = np.linspace(k.min(), k.max(), 50)
    if np.isfinite(s_fit):
        ax[0].loglog(kk, np.exp(b_fit) * kk**s_fit, "-", color=PAL[0], lw=1.6,
                     label=f"fit slope {s_fit:.2f}")
    c0, k0 = hc[0], k[0]
    ax[0].loglog(kk, c0 * (kk / k0)**s_pred, "--", color="grey", lw=1.3,
                 label=rf"$k^{{{s_pred:.2f}}}$  ($k(kh)^p$=const)")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel(r"$h_{\rm crit}$")
    ax[0].set_title(f"critical mesh size vs k  (p={p})")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].loglog(k, khc, "o-", color=PAL[1], ms=6, label="data")
    ax[1].loglog(kk, khc[0] * (kk / k[0])**(-1.0 / p), "--", color="green", lw=1.1,
                 label=rf"$k^{{{-1.0/p:.2f}}}$ (predicted)")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel(r"$kh_{\rm crit}$")
    ax[1].set_title(r"critical $kh$ vs k")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which="both", alpha=0.3)

    if sweep is not None and sweep.size:
        cmap = plt.get_cmap("viridis")
        kvals = np.unique(sweep[:, 0])
        for j, kv in enumerate(kvals):
            m = sweep[:, 0] == kv
            o = np.argsort(sweep[m, 1])
            ax[2].loglog(sweep[m, 1][o], sweep[m, 2][o], "-o", ms=3,
                         color=cmap(j / max(1, len(kvals) - 1)), label=f"k={kv:g}")
        ax[2].legend(fontsize=7, ncol=2)
    ax[2].set_xlabel(r"$kh$")
    ax[2].set_ylabel(r"$\eta_{\rm FEM}/\eta_{\rm best}$")
    ax[2].set_title("ratio sweeps (smooth = no resonances)")
    ax[2].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Robin/Mie manufactured: critical resolution vs k, p={p}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return s_fit


def main():
    """Render figures for every polynomial degree that has a CSV in the output dir."""
    args = parse_args()
    for p in args.pol_degs:
        crit = _load(args.output_dir / f"critical_p{p}.csv")
        if crit is None:
            print(f"[skip] p={p}: no critical_p{p}.csv")
            continue
        sweep = _load(args.output_dir / f"sweep_p{p}.csv")
        png = args.output_dir / f"resolution_p{p}.png"
        s_fit = make_png(p, crit, sweep, png)
        print(f"[ok] p={p}: fitted slope {s_fit:.3f} (theory {-(p+1)/p:.3f}) "
              f"-> {png.name}")


if __name__ == "__main__":
    main()
