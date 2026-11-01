# Figure 1 from results/test1.csv
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/test1.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "figures/fig1.pdf"
rows = list(csv.DictReader(open(src)))

plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.linewidth": 0.6})
series = [  # column, marker, line style, colour, label, value labels
    ("I_orig", "s", "--", "#eb6834", r"Eq. (12), original", "above"),
    ("I_L2", "^", "--", "#e87ba4", r"Prop. 1 ($L^2$ route)", None),
    ("I_dbar", "o", "-", "#2a78d6", r"Cor. 1 with $\bar\delta_{h,i-1}$", "below"),
    ("I_dhat", "D", "-", "#1baf7a", r"Cor. 1 with $\hat\delta_{h,i-1}$", None),
]
Ns = [40, 80, 160, 320]
fig, axes = plt.subplots(1, 2, figsize=(6.48, 1.8), sharey=True)
for ax, i, tag in zip(axes, [3, 4], ["(a)", "(b)"]):
    for col, mk, ls, color, label, ann in series:
        v = np.array([float(r[col]) for N in Ns for r in rows if int(r["N"]) == N and int(r["i"]) == i])
        ok = np.isfinite(v)  # void bounds are omitted
        N = np.array(Ns)[ok]
        ax.plot(N, v[ok], marker=mk, ls=ls, color=color, lw=1.0, ms=3.4 if mk == "D" else 3.8,
                mfc="white" if col == "I_dhat" else color, mew=0.9, label=label, clip_on=False, zorder=3)
        if not ann:
            continue
        for x, y in zip(N, v[ok]):
            ax.annotate(f"{y:.0f}" if y >= 100 else f"{y:.2f}", (x, y), xytext=(0, 4 if ann == "above" else -4.5),
                        textcoords="offset points", ha="center", va="bottom" if ann == "above" else "top",
                        fontsize=6.3, color="0.15", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none") if ann == "below" else None)
    ax.axhline(1.0, color="0.4", lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlim(34, 375)
    ax.set_title(f"{tag} $i={i}$", fontsize=8.5, pad=2, loc="left")
    ax.set_xlabel(r"$N$  ($h=\sqrt{2}/N$)", fontsize=8, labelpad=1)
    ax.minorticks_off()
    ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
    ax.grid(True, which="major", lw=0.3, alpha=0.45)
ticks = [1, 2, 5, 10, 20, 50, 100, 200]
axes[0].set_ylim(0.35, 420)
axes[0].set_yticks(ticks, [str(t) for t in ticks])
axes[0].set_ylabel(r"$I^{\mathrm{lb}}$", fontsize=8, labelpad=1)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=7.5, frameon=False,
           handlelength=2.4, columnspacing=1.4, borderaxespad=0.1)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88), w_pad=1.0)
fig.savefig(dst)
