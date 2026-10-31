import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base, outdir = sys.argv[1], sys.argv[2]
rows = [r for r in json.load(open(f"{base}/out1/set1_rows.json")) if r["aux"] == "paper"]
dhat = {}
for line in open(f"{base}/out_delta/test1_deltahat.md"):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 10 and c[0].isdigit():
        dhat[(int(c[0]), int(c[1]))] = float(c[9]) if c[9] != "void" else np.nan

plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.linewidth": 0.6})
SER = [  # key, marker, linestyle, colour, label, annotate
    ("et_orig", "s", "--", "#eb6834", r"Eq. (12), original", "above"),
    ("et_L2", "^", "--", "#e87ba4", r"Prop. 1 ($L^2$ route)", None),
    ("et_new", "o", "-", "#2a78d6", r"Cor. 1 with $\bar\delta_{h,i-1}$", "below"),
    ("hat", "D", "-", "#1baf7a", r"Cor. 1 with $\hat\delta_{h,i-1}$", None),
]
Ns = [40, 80, 160, 320]
fig, axes = plt.subplots(1, 2, figsize=(6.48, 1.8), sharey=True)
for ax, i, tag in zip(axes, [3, 4], ["(a)", "(b)"]):
    R = {r["N"]: r for r in rows if r["i"] == i}
    for key, mk, ls, col, lab, ann in SER:
        if key == "hat":
            v = np.array([dhat[(N, i)] for N in Ns])
        else:
            v = np.array([R[N]["g"] / R[N][key] if R[N][key] > 0 else np.nan for N in Ns])
        m = np.isfinite(v)
        N = np.array(Ns)[m]
        ax.plot(N, v[m], marker=mk, ls=ls, color=col, lw=1.0, ms=3.8 if mk != "D" else 3.4,
                mfc="white" if key == "hat" else col, mew=0.9, label=lab, clip_on=False, zorder=3)
        if ann:
            for x, y in zip(N, v[m]):
                txt = f"{y:.0f}" if y >= 100 else f"{y:.2f}"
                dy = (4, "bottom") if ann == "above" else (-4.5, "top")
                ax.annotate(txt, (x, y), xytext=(0, dy[0]), textcoords="offset points", ha="center",
                            va=dy[1], fontsize=6.3, color="0.15", zorder=4,
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none") if ann == "below" else None)
    ax.axhline(1.0, color="0.4", lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.set_xlim(34, 375)
    ax.set_title(f"{tag} $i={i}$", fontsize=8.5, pad=2, loc="left")
    ax.set_xlabel(r"$N$  ($h=\sqrt{2}/N$)", fontsize=8, labelpad=1)
    ax.minorticks_off(); ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
    ax.grid(True, which="major", lw=0.3, alpha=0.45)
axes[0].set_ylim(0.35, 420)
axes[0].set_yticks([1, 2, 5, 10, 20, 50, 100, 200]); axes[0].set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
axes[0].set_ylabel(r"$I^{\mathrm{lb}}$", fontsize=8, labelpad=1)
hs, ls_ = axes[0].get_legend_handles_labels()
fig.legend(hs, ls_, loc="upper center", ncol=4, fontsize=7.5, frameon=False, handlelength=2.4,
           columnspacing=1.4, borderaxespad=0.1)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88), w_pad=1.0)
fig.savefig(f"{outdir}/fig_set1.pdf")
print("ok")
