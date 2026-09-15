"""
Session-extreme study — the three kept charts.

    1   timing of the session high and low, calendar time, vs. permutation null
    2   late trading relative to the WAP of the first 90% of the session
    3   joint timing of high and low

Input: a pickle {name: {p, v, obs, null}}
    p    (n, T)  price path per session on a fixed grid (per-minute VWAP bars)
    v    (n, T)  volume per bar
    obs  dict of per-session statistics
    null dict of the same statistics under within-session increment permutation

"""
import pickle, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ------------------------------------------------------------------ settings
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "res.npz")
OUT  = HERE
TERM = 0.10
JOINT_CLOCK = "cal"
LABELS = {"A_randomwalk": "A · random walk", "B_volburst": "B · late vol burst",
          "C_randsign": "C · late drift, random sign", "D_converge": "D · convergence"}

OBS, NUL, ACC = "#1b4965", "#c3ccd4", "#c1503f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 165})

def load(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        out = {}
        for key in z.files:
            parts = key.split("|")
            name = parts[0]
            d = out.setdefault(name, {})
            if len(parts) == 2:
                d[parts[1]] = z[key]
            else:
                d.setdefault(parts[1], {})[parts[2]] = z[key]
        return out
    with open(path, "rb") as fh:
        return pickle.load(fh)


R = load(SRC)
regimes = list(R)
LAB = {k: LABELS.get(k, k) for k in regimes}
T = R[regimes[0]]["p"].shape[1]
K = int(round(T * (1 - TERM)))
tgrid = (np.arange(T) + 1) / T


def note(ax, txt, x=.03, y=.97, ha="left"):
    ax.text(x, y, txt, transform=ax.transAxes, va="top", ha=ha, fontsize=8,
            bbox=dict(fc="white", ec="0.8", lw=.6, pad=2.5))


# 1
def fig_q1():
    fig, axes = plt.subplots(2, len(regimes), figsize=(2.75 * len(regimes), 5.2), sharex=True)
    b = np.linspace(0, 1, 21)
    for j, reg in enumerate(regimes):
        o, nl = R[reg]["obs"], R[reg]["null"]
        for row, key, name in [(0, "tau_h_cal", "high"), (1, "tau_l_cal", "low")]:
            ax = axes[row, j]
            ax.axvspan(1 - TERM, 1, color=ACC, alpha=.10)
            ax.hist(nl[key], bins=b, density=True, color=NUL, label="random-walk null")
            ax.hist(o[key], bins=b, density=True, histtype="step", color=OBS, lw=1.6,
                    label="observed")
            po = float((o[key] > 1 - TERM).mean())
            pn = float((nl[key] > 1 - TERM).mean())
            note(ax, f"in last {TERM:.0%} of session:\n{po:.0%}   (null {pn:.0%})   ×{po/pn:.1f}")
            ax.set_ylim(0, max(4.0, ax.get_ylim()[1]))
            if row == 0:
                ax.set_title(LAB[reg], loc="left", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"session {name}\ndensity")
            if row == 1:
                ax.set_xlabel("fraction of session elapsed")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower center")
    fig.suptitle("Do extremes come late?", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/extremes_late.png", bbox_inches="tight")
    plt.close(fig)


# 2
def fig_q4():
    fig, axes = plt.subplots(2, len(regimes), figsize=(2.75 * len(regimes), 5.6))
    for j, reg in enumerate(regimes):
        o, nl, p, v = R[reg]["obs"], R[reg]["null"], R[reg]["p"], R[reg]["v"]
        w = v[:, :K] / v[:, :K].sum(1, keepdims=True)
        ref = (p[:, :K] * w).sum(1)                       # WAP of the first 90%

        ax = axes[0, j]                                   # path relative to WAP90
        d = p - ref[:, None]
        qs = np.percentile(d, [10, 25, 50, 75, 90], axis=0)
        ax.axhline(0, color="0.6", lw=.8)
        ax.axvspan(1 - TERM, 1, color=ACC, alpha=.10)
        ax.fill_between(tgrid, qs[0], qs[4], color=OBS, alpha=.12, lw=0)
        ax.fill_between(tgrid, qs[1], qs[3], color=OBS, alpha=.25, lw=0)
        ax.plot(tgrid, qs[2], color=OBS, lw=1.6)
        ax.set_title(LAB[reg], loc="left", fontsize=9.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of session elapsed")
        if j == 0:
            ax.set_ylabel("price − WAP of first 90%\nEUR (median, 25–75, 10–90)")

        ax = axes[1, j]                                   # terminal VWAP vs WAP90
        lim = max(np.percentile(np.abs(nl["dev_cur"]), 99),
                  np.percentile(np.abs(o["dev_cur"]), 99))
        bb = np.linspace(-lim, lim, 41)
        ax.hist(nl["dev_cur"], bins=bb, density=True, color=NUL, label="null")
        ax.hist(o["dev_cur"], bins=bb, density=True, histtype="step", color=OBS, lw=1.6,
                label="observed")
        med = float(np.median(o["dev_cur"]))
        amed = float(np.median(np.abs(o["dev_cur"])))
        anull = float(np.median(np.abs(nl["dev_cur"])))
        ax.axvline(med, color=ACC, lw=1.4)
        note(ax, f"median {med:+.1f} EUR\ntypical size {amed:.1f} vs null {anull:.1f} EUR")
        ax.set_xlabel("last 10% VWAP − WAP_90  (EUR)")
        if j == 0:
            ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("here does late trading sit relative to the WAP_90?",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/vs_wap.png", bbox_inches="tight")
    plt.close(fig)


# 3
def fig_joint(clock=JOINT_CLOCK):
    kh, kl = ("tau_h_cal", "tau_l_cal") if clock == "cal" else ("tau_h_vol", "tau_l_vol")
    axis_lab = "fraction of session elapsed" if clock == "cal" else "volume-clock position"
    ncol = 2
    nrow = int(np.ceil(len(regimes) / ncol))
    fig = plt.figure(figsize=(3.9 * ncol, 3.9 * nrow))
    outer = gridspec.GridSpec(nrow, ncol, wspace=0.30, hspace=0.34)
    for j, reg in enumerate(regimes):
        o, nl = R[reg]["obs"], R[reg]["null"]
        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[j],
                                              width_ratios=[4, 1], height_ratios=[1, 4],
                                              hspace=.05, wspace=.05)
        axm = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=axm)
        axr = fig.add_subplot(gs[1, 1], sharey=axm)
        axm.axvspan(1 - TERM, 1, color=ACC, alpha=.07)
        axm.axhspan(1 - TERM, 1, color=ACC, alpha=.07)
        axm.scatter(o[kh], o[kl], s=5, alpha=.35, color=OBS, linewidths=0)
        axm.set_xlabel(f"high  ({axis_lab})", fontsize=8.5)
        axm.set_ylabel(f"low  ({axis_lab})", fontsize=8.5)
        b = np.linspace(0, 1, 26)
        axt.hist(nl[kh], bins=b, density=True, color=NUL)
        axt.hist(o[kh], bins=b, density=True, histtype="step", color=OBS, lw=1.2)
        axr.hist(nl[kl], bins=b, density=True, color=NUL, orientation="horizontal")
        axr.hist(o[kl], bins=b, density=True, histtype="step", color=OBS, lw=1.2,
                 orientation="horizontal")
        for a in (axt, axr):
            a.tick_params(labelbottom=False, labelleft=False, length=0)
            for sp in a.spines.values():
                sp.set_visible(False)
        axt.set_title(LAB[reg], loc="left", fontsize=8.5)
        axm.set_xlim(0, 1); axm.set_ylim(0, 1)
        axm.set_xticks([0, .5, 1]); axm.set_yticks([0, .5, 1])
        axt.set_ylim(bottom=0); axr.set_xlim(left=0)
    fig.suptitle("Joint timing of the session high and low",
                 y=.99, fontsize=10.5)
    fig.savefig(f"{OUT}/joint_timing.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_q1()
    fig_q4()
    fig_joint()
    print("written to", OUT)
