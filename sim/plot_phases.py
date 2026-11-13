"""Figure: three phases of unsteady Stokes cavity flow (dissipation, lid power, kinetic energy)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "cavity_phases_N64_dt0.000125.npz"
# optional: figure width in cm and output file (default: 16 cm, ../fig-cavity-phases.pdf)
#   e.g. python3 plot_phases.py cavity_phases_N64_dt0.000125.npz 13.8 ../pof/fig-cavity-phases.pdf
W_CM = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "../fig-cavity-phases.pdf"
d = np.load(src)
t, K, P, W, Pd, cr, nd, on, Wc = d["rec"].T
P_S, K_S, t_B, T_off = float(d["P_S"]), float(d["K_S"]), float(d["t_B"]), float(d["T_off"])
t_max = 0.30

# palette (reference instance of the dataviz method, light surface): series 1-3, ink, chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
})

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(W_CM / 2.54, 9.2 / 2.54),
                               gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.10})
sel = t <= t_max + 1e-12
for ax in (ax1, ax2):
    ax.set_facecolor("white")
    ax.axvspan(t_B, T_off, color=BAND, lw=0, zorder=0)          # phase B band
    for x in (t_B, T_off):
        ax.axvline(x, color=AXIS, lw=0.6, zorder=1)
    ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)

# --- top panel: power
P_top = 8.0
ax1.plot(t[sel], Wc[sel], color=ORANGE, lw=1.4, solid_joinstyle="round", zorder=3, label=r"lid power $\dot W_{\mathrm{lid}}$")
ax1.plot(t[sel], P[sel], color=BLUE, lw=1.4, solid_joinstyle="round", zorder=4, label=r"dissipation $\mathcal{P}$")
ax1.axhline(P_S, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
ax1.set_ylim(0, P_top)
ax1.set_yticks([0, 2, 4, 6, 8])
ax1.set_ylabel(r"power  [$\eta U^2$]")
ax1.text(0.298, P_S + 0.18, r"Helmholtz minimum $\mathcal{P}_S$", ha="right", va="bottom", fontsize=8, color=INK2, zorder=5)
ax1.text(0.0155, 6.55, r"$\dot W_{\mathrm{lid}}$", color=INK, fontsize=9, ha="left", va="center", zorder=6)
ax1.text(0.0155, 4.05, r"$\mathcal{P}$", color=INK, fontsize=9, ha="left", va="center", zorder=6)
ax1.text(0.2335, 1.55, r"$\dot W_{\mathrm{lid}} = 0$", color=INK, fontsize=8, ha="left", va="center", zorder=6)
ax1.text(0.123, 2.55, r"$\mathcal{P} = \dot W_{\mathrm{lid}} = \mathcal{P}_S$", color=INK, fontsize=8, ha="center", va="center", zorder=6)
ax1.text(0.2105, 2.55, r"$\mathcal{P} = -\dot{\mathcal{K}}$", color=INK, fontsize=8, ha="left", va="center", zorder=6)
leg = ax1.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.6, borderaxespad=0.2, labelcolor=INK2)
# phase labels
ytop = P_top * 1.06
ax1.text(0.0, ytop, "(a)  open, nonstationary", ha="left", va="bottom", fontsize=8, color=INK, clip_on=False)
ax1.text(0.5 * (t_B + T_off), ytop, "(b)  open, stationary", ha="center", va="bottom", fontsize=8, color=INK, clip_on=False)
ax1.text(0.5 * (T_off + t_max), ytop, "(c)  closed, relaxing", ha="center", va="bottom", fontsize=8, color=INK, clip_on=False)

# --- bottom panel: energy
ax2.plot(t[sel], K[sel], color=AQUA, lw=1.4, solid_joinstyle="round", zorder=3)
ax2.axhline(K_S, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
ax2.set_ylim(0, 0.024)
ax2.set_yticks([0, 0.01, 0.02])
ax2.set_ylabel(r"energy  [$\rho U^2 L^2$]")
ax2.set_xlabel(r"$t$  [$\rho L^2/\eta$]")
ax2.set_xlim(0, t_max)
ax2.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ax2.text(0.06, 0.0205, r"kinetic energy $\mathcal{K}$", color=INK, fontsize=9, ha="left", va="bottom", zorder=6)
ax2.text(0.298, K_S - 0.0012, r"$\mathcal{K}_S$", ha="right", va="top", fontsize=8, color=INK2)
ax2.text(0.213, 0.0125, r"$\dot{\mathcal{K}} + \mathcal{P} = 0$", color=INK, fontsize=8, ha="left", va="center", zorder=6)
ax2.text(0.031, 0.0055, "$\\dot{\\mathcal{K}} + \\mathcal{P}$\n$= \\dot W_{\\mathrm{lid}}$", color=INK, fontsize=8, ha="center", va="center", zorder=6, linespacing=1.1)

fig.subplots_adjust(left=1.68 / W_CM, right=1 - 0.24 / W_CM, top=0.90, bottom=0.135)  # margins fixed in cm
fig.savefig(OUT)
if len(sys.argv) <= 3:
    fig.savefig("fig-cavity-phases.png", dpi=170)
print("saved; P_S=%.4f K_S=%.5f t_B=%.4f" % (P_S, K_S, t_B))
