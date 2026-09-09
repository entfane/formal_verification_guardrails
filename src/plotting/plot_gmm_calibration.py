import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: File not found: {path}")
        return []

# ── config ───────────────────────────────────────────────────────────────────

# model display name -> (results file, {"pess": tau, "star": tau})
MODELS = {
    "GPT-2": ("results/gmm-calibration/entfane_gpt2_constitutional_classifier_results.jsonl",
              {"pess": 0.15, "star": 0.37}),
    "BERT":  ("results/gmm-calibration/urbas_bert_aegis_results.jsonl",
              {"pess": 0.13, "star": 0.48}),
    "Llama": ("results/gmm-calibration/entfane_Toxic_Llama8B_results.jsonl",
              {"pess": 0.8, "star": 0.8}),
}

MARKERS = {"FULL": "o", "DIAG": "s"}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
})

# ── figure setup ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=0.0, vmax=1.0)

for ax, (model_name, (path, operating_taus)) in zip(axes, MODELS.items()):
    data = load_jsonl(path)

    for cov_type, marker in MARKERS.items():
        rows = [d for d in data if d["K_type"] == cov_type]
        if not rows:
            continue
        x = [d["predicted_coverage"] for d in rows]
        y = [d["actual_recall"] for d in rows]
        c = [d["tau"] for d in rows]
        ax.scatter(x, y, c=c, cmap=cmap, norm=norm, marker=marker,
                   s=32, alpha=0.85, edgecolors="none", zorder=2)

    # Highlight the model's own operating points (tau_pess / tau*), best-fit K only isn't
    # tracked here, so just outline every (K, cov) point at those exact taus.
    for tau in set(operating_taus.values()):
        rows = [d for d in data if abs(d["tau"] - tau) < 1e-9]
        for d in rows:
            ax.scatter(d["predicted_coverage"], d["actual_recall"],
                       facecolors="none", edgecolors="black",
                       marker=MARKERS[d["K_type"]], s=90, linewidths=1.2, zorder=3)

    # y = x reference (perfect calibration)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.5, zorder=1)

    ax.set_title(model_name, fontweight="bold")
    ax.set_xlabel("Predicted coverage (GMM)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal")

axes[0].set_ylabel("Actual held-out recall")

# ── shared legend + colorbar ─────────────────────────────────────────────────

legend_handles = [
    Line2D([0], [0], marker="o", color="grey", linestyle="none", markersize=6, label="FULL"),
    Line2D([0], [0], marker="s", color="grey", linestyle="none", markersize=6, label="DIAG"),
    Line2D([0], [0], marker="o", color="none", markeredgecolor="black", linestyle="none",
           markersize=8, label=r"$\tau_{pess}$ / $\tau^*$"),
    Line2D([0], [0], color="grey", linestyle=":", linewidth=1.5, label="Perfect calibration"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
cbar.set_label(r"$\tau$")

# ── save ─────────────────────────────────────────────────────────────────────

pdf_path = "results/plots/gmm_calibration.pdf"
png_path = "results/plots/gmm_calibration.png"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
print(f"Saved -> {pdf_path} / .png")
