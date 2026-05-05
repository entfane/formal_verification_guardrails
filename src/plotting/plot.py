import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_single(path):
    return load_jsonl(path)[0]

# ── load data ─────────────────────────────────────────────────────────────────

bert_multi   = load_jsonl("results/multiple-hyperrect/toxic_bert_hyperrect_results.jsonl")
gpt2_multi   = load_jsonl("results/multiple-hyperrect/toxic_gpt2_hyperrect_results.jsonl")
bert_single  = load_single("results/single-hyperrect/toxic_bert_single_hyperrect_results.jsonl")
gpt2_single  = load_single("results/single-hyperrect/toxic_gpt2_single_hyperrect_results.jsonl")

# Load new Llama data
llama_multi  = load_jsonl("results/multiple-hyperrect/toxic_llama_hyperrect_results.jsonl")
llama_single = load_single("results/single-hyperrect/toxic_llama_single_hyperrect_results.jsonl")

# ── colours & config ──────────────────────────────────────────────────────────

BERT_COLOR  = "#2563EB"
GPT2_COLOR  = "#DC2626"
LLAMA_COLOR = "#059669"  # Emerald Green for Llama

METRICS = [
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("f1",        "F1 Score"),
]

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

# ── plotting function ─────────────────────────────────────────────────────────

def plot_vertically(model_configs, filename_prefix):
    """
    Generates a figure with 3 vertically stacked subplots.
    model_configs: list of dictionaries containing model data and styling.
    """
    # 3 rows, 1 column. sharex=True keeps the x-axis labels only on the bottom plot.
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # Collect all x values to define global min/max boundaries
    all_x = []
    for cfg in model_configs:
        all_x.extend([r["min_cluster_size"] for r in cfg["multi_data"]])
    
    x_min = min(all_x)
    x_max = max(all_x)
    span = x_max - x_min

    for ax, (metric, title) in zip(axes, METRICS):
        
        for cfg in model_configs:
            name   = cfg["name"]
            m_data = cfg["multi_data"]
            s_data = cfg["single_data"]
            color  = cfg["color"]
            marker = cfg["marker"]

            x_vals = [r["min_cluster_size"] for r in m_data]
            y_vals = [r[metric] for r in m_data]
            h_val  = s_data[metric]

            # line graphs
            ax.plot(x_vals, y_vals,
                    color=color, marker=marker, linewidth=2, markersize=4,
                    label=f"{name} – multiple hyperrects")

            # horizontal baselines
            ax.hlines(h_val, x_min, x_max,
                      colors=color, linestyles="--", linewidth=1.6,
                      label=f"{name} – single hyperrect")

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)

        # y-axis limits & formatting
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)

        # ── FIXED: Dynamic x-ticks to prevent disappearing on smaller ranges ──
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ax.set_xlim(x_min - span * 0.02, x_max + span * 0.04)

    # Label only the bottom-most x-axis
    axes[-1].set_xlabel("Min Cluster Size")
    axes[-1].tick_params(axis="x", which="major", rotation=0)

    # shared legend placed at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=2, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()

    # save
    pdf_path = f"results/{filename_prefix}.pdf"
    png_path = f"results/{filename_prefix}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved → {pdf_path} / .png")

    # save
    pdf_path = f"results/{filename_prefix}.pdf"
    png_path = f"results/{filename_prefix}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved → {pdf_path} / .png")

# ── generate figures ──────────────────────────────────────────────────────────

# Figure 1: BERT and GPT-2
bert_gpt2_configs = [
    {"name": "BERT",  "multi_data": bert_multi, "single_data": bert_single, "color": BERT_COLOR, "marker": "o"},
    {"name": "GPT-2", "multi_data": gpt2_multi, "single_data": gpt2_single, "color": GPT2_COLOR, "marker": "s"}
]
plot_vertically(bert_gpt2_configs, "hyperrect_comparison_bert_gpt2")

# Figure 2: Llama
llama_configs = [
    {"name": "Llama", "multi_data": llama_multi, "single_data": llama_single, "color": LLAMA_COLOR, "marker": "^"}
]
plot_vertically(llama_configs, "hyperrect_comparison_llama")

plt.show()