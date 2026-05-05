import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    try:
        with open(path) as f:
            # Sort by K immediately to ensure continuous lines
            data = [json.loads(line) for line in f if line.strip()]
            return sorted(data, key=lambda x: x["K"])
    except FileNotFoundError:
        print(f"Warning: File not found: {path}")
        return []

def split_by_k_type(data):
    """Splits data by K_type."""
    full_data = [d for d in data if d["K_type"] == "FULL"]
    diag_data = [d for d in data if d["K_type"] == "DIAG"]
    return full_data, diag_data

# ── config & colours ──────────────────────────────────────────────────────────

COLORS = {
    "BERT":  "#2563EB", # Blue
    "GPT-2": "#DC2626", # Red
    "Llama": "#059669", # Emerald Green
}

# Define the 3 metrics to plot horizontally
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

# ── load data ─────────────────────────────────────────────────────────────────

files = {
    "BERT":  "results/gmm/toxic_bert_gmm_results.jsonl",
    "GPT-2": "results/gmm/toxic_gpt2_gmm_results.jsonl",
    "Llama": "results/gmm/toxic_llama_gmm_results.jsonl",
}

# Load and preprocess all data beforehand to determine global x-limits
processed_data = {}
all_k = []
for model_name, path in files.items():
    data = load_jsonl(path)
    if data:
        full, diag = split_by_k_type(data)
        processed_data[model_name] = {"FULL": full, "DIAG": diag}
        all_k.extend([d["K"] for d in data])

# ── figure setup ──────────────────────────────────────────────────────────────

# 1 row, 3 columns. We don't share the y-axis because dynamic zoom ranges differ.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Calculate global x-limits based on all data
if all_k:
    x_min, x_max = min(all_k), max(all_k)
    span = x_max - x_min if x_max > x_min else 1
    x_lims = (x_min - span * 0.05, x_max + span * 0.05)
else:
    x_lims = (0, 10) # Fallback

# ── plotting ──────────────────────────────────────────────────────────────────

for ax, (metric_key, metric_title) in zip(axes, METRICS):
    
    # Track all y-values for this specific subplot so we can zoom in properly
    all_y_vals = []
    
    # Iterate through models to plot their lines on this specific axis
    for model_name, data_dict in processed_data.items():
        color = COLORS[model_name]
        
        full_data = data_dict["FULL"]
        diag_data = data_dict["DIAG"]

        # Plot FULL (solid line, circle markers)
        if full_data:
            k_vals = [d["K"] for d in full_data]
            y_vals = [d[metric_key] for d in full_data]
            all_y_vals.extend(y_vals)
            
            ax.plot(k_vals, y_vals,
                    color=color, linestyle="-", marker="o", 
                    linewidth=2, markersize=5,
                    label=f"{model_name} – FULL")

        # Plot DIAG (dashed line, square markers)
        if diag_data:
            k_vals = [d["K"] for d in diag_data]
            y_vals = [d[metric_key] for d in diag_data]
            all_y_vals.extend(y_vals)
            
            ax.plot(k_vals, y_vals,
                    color=color, linestyle="--", marker="s", 
                    linewidth=2, markersize=5,
                    label=f"{model_name} – DIAG")

    # ── aesthetics & formatting per subplot ──
    ax.set_title(metric_title, fontweight="bold")
    ax.set_xlabel("Number of Components (K)")
    ax.set_ylabel(metric_title)

    # ── DYNAMIC ZOOM LOGIC ──
    if all_y_vals:
        y_min = min(all_y_vals)
        y_max = max(all_y_vals)
        y_span = y_max - y_min if y_max > y_min else 0.1
        # Add 10% padding to the top and bottom of the data range
        ax.set_ylim(y_min - (y_span * 0.1), y_max + (y_span * 0.1))

    # Keep formatting nice
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    
    # Force integer K values on x-axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(x_lims)

# ── shared formatting & legend ───────────────────────────────────────────────

# Create a shared legend placed at the bottom, split into 3 columns for neatness
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", ncol=3, fontsize=10,
    framealpha=0.9, bbox_to_anchor=(0.5, -0.15),
)

# Adjust layout to make room for titles and the bottom legend
plt.tight_layout()

# ── save ──────────────────────────────────────────────────────────────────────

pdf_path = "results/gmm/gmm_full_comparison_horizontal.pdf"
png_path = "results/gmm/gmm_full_comparison_horizontal.png"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
print(f"Saved → {pdf_path} / .png")

plt.show()