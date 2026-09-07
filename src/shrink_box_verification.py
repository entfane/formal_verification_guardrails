import argparse
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from utils import load_align_mat
from verifier import Verifier
from deterministic_verification_filtered import get_classifier_head, filter_by_classifier_score

RESULTS_PATH = "results/shrunk-hyperrect/shrunk_results.jsonl"


def split_dataset(dataset, label_col, n_eval_per_class=100, seed=42):
    """Reserve n_eval_per_class harmful + n_eval_per_class harmless points for eval;
    everything else (harmful only) is the construction pool."""
    dataset = dataset.shuffle(seed=seed)
    harmful_idx  = [i for i, ex in enumerate(dataset) if ex[label_col] == 1]
    harmless_idx = [i for i, ex in enumerate(dataset) if ex[label_col] == 0]

    eval_harmful_idx  = set(harmful_idx[:n_eval_per_class])
    eval_harmless_idx = set(harmless_idx[:n_eval_per_class])
    eval_idx = eval_harmful_idx | eval_harmless_idx

    construction = dataset.select([i for i in range(len(dataset)) if i not in eval_idx])
    construction = construction.filter(lambda ex: ex[label_col] == 1)
    eval_harmful  = dataset.select(sorted(eval_harmful_idx))
    eval_harmless = dataset.select(sorted(eval_harmless_idx))
    return construction, eval_harmful, eval_harmless


def box_at_alpha(construction_aligned, alpha):
    """Per-axis: drop the outer alpha% on each side. alpha=0 -> exact min/max."""
    lo, hi = np.percentile(construction_aligned, [alpha, 100 - alpha], axis=0)
    return np.stack([lo, hi], axis=1)  # shape (D, 2), same format as calculate_hyperrectangle


def verify_at_alpha(verifier, construction_aligned, weights, bias, threshold, align_mat, alpha):
    box = box_at_alpha(construction_aligned, alpha)
    result, sigma_z_min = verifier.verify([box], weights, bias, threshold, [align_mat])
    return box, result, sigma_z_min


def sweep_alphas(verifier, construction_aligned, weights, bias, threshold, align_mat,
                  harmful_aligned, benign_aligned, step=0.1, alpha_max=49.0, on_step=None):
    """Gentle step-by-step sweep: alpha = 0, step, 2*step, ... until UNSAT (or alpha_max).
    Returns the list of per-step rows and the row where it first went UNSAT."""
    rows = []
    alpha = 0.0
    while alpha <= alpha_max:
        box, result, sigma_z_min = verify_at_alpha(
            verifier, construction_aligned, weights, bias, threshold, align_mat, alpha
        )
        row = {
            "alpha": alpha,
            "result": result,
            "sigma_z_min": float(sigma_z_min),
            "pct_harmful_inside": 100 * count_inside(harmful_aligned, box) / len(harmful_aligned),
            "pct_benign_inside": 100 * count_inside(benign_aligned, box) / len(benign_aligned),
        }
        rows.append(row)
        if on_step:
            on_step(row)
        if result == verifier.UNSAT:
            return rows, row
        alpha += step

    return rows, None  # never reached UNSAT


def count_inside(points_aligned, box):
    lo, hi = box[:, 0], box[:, 1]
    inside = np.all((points_aligned >= lo) & (points_aligned <= hi), axis=1)
    return int(inside.sum())


def log_result(record, results_path=RESULTS_PATH):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shrink the box per SVD axis until UNSAT; report held-out containment.")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--pooling", "-p", type=str, choices=["first", "last"], required=True)
    parser.add_argument("--dataset", "-d", type=str, default="entfane/preprocessed_toxigen")
    parser.add_argument("--split", "-s", type=str, default="train")
    parser.add_argument("--input-col", "-i", type=str, required=True)
    parser.add_argument("--output-col", "-o", type=str, default=None)
    parser.add_argument("--label-col", "-c", type=str, default="prompt_label")
    parser.add_argument("--threshold", "-t", type=float, required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--max-len", "-l", type=int, default=128)
    parser.add_argument("--n-eval", "-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.5, help="Alpha %% increment per sweep step")
    parser.add_argument("--alpha-max", type=float, default=49.0, help="Stop the sweep here if UNSAT is never reached")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    classifier = AutoModelForSequenceClassification.from_pretrained(args.model, device_map="auto", num_labels=1)
    classifier.eval()

    dataset = load_dataset(args.dataset, split=args.split)
    construction_ds, eval_harmful_ds, eval_harmless_ds = split_dataset(
        dataset, args.label_col, n_eval_per_class=args.n_eval, seed=args.seed
    )

    verifier = Verifier(args.pooling)
    weights, bias = get_classifier_head(classifier)
    weights = weights.squeeze().detach().cpu().float().numpy()
    bias = bias.squeeze().detach().cpu().float().numpy() if bias is not None else None

    construction_emb = verifier.extract_embeddings(
        construction_ds, classifier, tokenizer, args.pooling, args.input_col, args.output_col,
        args.batch_size, args.max_len
    )
    n_total = len(construction_emb)
    construction_emb, _ = filter_by_classifier_score(construction_emb, weights, bias, args.threshold)
    n_kept = len(construction_emb)
    print(f"Construction set: {n_total} -> {n_kept} points scoring >= {args.threshold} (dropped {n_total - n_kept})")

    align_mat = load_align_mat(args.dataset, args.model, construction_emb, False)
    construction_aligned = construction_emb @ align_mat

    harmful_aligned = verifier.extract_embeddings(
        eval_harmful_ds, classifier, tokenizer, args.pooling, args.input_col, args.output_col,
        args.batch_size, args.max_len
    ) @ align_mat
    benign_aligned = verifier.extract_embeddings(
        eval_harmless_ds, classifier, tokenizer, args.pooling, args.input_col, args.output_col,
        args.batch_size, args.max_len
    ) @ align_mat

    def on_step(row):
        print(f"alpha={row['alpha']:5.2f}%  {row['result']:5s}  sigma(z_min)={row['sigma_z_min']:.3e}  "
              f"harmful_inside={row['pct_harmful_inside']:5.1f}%  benign_inside={row['pct_benign_inside']:5.1f}%")
        log_result({"model": args.model, "tau": args.threshold,
                    "n_construction_kept": n_kept, "n_construction_total": n_total, **row})

    rows, unsat_row = sweep_alphas(
        verifier, construction_aligned, weights, bias, args.threshold, align_mat,
        harmful_aligned, benign_aligned, step=args.step, alpha_max=args.alpha_max, on_step=on_step
    )

    if unsat_row is None:
        print(f"Never reached UNSAT up to alpha={args.alpha_max}%; try a larger --alpha-max.")
    else:
        print(f"\nFirst UNSAT at alpha={unsat_row['alpha']:.2f}%  "
              f"(held-out harmful still inside: {unsat_row['pct_harmful_inside']:.1f}%)")
