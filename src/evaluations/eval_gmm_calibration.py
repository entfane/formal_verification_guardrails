import argparse
import json
import math
import os

import numpy as np
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from verifier import Verifier

RESULTS_DIR = "results/gmm-calibration"
CACHE_DIR = ".embedding_cache"


def get_classifier_head(model):
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    _, head = linear_layers[-1]
    weight = head.weight.squeeze().detach().cpu().float().numpy()
    bias = head.bias.detach().cpu().float().numpy().item() if head.bias is not None else 0.0
    return weight, bias


def split_construction_eval(dataset, label_col, n_construction, n_eval, seed=42):
    """Keep only harmful (label==1) rows, shuffle, then split into a construction
    pool and a disjoint held-out eval pool."""
    harmful = dataset.filter(lambda ex: ex[label_col] == 1).shuffle(seed=seed)
    if len(harmful) < n_construction + n_eval:
        raise ValueError(
            f"Not enough harmful samples: need {n_construction + n_eval}, got {len(harmful)}"
        )
    construction = harmful.select(range(n_construction))
    eval_ds = harmful.select(range(n_construction, n_construction + n_eval))
    return construction, eval_ds


def get_embeddings(cache_key, dataset, verifier, classifier, tokenizer, pooling, input_col, output_col, batch_size, max_len):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    embeddings = verifier.extract_embeddings(
        dataset, classifier, tokenizer, pooling, input_col, output_col, batch_size, max_len
    )
    np.save(cache_path, embeddings)
    return embeddings


def fit_gmm(embeddings, n_components, cov_type, seed=42):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=seed,
        reg_covar=1e-3,
    )
    gmm.fit(embeddings)
    return gmm


def component_stats(gmm, weight, bias, cov_type):
    """Precompute per-component (pi_k, mean_y, std_y) of the pushed-forward
    pre-sigmoid score y = w . z + b, once per GMM (reused across every tau)."""
    stats = []
    for k in range(gmm.n_components):
        pi_k = gmm.weights_[k]
        mu_k = gmm.means_[k]
        sigma_k = gmm.covariances_[k]

        mean_y = float(weight @ mu_k) + bias
        if cov_type == "diag":
            var_y = float(np.sum(weight ** 2 * sigma_k))
        else:  # full
            var_y = float(weight.T @ sigma_k @ weight)
        std_y = math.sqrt(max(var_y, 1e-12))
        stats.append((pi_k, mean_y, std_y))
    return stats


def predicted_coverage(stats, tau):
    """Analytic P(sigmoid(y) > tau) under the fitted GMM, closed-form."""
    threshold_logit = math.log(tau / (1.0 - tau))
    total = 0.0
    for pi_k, mean_y, std_y in stats:
        z = (threshold_logit - mean_y) / std_y
        total += pi_k * norm.sf(z)
    return total


def actual_recall(eval_scores, tau):
    return float(np.mean(eval_scores > tau))


def log_result(record, results_path):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMM calibration check: compare the GMM's analytic predicted coverage "
                    "P(score > tau) against the real held-out recall at each tau, for a sweep "
                    "of tau values and (K, covariance_type) GMM configurations."
    )
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--pooling", "-p", type=str, choices=["first", "last"], required=True)

    parser.add_argument("--dataset", "-d", type=str, default="entfane/preprocessed_toxigen")
    parser.add_argument("--split", "-s", type=str, default="train")
    parser.add_argument("--input-col", "-i", type=str, default="prompt")
    parser.add_argument("--output-col", "-o", type=str, default="generation")
    parser.add_argument("--label-col", "-c", type=str, default="prompt_label")

    parser.add_argument("--n-construction", type=int, default=9000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--k-max", type=int, default=5, help="Sweep K = 1..k-max GMM components")
    parser.add_argument("--cov-types", type=str, default="full,diag",
                        help="Comma-separated covariance types to sweep (default: full,diag)")

    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tau-max", type=float, default=0.95)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--extra-taus", type=str, default=None,
                        help="Comma-separated extra tau values to include exactly "
                             "(e.g. a model's tau_pess and tau_star)")

    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--max-len", "-l", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Output jsonl path (default: results/gmm-calibration/<model>_results.jsonl)")
    args = parser.parse_args()

    taus = sorted(set(
        round(float(t), 6)
        for t in np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    ))
    if args.extra_taus:
        taus = sorted(set(taus) | {round(float(t), 6) for t in args.extra_taus.split(",")})

    cov_types = [c.strip() for c in args.cov_types.split(",") if c.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model, device_map="auto", num_labels=1
    )
    classifier.eval()

    dataset = load_dataset(args.dataset, split=args.split)
    construction_ds, eval_ds = split_construction_eval(
        dataset, args.label_col, args.n_construction, args.n_eval, seed=args.seed
    )

    verifier = Verifier(args.pooling)

    model_slug = args.model.replace("/", "_")
    cache_key_base = f"{model_slug}_{args.dataset.replace('/', '_')}_{args.split}_seed{args.seed}"

    print(f"Extracting construction embeddings ({len(construction_ds)} harmful samples)...")
    construction_emb = get_embeddings(
        f"{cache_key_base}_construction{args.n_construction}", construction_ds, verifier,
        classifier, tokenizer, args.pooling, args.input_col, args.output_col,
        args.batch_size, args.max_len
    )

    print(f"Extracting held-out eval embeddings ({len(eval_ds)} harmful samples)...")
    eval_emb = get_embeddings(
        f"{cache_key_base}_eval{args.n_eval}", eval_ds, verifier,
        classifier, tokenizer, args.pooling, args.input_col, args.output_col,
        args.batch_size, args.max_len
    )

    weight, bias = get_classifier_head(classifier)

    # Real classifier scores on the held-out set -> the ground-truth recall curve.
    eval_pre_sigm = eval_emb @ weight + bias
    eval_scores = 1.0 / (1.0 + np.exp(-eval_pre_sigm))

    output_path = args.output or os.path.join(RESULTS_DIR, f"{model_slug}_results.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    open(output_path, "w").close()  # truncate / start fresh

    for cov_type in cov_types:
        for k in range(1, args.k_max + 1):
            try:
                gmm = fit_gmm(construction_emb, n_components=k, cov_type=cov_type, seed=args.seed)
            except ValueError as e:
                print(f"  Skipping K={k}, {cov_type}: GMM fit failed ({e})")
                continue

            stats = component_stats(gmm, weight, bias, cov_type)

            for tau in taus:
                pred_cov = predicted_coverage(stats, tau)
                real_recall = actual_recall(eval_scores, tau)
                row = {
                    "model": args.model,
                    "K": k,
                    "K_type": cov_type.upper(),
                    "tau": tau,
                    "predicted_coverage": pred_cov,
                    "actual_recall": real_recall,
                    "n_construction": len(construction_emb),
                    "n_eval": len(eval_emb),
                }
                log_result(row, output_path)
            print(f"  K={k:<2d} {cov_type:<7s} done "
                  f"(tau range {taus[0]}-{taus[-1]}, {len(taus)} points)")

    print(f"\nWrote calibration results to {output_path}")
