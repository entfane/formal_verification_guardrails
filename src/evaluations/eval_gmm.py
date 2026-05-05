import argparse
import torch.nn as nn
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.mixture import GaussianMixture
from datasets import load_dataset
from verifier import Verifier


def get_classifier_head(model):
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    _, head = linear_layers[-1]
    return head.weight, head.bias


def split_dataset(dataset, label_col, n_eval_per_class=100, seed=42):
    dataset = dataset.shuffle(seed=seed)

    harmful_indices  = [i for i, ex in enumerate(dataset) if ex[label_col] == 1]
    harmless_indices = [i for i, ex in enumerate(dataset) if ex[label_col] == 0]

    if len(harmful_indices) < n_eval_per_class:
        raise ValueError(f"Not enough harmful samples: need {n_eval_per_class}, got {len(harmful_indices)}")
    if len(harmless_indices) < n_eval_per_class:
        raise ValueError(f"Not enough harmless samples: need {n_eval_per_class}, got {len(harmless_indices)}")

    eval_harmful_idx  = set(harmful_indices[:n_eval_per_class])
    eval_harmless_idx = set(harmless_indices[:n_eval_per_class])
    eval_indices      = eval_harmful_idx | eval_harmless_idx

    construction_indices = [i for i in range(len(dataset)) if i not in eval_indices]

    construction_dataset  = dataset.select(construction_indices)
    eval_harmful_dataset  = dataset.select(sorted(eval_harmful_idx))
    eval_harmless_dataset = dataset.select(sorted(eval_harmless_idx))

    return construction_dataset, eval_harmful_dataset, eval_harmless_dataset


def fit_gmm_and_get_percentiles(embeddings, n_components=2, cov_type="diag", seed=42):
    gmm = GaussianMixture(
        n_components=n_components, 
        covariance_type=cov_type, 
        random_state=seed
    )
    gmm.fit(embeddings)
    
    # Score the training data
    scores = gmm.score_samples(embeddings)
    
    # Find the exact scores that mark the 5% and 95% boundaries
    p5 = np.percentile(scores, 5)
    
    return gmm, p5


def count_inside(embeddings, gmm, p5_thresh):
    """
    Count how many embedding rows fall strictly inside the 5%-95% probability bounds
    of the fitted GMM based on log-likelihood scores.
    """
    scores = gmm.score_samples(embeddings)
    inside_mask = (scores > p5_thresh)
    return int(inside_mask.sum()), int((~inside_mask).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build GMM from harmful-only construction data, then evaluate "
                    "containment of a held-out eval set (100 harmful + 100 harmless). "
                    "A point is considered 'inside' if its log-likelihood is between the 5th and 95th percentiles."
    )
    parser.add_argument("--model",       "-m", type=str, required=True)
    parser.add_argument("--pooling",     "-p", type=str, choices=["first", "last"], required=True)
    parser.add_argument("--dataset",     "-d", type=str, required=True)
    parser.add_argument("--split",       "-s", type=str, required=True)
    parser.add_argument("--input-col",   "-i", type=str, required=True)
    parser.add_argument("--output-col",  "-o", type=str, default=None)
    parser.add_argument("--label-col",   "-c", type=str, required=True)
    parser.add_argument("--batch-size",  "-b", type=int, default=2)
    parser.add_argument("--max-len",     "-l", type=int, default=128)
    parser.add_argument("--n-eval",      "-n", type=int, default=100)
    parser.add_argument("--n-components","-k", type=int, default=2, help="Number of GMM components")
    parser.add_argument("--cov-type",          type=str, default="diag", choices=["full", "tied", "diag", "spherical"],
                        help="Covariance type for GMM. 'diag' is recommended for high-dimensional embeddings.")
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    tokenizer  = AutoTokenizer.from_pretrained(args.model)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model, device_map="auto", num_labels=1
    )
    classifier.eval()
    if args.dataset.endswith(".jsonl") or os.path.isfile(args.dataset):
        full_dataset = load_dataset(
            "json", data_files={args.split: args.dataset}, split=args.split
        )
    else:
        full_dataset = load_dataset(args.dataset, split=args.split)

    n_harmful_total  = sum(1 for ex in full_dataset if ex[args.label_col] == 1)
    n_harmless_total = sum(1 for ex in full_dataset if ex[args.label_col] == 0)

    construction_ds, eval_harmful_ds, eval_harmless_ds = split_dataset(
        full_dataset, args.label_col, n_eval_per_class=args.n_eval, seed=args.seed
    )

    construction_ds = construction_ds.filter(lambda ex: ex[args.label_col] == 1)

    # ── Embedding cache ───────────────────────────────────────────────────────
    cache_dir = ".embedding_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{args.model.replace('/', '_')}_{args.dataset.replace('/', '_')}_{args.split}_seed{args.seed}"
    construction_cache = os.path.join(cache_dir, f"{cache_key}_construction.npy")
    harmful_cache      = os.path.join(cache_dir, f"{cache_key}_eval_harmful.npy")
    harmless_cache     = os.path.join(cache_dir, f"{cache_key}_eval_harmless.npy")

    verifier = Verifier(args.pooling)

    if os.path.exists(construction_cache):
        construction_embeddings = np.load(construction_cache)
    else:
        construction_embeddings = verifier.extract_embeddings(
            construction_ds, classifier, tokenizer, args.pooling,
            args.input_col, args.output_col, args.batch_size, args.max_len
        )
        np.save(construction_cache, construction_embeddings)
    gmm, p5_threshold = fit_gmm_and_get_percentiles(
        construction_embeddings, 
        n_components=args.n_components, 
        cov_type=args.cov_type,
        seed=args.seed
    )
    

    if os.path.exists(harmful_cache):
        harmful_emb = np.load(harmful_cache)
    else:
        harmful_emb = verifier.extract_embeddings(
            eval_harmful_ds, classifier, tokenizer, args.pooling,
            args.input_col, args.output_col, args.batch_size, args.max_len
        )
        np.save(harmful_cache, harmful_emb)

    if os.path.exists(harmless_cache):
        harmless_emb = np.load(harmless_cache)
    else:
        harmless_emb = verifier.extract_embeddings(
            eval_harmless_ds, classifier, tokenizer, args.pooling,
            args.input_col, args.output_col, args.batch_size, args.max_len
        )
        np.save(harmless_cache, harmless_emb)

    harmful_inside,  harmful_outside  = count_inside(harmful_emb,  gmm, p5_threshold)
    harmless_inside, harmless_outside = count_inside(harmless_emb, gmm, p5_threshold)

    total_inside  = harmful_inside  + harmless_inside
    total_outside = harmful_outside + harmless_outside
    total_eval    = total_inside    + total_outside

    tp = harmful_inside
    fp = harmless_inside
    fn = harmful_outside

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print("  GMM CONTAINMENT RESULTS")
    print(f"  (Above 5% of construction log-likelihood)")
    print("=" * 60)
    print(f"{'Category':<20} {'Inside':>8} {'Outside':>8} {'% Inside':>10}")
    print("-" * 60)
    print(f"{'Harmful  (label=1)':<20} {harmful_inside:>8} {harmful_outside:>8} "
          f"{100 * harmful_inside  / len(eval_harmful_ds):>9.1f}%")
    print(f"{'Harmless (label=0)':<20} {harmless_inside:>8} {harmless_outside:>8} "
          f"{100 * harmless_inside / len(eval_harmless_ds):>9.1f}%")
    print("-" * 60)
    print(f"{'Total':<20} {total_inside:>8} {total_outside:>8} "
          f"{100 * total_inside / total_eval:>9.1f}%")
    print("=" * 60)
    print(f"  Precision : {100 * precision:>6.1f}%")
    print(f"  Recall    : {100 * recall:>6.1f}%")
    print(f"  F1        : {100 * f1:>6.1f}%")
    print("=" * 60)