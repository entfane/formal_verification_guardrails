import argparse
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from hyperrectangles import compute_hyperrectangles
from verifier import Verifier
from datasets import load_dataset
import os
import json


MIN_CLUSTER = 6


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


def count_inside(embeddings, hyperrectangles, align_matrices):
    """Count how many embedding rows fall inside at least one hyperrectangle,
    rotating each point with the corresponding cluster's alignment matrix."""
    inside_any = np.zeros(len(embeddings), dtype=bool)
    for rect, align_mat in zip(hyperrectangles, align_matrices):
        aligned = embeddings @ align_mat
        lower = rect[:, 0]
        upper = rect[:, 1]
        inside_any |= np.all((aligned >= lower) & (aligned <= upper), axis=1)
    return int(inside_any.sum()), int((~inside_any).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build hyperrectangles from harmful-only construction data, then evaluate "
                    "containment of a held-out eval set (100 harmful + 100 harmless). "
                    "A point is considered 'inside' if it falls inside ANY hyperrectangle."
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
    parser.add_argument("--min-cluster",       type=int, default=MIN_CLUSTER)
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

    min_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

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

    with open("new_results/multiple-hyperrect/toxic_llama_results.jsonl", 'w', encoding='utf-8') as f:
        for min_cluster_size in min_sizes:

            hyperrectangles, align_matrices = compute_hyperrectangles(
                construction_embeddings, min_cluster_size=min_cluster_size
            )

        

            harmful_inside,  harmful_outside  = count_inside(harmful_emb,  hyperrectangles, align_matrices)
            harmless_inside, harmless_outside = count_inside(harmless_emb, hyperrectangles, align_matrices)

            total_inside  = harmful_inside  + harmless_inside
            total_outside = harmful_outside + harmless_outside
            total_eval    = total_inside    + total_outside

            tp = harmful_inside
            fp = harmless_inside
            fn = harmful_outside

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            result = {"min_cluster_size": min_cluster_size, "num_hyperrectangles": len(hyperrectangles), "precision": precision, "recall": recall, "f1": f1}
            jsonl_line = json.dumps(result) + "\n"
            f.write(jsonl_line)

