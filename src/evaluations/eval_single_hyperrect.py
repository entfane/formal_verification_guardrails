import argparse
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_align_mat
from hyperrectangles import calculate_hyperrectangle
from verifier import Verifier
from datasets import load_dataset
import os


def get_classifier_head(model):
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    _, head = linear_layers[-1]
    return head.weight, head.bias


def split_dataset(dataset, label_col, n_eval_per_class=100, seed=42):
    """
    Shuffle the dataset, then reserve n_eval_per_class samples per label (0=harmless, 1=harmful)
    for evaluation. All remaining samples go to the construction set.

    Returns:
        construction_dataset : HF Dataset used to build the hyperrectangle
        eval_harmful_dataset : HF Dataset with label==1 eval samples
        eval_harmless_dataset: HF Dataset with label==0 eval samples
    """
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


def extract_and_align(verifier, dataset, classifier, tokenizer, pooling, input_col, output_col,
                      batch_size, max_len, align_mat):
    """Extract embeddings for a dataset and project with the alignment matrix."""
    embeddings = verifier.extract_embeddings(
        dataset, classifier, tokenizer, pooling, input_col, output_col, batch_size, max_len
    )
    return embeddings @ align_mat


def count_inside(embeddings, hyperrectangle):
    """Count how many embedding rows fall inside/outside the hyperrectangle."""
    lower = hyperrectangle[:, 0]
    upper = hyperrectangle[:, 1]
    mask = np.all((embeddings >= lower) & (embeddings <= upper), axis=1)
    return int(mask.sum()), int((~mask).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a hyperrectangle from most of the data, then evaluate "
                    "containment of a held-out eval set (100 harmful + 100 harmless)."
    )
    parser.add_argument("--model",      "-m", type=str, required=True,
                        help="HuggingFace model ID (e.g. 'bert-base-uncased')")
    parser.add_argument("--pooling",    "-p", type=str, choices=["first", "last"], required=True,
                        help="Pooling strategy: 'first' for encoder ([CLS]), 'last' for decoder")
    parser.add_argument("--dataset",    "-d", type=str, required=True,
                        help="HuggingFace dataset ID or path to a .jsonl file")
    parser.add_argument("--split",      "-s", type=str, required=True,
                        help="Dataset split (e.g. 'train')")
    parser.add_argument("--input-col",  "-i", type=str, required=True,
                        help="Column containing the user input text")
    parser.add_argument("--output-col", "-o", type=str, default=None,
                        help="Column containing the assistant output text (optional)")
    parser.add_argument("--label-col",  "-c", type=str, required=True,
                        help="Column containing the binary label (0=harmless, 1=harmful)")
    parser.add_argument("--batch-size", "-b", type=int, default=2,
                        help="Batch size for embedding extraction (default: 2)")
    parser.add_argument("--max-len",    "-l", type=int, default=128,
                        help="Max token length for tokenizer (default: 128)")
    parser.add_argument("--n-eval",     "-n", type=int, default=100,
                        help="Number of eval samples per class (default: 100)")
    parser.add_argument("--seed",             type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
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
    print(f"  Construction set (harmful only): {len(construction_ds)} records")
    print(f"  Construction set : {len(construction_ds)} records")
    print(f"  Eval harmful     : {len(eval_harmful_ds)} records  (label=1)")
    print(f"  Eval harmless    : {len(eval_harmless_ds)} records (label=0)")

    verifier = Verifier(args.pooling)

    construction_embeddings = verifier.extract_embeddings(
        construction_ds, classifier, tokenizer, args.pooling, 
        args.input_col, args.output_col, args.batch_size, args.max_len
    )
    align_mat = load_align_mat(args.dataset, args.model, construction_embeddings, False)
    construction_embeddings = construction_embeddings @ align_mat

    hyperrectangle = calculate_hyperrectangle(construction_embeddings)


    harmful_emb = extract_and_align(
        verifier, eval_harmful_ds, classifier, tokenizer, args.pooling,
        args.input_col, args.output_col, args.batch_size, args.max_len, align_mat
    )

    harmless_emb = extract_and_align(
        verifier, eval_harmless_ds, classifier, tokenizer, args.pooling,
        args.input_col, args.output_col, args.batch_size, args.max_len, align_mat
    )

    harmful_inside,  harmful_outside  = count_inside(harmful_emb,  hyperrectangle)
    harmless_inside, harmless_outside = count_inside(harmless_emb, hyperrectangle)

    total_inside  = harmful_inside  + harmless_inside
    total_outside = harmful_outside + harmless_outside
    total_eval    = total_inside    + total_outside

    tp = harmful_inside
    fp = harmless_inside
    fn = harmful_outside

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 55)
    print("  HYPERRECTANGLE CONTAINMENT RESULTS")
    print("=" * 55)
    print(f"{'Category':<20} {'Inside':>8} {'Outside':>8} {'% Inside':>10}")
    print("-" * 55)
    print(f"{'Harmful  (label=1)':<20} {harmful_inside:>8} {harmful_outside:>8} "
          f"{100 * harmful_inside  / len(eval_harmful_ds):>9.1f}%")
    print(f"{'Harmless (label=0)':<20} {harmless_inside:>8} {harmless_outside:>8} "
          f"{100 * harmless_inside / len(eval_harmless_ds):>9.1f}%")
    print("-" * 55)
    print(f"{'Total':<20} {total_inside:>8} {total_outside:>8} "
          f"{100 * total_inside / total_eval:>9.1f}%")
    print("=" * 55)
    print(f"  Precision : {100 * precision:>6.1f}%")
    print(f"  Recall    : {100 * recall:>6.1f}%")
    print(f"  F1        : {100 * f1:>6.1f}%")
    print("=" * 55)