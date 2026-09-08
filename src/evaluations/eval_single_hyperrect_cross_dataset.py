import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_align_mat
from hyperrectangles import calculate_hyperrectangle
from verifier import Verifier
from datasets import load_dataset
import os
import json


def count_inside(embeddings, hyperrectangle):
    """Count how many embedding rows fall inside/outside the hyperrectangle."""
    lower = hyperrectangle[:, 0]
    upper = hyperrectangle[:, 1]
    mask = np.all((embeddings >= lower) & (embeddings <= upper), axis=1)
    return int(mask.sum()), int((~mask).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a hyperrectangle from Toxigen construction data, then evaluate "
                    "containment on a held-out toxicity dataset (e.g. HateXplain, "
                    "CivilComments, or a held-out Toxigen target group)."
    )
    parser.add_argument("--model",   "-m", type=str, required=True,
                        help="HuggingFace model ID (e.g. 'bert-base-uncased')")
    parser.add_argument("--pooling", "-p", type=str, choices=["first", "last"], required=True,
                        help="Pooling strategy: 'first' for encoder ([CLS]), 'last' for decoder")

    parser.add_argument("--construction-dataset", type=str, default="entfane/construction_points",
                        help="Toxigen construction dataset (toxic samples only)")
    parser.add_argument("--construction-split",   type=str, default="train")
    parser.add_argument("--construction-input-col",  type=str, default="prompt")
    parser.add_argument("--construction-output-col", type=str, default="generation")

    parser.add_argument("--eval-dataset",    type=str, required=True,
                        help="HuggingFace dataset ID or path to a .jsonl file")
    parser.add_argument("--eval-split",      type=str, required=True)
    parser.add_argument("--eval-input-col",  type=str, required=True)
    parser.add_argument("--eval-output-col", type=str, default=None)
    parser.add_argument("--eval-label-col",  type=str, required=True,
                        help="Binary label column (0=harmless, 1=harmful)")

    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--max-len",    "-l", type=int, default=128)
    parser.add_argument("--output",     "-out", type=str, default=None,
                        help="Output jsonl path (default: results/single-hyperrect-cross-dataset/<model>_results.jsonl)")
    args = parser.parse_args()

    tokenizer  = AutoTokenizer.from_pretrained(args.model)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model, device_map="auto", num_labels=1
    )
    classifier.eval()

    construction_ds = load_dataset(args.construction_dataset, split=args.construction_split)

    if args.eval_dataset.endswith(".jsonl") or os.path.isfile(args.eval_dataset):
        eval_ds = load_dataset(
            "json", data_files={args.eval_split: args.eval_dataset}, split=args.eval_split
        )
    else:
        eval_ds = load_dataset(args.eval_dataset, split=args.eval_split)

    eval_harmful_ds  = eval_ds.filter(lambda ex: ex[args.eval_label_col] == 1)
    eval_harmless_ds = eval_ds.filter(lambda ex: ex[args.eval_label_col] == 0)

    print(f"  Construction set (Toxigen, toxic only): {len(construction_ds)} records")
    print(f"  Eval harmful     : {len(eval_harmful_ds)} records  (label=1)")
    print(f"  Eval harmless    : {len(eval_harmless_ds)} records (label=0)")

    verifier = Verifier(args.pooling)

    construction_embeddings = verifier.extract_embeddings(
        construction_ds, classifier, tokenizer, args.pooling,
        args.construction_input_col, args.construction_output_col, args.batch_size, args.max_len
    )
    align_mat = load_align_mat(args.construction_dataset, args.model, construction_embeddings, False)
    construction_embeddings = construction_embeddings @ align_mat

    hyperrectangle = calculate_hyperrectangle(construction_embeddings)

    harmful_emb = verifier.extract_embeddings(
        eval_harmful_ds, classifier, tokenizer, args.pooling,
        args.eval_input_col, args.eval_output_col, args.batch_size, args.max_len
    ) @ align_mat

    harmless_emb = verifier.extract_embeddings(
        eval_harmless_ds, classifier, tokenizer, args.pooling,
        args.eval_input_col, args.eval_output_col, args.batch_size, args.max_len
    ) @ align_mat

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
    print("  HYPERRECTANGLE CONTAINMENT RESULTS (CROSS-DATASET)")
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

    output_path = args.output or f"results/single-hyperrect-cross-dataset/{args.model.replace('/', '_')}_results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"precision": precision, "recall": recall, "f1": f1}) + "\n")
