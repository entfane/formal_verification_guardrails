import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from hyperrectangles import compute_hyperrectangles
from verifier import Verifier
from datasets import load_dataset
import os
import json


MIN_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]


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
        description="Build hyperrectangles (via HDBSCAN) from Toxigen construction data, then "
                    "evaluate containment on a held-out toxicity dataset (e.g. HateXplain, "
                    "CivilComments, or a held-out Toxigen target group). A point is considered "
                    "'inside' if it falls inside ANY hyperrectangle."
    )
    parser.add_argument("--model",   "-m", type=str, required=True)
    parser.add_argument("--pooling", "-p", type=str, choices=["first", "last"], required=True)

    parser.add_argument("--construction-dataset", type=str, default="entfane/construction_points",
                        help="Toxigen construction dataset (toxic samples only)")
    parser.add_argument("--construction-split",   type=str, default="train")
    parser.add_argument("--construction-input-col",  type=str, default="prompt")
    parser.add_argument("--construction-output-col", type=str, default="generation")

    parser.add_argument("--eval-dataset",    type=str, required=True)
    parser.add_argument("--eval-split",      type=str, required=True)
    parser.add_argument("--eval-input-col",  type=str, required=True)
    parser.add_argument("--eval-output-col", type=str, default=None)
    parser.add_argument("--eval-label-col",  type=str, required=True,
                        help="Binary label column (0=harmless, 1=harmful)")

    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--max-len",    "-l", type=int, default=128)
    parser.add_argument("--output",     "-out", type=str, default=None,
                        help="Output jsonl path (default: results/multiple-hyperrect-cross-dataset/<model>_results.jsonl)")
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
    harmful_emb = verifier.extract_embeddings(
        eval_harmful_ds, classifier, tokenizer, args.pooling,
        args.eval_input_col, args.eval_output_col, args.batch_size, args.max_len
    )
    harmless_emb = verifier.extract_embeddings(
        eval_harmless_ds, classifier, tokenizer, args.pooling,
        args.eval_input_col, args.eval_output_col, args.batch_size, args.max_len
    )

    output_path = args.output or f"results/multiple-hyperrect-cross-dataset/{args.model.replace('/', '_')}_results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for min_cluster_size in MIN_SIZES:

            hyperrectangles, align_matrices = compute_hyperrectangles(
                construction_embeddings, min_cluster_size=min_cluster_size
            )

            harmful_inside,  harmful_outside  = count_inside(harmful_emb,  hyperrectangles, align_matrices)
            harmless_inside, harmless_outside = count_inside(harmless_emb, hyperrectangles, align_matrices)

            tp = harmful_inside
            fp = harmless_inside
            fn = harmful_outside

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            result = {"min_cluster_size": min_cluster_size, "num_hyperrectangles": len(hyperrectangles),
                      "precision": precision, "recall": recall, "f1": f1}
            f.write(json.dumps(result) + "\n")
