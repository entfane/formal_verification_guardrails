import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.mixture import GaussianMixture
from datasets import load_dataset
from verifier import Verifier
import os
import json


def fit_gmm_and_get_percentile(embeddings, n_components=2, cov_type="diag", seed=42):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=seed,
        reg_covar=1e-3,
    )
    gmm.fit(embeddings)
    scores = gmm.score_samples(embeddings)
    p5 = np.percentile(scores, 5)
    return gmm, p5


def count_inside(embeddings, gmm, p5_thresh):
    """Count how many embedding rows fall inside the 5th-percentile log-likelihood bound of the fitted GMM."""
    scores = gmm.score_samples(embeddings)
    inside_mask = (scores > p5_thresh)
    return int(inside_mask.sum()), int((~inside_mask).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit a GMM to Toxigen construction data, then evaluate containment on a "
                    "held-out toxicity dataset (e.g. HateXplain, CivilComments, or a held-out "
                    "Toxigen target group). A point is considered 'inside' if its log-likelihood "
                    "exceeds the 5th percentile of the construction distribution."
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
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--output",     "-out", type=str, default=None,
                        help="Output jsonl path (default: results/gmm-cross-dataset/<model>_results.jsonl)")
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

    output_path = args.output or f"results/gmm-cross-dataset/{args.model.replace('/', '_')}_results.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for cov_type in ["full", "diag"]:
            for k in range(1, 6):

                try:
                    gmm, p5_threshold = fit_gmm_and_get_percentile(
                        construction_embeddings,
                        n_components=k,
                        cov_type=cov_type,
                        seed=args.seed
                    )
                except ValueError as e:
                    print(f"  Skipping K={k}, {cov_type}: GMM fit failed ({e})")
                    continue
                harmful_inside,  harmful_outside  = count_inside(harmful_emb,  gmm, p5_threshold)
                harmless_inside, harmless_outside = count_inside(harmless_emb, gmm, p5_threshold)

                tp = harmful_inside
                fp = harmless_inside
                fn = harmful_outside

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                result = {"K": k, "K_type": cov_type.upper(), "precision": precision, "recall": recall, "f1": f1}
                f.write(json.dumps(result) + "\n")
