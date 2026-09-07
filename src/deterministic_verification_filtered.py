import argparse
import json
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_align_mat
from hyperrectangles import calculate_hyperrectangle, compute_hyperrectangles
from verifier import Verifier
from datasets import load_dataset
import os

RESULTS_PATH = "results/filtered-hyperrect/filtered_construction_results.jsonl"


def log_result(model, threshold, n_kept, n_total, result, min_corner_score, results_path=RESULTS_PATH):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    record = {
        "model": model,
        "tau": threshold,
        "kept": n_kept,
        "total": n_total,
        "result": result,
        "sigma_z_min": float(min_corner_score),
    }
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_classifier_head(model):
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    _, head = linear_layers[-1]

    weights = head.weight
    bias = head.bias

    return weights, bias


def filter_by_classifier_score(embeddings, weights, bias, threshold):
    """
    Keep only the construction points the classifier itself already scores
    at or above the threshold (i.e. sigma(z) >= threshold), so the
    hyperrectangle is built exclusively from points the classifier gets right.
    """
    pre_sigm = embeddings @ weights
    if bias is not None:
        pre_sigm = pre_sigm + bias
    scores = 1 / (1 + np.exp(-pre_sigm))
    mask = scores >= threshold
    return embeddings[mask], scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify classifier using a hyper-rectangle built only from "
                    "construction points the classifier already scores >= threshold"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. 'bert-base-uncased')"
    )
    parser.add_argument(
        "--pooling", "-p",
        type=str,
        choices=["first", "last"],
        required=True,
        help="Pooling strategy: 'first' for encoder models ([CLS]), 'last' for decoder models"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="HuggingFace dataset ID (e.g. 'glue')"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        required=True,
        help="HuggingFace dataset split (e.g. 'train')"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        required=True,
        help="Classification threshold (e.g. 0.5)"
    )
    parser.add_argument("--input-col",  "-i", type=str,   required=True,  help="Dataset column to use as input text")
    parser.add_argument("--output-col", "-o", type=str, default=None, help="Dataset column to use as output label (optional)")
    parser.add_argument("--batch-size", "-b", type=int,   default=2,     help="Batch size for embedding extraction (default: 2)")
    parser.add_argument("--max-len",    "-l", type=int,   default=128,    help="Max token length for tokenizer (default: 128)")
    parser.add_argument("--use-single-hyper-rectangle", type = str, default="true", help="Boolean, whether to use a single hyper-rectangle or use multiple eps-cubes (default: single hyper-rectangle)")
    parser.add_argument("--min-cluster", type=int, default=5, help="Minimum cluster size (default: 5)")

    args = parser.parse_args()

    HF_MODEL    = args.model
    DATASET_NAME = args.dataset
    DATASET_SPLIT = args.split
    THRESHOLD   = args.threshold
    INPUT_COL = args.input_col
    OUTPUT_COL = args.output_col
    BATCH_SIZE = args.batch_size
    MAX_LEN = args.max_len
    POOLING = args.pooling
    SINGLE_HYPER_RECTANGLE = args.use_single_hyper_rectangle.lower() == "true"
    MIN_CLUSTER = args.min_cluster

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    classifier = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, device_map = "auto", num_labels = 1)
    classifier.eval()
    if DATASET_NAME.endswith(".jsonl") or os.path.isfile(DATASET_NAME):
        dataset = load_dataset("json", data_files={DATASET_SPLIT: DATASET_NAME}, split=DATASET_SPLIT)
    else:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    verifier = Verifier(POOLING)
    embeddings = verifier.extract_embeddings(dataset, classifier, tokenizer, POOLING, INPUT_COL, OUTPUT_COL, BATCH_SIZE, MAX_LEN)

    weights, bias = get_classifier_head(classifier)
    weights = weights.squeeze().detach().cpu().float().numpy()
    bias = bias.squeeze().detach().cpu().float().numpy() if bias is not None else None

    n_before = len(embeddings)
    embeddings, scores = filter_by_classifier_score(embeddings, weights, bias, THRESHOLD)
    n_after = len(embeddings)
    print(f"Construction set: {n_before} points -> {n_after} points scoring >= {THRESHOLD} "
          f"(dropped {n_before - n_after})")
    if n_after == 0:
        raise ValueError(f"No construction points scored >= {THRESHOLD}; cannot build a hyper-rectangle.")

    if SINGLE_HYPER_RECTANGLE:
        align_mat  = load_align_mat(DATASET_NAME, HF_MODEL, embeddings, False)
        embeddings = embeddings @ align_mat
        align_mat = [align_mat]
        hyperrectangles = [calculate_hyperrectangle(embeddings)]
    else:
        hyperrectangles, align_mat = compute_hyperrectangles(embeddings, min_cluster_size=MIN_CLUSTER)

    result, min_corner_score = verifier.verify(hyperrectangles, weights, bias, THRESHOLD, align_mat)
    print(result)
    print(f"sigma(z_min) = {min_corner_score}")
    if result == verifier.UNSAT:
        print(f"Everything inside the hyper-rectangle classified > {THRESHOLD}")
    else:
        print(f"There exists a point within the hyper-rectangle which is classified <= {THRESHOLD}")

    log_result(HF_MODEL, THRESHOLD, n_after, n_before, result, min_corner_score)
