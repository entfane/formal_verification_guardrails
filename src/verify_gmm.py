import argparse
import math
import os

import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def get_classifier_head(model):
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    _, head = linear_layers[-1]
    weight = head.weight.squeeze().detach().cpu().float().numpy()
    bias   = head.bias.detach().cpu().float().numpy().item() if head.bias is not None else 0.0
    return weight, bias


def build_chat_texts(dataset, input_col, output_col, tokenizer):
    texts = []
    for row in dataset:
        inpt = row[input_col]
        if output_col is not None:
            outpt = row[output_col]
            if tokenizer.chat_template:
                messages = [
                    {"role": "user", "content": inpt},
                    {"role": "assistant", "content": str(outpt)},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = inpt + "\n" + outpt
        else:
            text = inpt
        texts.append(text)
    return texts

def extract_embeddings(dataset, classifier, tokenizer, pooling, device, batch_size = 2, max_len = 128):
        embeds = []
        padding_side = "left" if pooling == "last" else "right"
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i: i + batch_size]
            
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            padding_side=padding_side, max_length=max_len).to(device)
            
            with torch.no_grad():
                hidden = classifier.base_model(**enc).last_hidden_state

            if pooling == "first":
                hiddens = hidden[:, 0]
            else:
                last_idxs = enc['attention_mask'].sum(dim=1) - 1
                hiddens = hidden[torch.arange(hidden.size(0)), last_idxs]
            embeds.append(hiddens.cpu().float())

        return torch.cat(embeds, dim=0).float().numpy()

def verify_gmm(embeddings, weight, bias, threshold, n_components, cov_type):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=0,
    )
    gmm.fit(embeddings)

    threshold_logit = math.log(threshold / (1.0 - threshold))

    total_prob = 0.0
    for k in range(n_components):
        pi_k = gmm.weights_[k]
        mu_k = gmm.means_[k]
        sigma_k = gmm.covariances_[k]

        mean_y = float(weight @ mu_k) + bias
        if cov_type == 'diag':
            var_y = float(np.sum(weight**2 * sigma_k))
        else:
            var_y = float(weight.T @ sigma_k @ weight)
        std_y = math.sqrt(max(var_y, 1e-12))

        z = (threshold_logit - mean_y) / std_y
        prob_k = norm.sf(z) # CDF (<= z)
        total_prob += pi_k * prob_k

    return total_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMM-based probabilistic verification of a classifier head"
    )
    parser.add_argument("--model",        "-m", type=str,   required=True,
                        help="HuggingFace model ID (e.g. 'bert-base-uncased')")
    parser.add_argument("--pooling",      "-p", type=str,   required=True,
                        choices=["first", "last"],
                        help="Pooling strategy: 'first' ([CLS]) or 'last' (decoder)")
    parser.add_argument("--dataset",      "-d", type=str,   required=True,
                        help="HuggingFace dataset ID or path to a .jsonl file")
    parser.add_argument("--split",        "-s", type=str,   required=True,
                        help="Dataset split (e.g. 'train', 'test')")
    parser.add_argument("--threshold",    "-t", type=float, required=True,
                        help="Classification threshold in (0, 1) (e.g. 0.5)")
    parser.add_argument("--input-col",    "-i", type=str,   required=True,
                        help="Dataset column to use as input text")
    parser.add_argument("--output-col",   "-o", type=str,   default=None,
                        help="Dataset column to use as output / response text (optional; "
                             "enables chat-template formatting when provided)")
    parser.add_argument("--n-components", "-n", type=int,   default=2,
                        help="Number of GMM components (default: 2)")
    parser.add_argument("--batch-size",   "-b", type=int,   default=2,
                        help="Batch size for embedding extraction (default: 2)")
    parser.add_argument("--max-len",      "-l", type=int,   default=128,
                        help="Max token length for tokenizer (default: 128)")
    parser.add_argument("--cov-type", type=str, default='full')
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, device_map="auto", num_labels=1
    )
    model.eval()

    if args.dataset.endswith(".jsonl") or os.path.isfile(args.dataset):
        dataset = load_dataset(
            "json",
            data_files={args.split: args.dataset},
            split=args.split,
        )
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    texts = build_chat_texts(dataset, args.input_col, args.output_col, tokenizer)

    embeddings = extract_embeddings(
        texts, model, tokenizer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=DEVICE,
    )

    weight, bias = get_classifier_head(model)

    prob = verify_gmm(embeddings, weight, bias, args.threshold, args.n_components, args.cov_type)

    print(f"P(score > {args.threshold}) = {prob}")