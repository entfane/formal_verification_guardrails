from typing import List
import numpy as np
import torch
from tqdm import tqdm

class Verifier:

    SAT = "SAT"
    UNSAT = "UNSAT"

    def __init__(self, pooling):
        self.pooling = pooling

    def __format_inputs(self, dataset, tokenizer, input_col, output_col):

        formatted_inputs = []
        if tokenizer.chat_template:
            
            for record in dataset:
                inpt = record[input_col]
                if output_col:
                    outpt = record[output_col]
                else:
                    outpt = ""
                messages = [{'role': 'user', 'content': inpt},
                            {'role': 'assistant', "content": outpt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                formatted_inputs.append(text)
            
        else:
            for record in dataset:
                inpt = record[input_col]
                if output_col:
                    outpt = record[output_col]
                else:
                    outpt = ""
                
                formatted_inputs.append(inpt + outpt)
        return formatted_inputs

    def extract_embeddings(self, dataset, classifier, tokenizer, pooling, input_col, output_col = None, batch_size = 2, max_len = 128, ):
        formatted_inputs = self.__format_inputs(dataset, tokenizer, input_col, output_col)
        padding = "left" if pooling == "last" else "right"
        embeds = []
        for i in tqdm(range(0, len(formatted_inputs), batch_size)):
            batch = formatted_inputs[i: i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            padding_side=padding, max_length=max_len).to(classifier.device)
            
            with torch.no_grad():
                hidden = classifier.base_model(**enc).last_hidden_state

            if self.pooling == "first":
                hiddens = hidden[:, 0]
            else:
                last_idxs = enc['attention_mask'].sum(dim=1) - 1
                hiddens = hidden[torch.arange(hidden.size(0)), last_idxs]
            embeds.append(hiddens.cpu().float())

        return torch.cat(embeds, dim=0).numpy()
    

            

    def verify(self, hyperrectangles: List, weights: np.ndarray, bias: float, threshold: float, align_matrices: List[np.ndarray]) -> str:

        for (hyperrectangle, align_mat) in zip(hyperrectangles, align_matrices):
            lo = hyperrectangle[:, 0]
            hi = hyperrectangle[:, 1]
            aligned_weights = weights @ align_mat
            worst_point = np.where(aligned_weights >= 0, lo, hi)
            pre_sigm = float(worst_point @ aligned_weights)
            if bias is not None:
                pre_sigm += bias
            z = 1/(1 + np.exp(-pre_sigm))
            if z <= threshold:
                return self.SAT
        
        return self.UNSAT