import numpy as np
import os


# Adapted from ANTONIONLP/ANTONIO
# Repository: https://github.com/ANTONIONLP/ANTONIO
def load_align_mat(dataset_name, encoding_model_name, data, load_saved_align_mat, path='datasets'):
    if load_saved_align_mat:
        align_mat = np.load(f'{path}/{dataset_name}/embeddings/{encoding_model_name}/align_mat.npy')

    else:
        # Rotate the data, aligning them to the axis
        _, _, vh = np.linalg.svd(a=data)
        align_mat = vh.T
        save_path = f'{path}/{dataset_name}/embeddings/{encoding_model_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(f'{save_path}/align_mat.npy', align_mat)

    return align_mat