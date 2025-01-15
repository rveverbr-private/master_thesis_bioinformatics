#!/usr/bin/env python
# coding: utf-8

from IPython import get_ipython
if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import scanpy as sc
import latenta as la
import jax
import jax.numpy as jnp
import optax
import tqdm.auto as tqdm
import scipy
import random
import re
import dill as pickle
import sklearn.decomposition
import os
from general_functions.model_generation_functions import *


# To load the pickled AnnData object back into memory
with open("../../real_data/wild_type_cells.pkl", "rb") as f:
    adata = pickle.load(f)


# basic model without effects.
base_model = create_base_model(adata)


def map_nested_fn(fn):
    """Recursively apply `fn` to key-value pairs of a nested dict."""
    def map_fn(nested_dict):
        return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}
    return map_fn



@map_nested_fn
def label_fn(k, v):
    # print(k)
    if ("kernel" in k) or ("bias" in k):
        return "a"
    else:
        return "b"

optimizer = optax.multi_transform({"b": optax.adam(5e-3), "a": optax.adam(1e-4)}, label_fn)

adata.X = adata.X.toarray()
adata.X


pca,cells,X =adata_pca(adata)
x = latent_x(pca,cells,X)
initialization_func1 = lambda: unbiased_initialization(x)
initialization_func1.__name__ = 'unbiased_initialization'
x_bias_loc_list = pre_caluclate_pca_biases(pca,cells,X)
placeholder = [0]
initialization_func2 = lambda: semi_biased_initialization(x,x_bias_loc_list, placeholder[0])
initialization_func2.__name__ = 'semi_biased_initialization'
initialization_funcs = [initialization_func1, initialization_func2]
np.random.seed(42)

_, model_dict = latenta_processes_basic(base_model, optimizer, "base_model", "base_model_0_0", 1000)
base_model = model_dict["base_model_0_0"][0]

scores, all_model_dict_per_generation, lineages = model_testing_genetic_algorithm(base_model, optimizer, "base_model", n_model_repeats=1, training_iterations=3000,
                          n_starting_population=100, algorithm_iterations=1, initialization_funcs=initialization_funcs, placeholder=placeholder)

dir_path = "random_model"

# Create the directory
os.makedirs(dir_path, exist_ok=True)


with open("random_model/scores.pkl", "wb") as f:
    pickle.dump(scores, f)


with open("random_model/dictionary.pkl", "wb") as f:
    pickle.dump(all_model_dict_per_generation, f)    


with open("random_model/lineages.pkl", "wb") as f:
    pickle.dump(lineages, f) 