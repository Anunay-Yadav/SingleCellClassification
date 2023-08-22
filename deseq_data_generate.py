import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch import distributions
import scanpy as sc
import os
from numpy.random import seed
# from tensorflow import set_random_seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
from pydeseq2.dds import *
dataset_train = 'dataset/Bh.h5ad'
dataset_test = 'dataset/smartseq2.h5ad'
import sys
sys.path.insert(0, '')

from scripts.utils import *
import numpy as np
torch.manual_seed(0)
np.random.seed(0)

def preprocess_deseq(adata_test, n_top_genes = 3000, max_value = 10, get_hvgs=False, scale_and_hvgs = False, calculate_hvg_and_log1p = True):
        """
        INPUT:
        file_path: path to .h5ad containing scRNA-seq
        """
        ## convert to h5ad
        # adata_test = sc.AnnData(genes, labels)
        adata_test.X = adata_test.to_df().to_numpy()
        ## make var names unique
        adata_test.obs_names_make_unique()
        adata_test.var_names_make_unique()
        adata_test = DeseqDataSet(adata = adata_test, design_factors='celltype')
        adata_test.vst()
        ## filter cells with count less than 200

        ## LogNormalise
        if not(scale_and_hvgs):
                return {'data':adata_test}

        if get_hvgs:
                ## Get HVGS
                # 
                if calculate_hvg_and_log1p:
                        sc.pp.log1p(adata_test)
                        sc.pp.highly_variable_genes(adata_test, n_top_genes = n_top_genes)
                adata_test = adata_test[:, adata_test.var.highly_variable]

                ## scale data
                sc.pp.scale(adata_test, max_value=max_value)
                return {'data' : adata_test, 'hvg': adata_test.var.highly_variable}

        ## scale data
        sc.pp.scale(adata_test, max_value=max_value)
        return {'data':adata_test}
    



adata_train=sc.read(dataset_train)
adata_test=sc.read(dataset_test)

print("Starting preprocessing...")
train_dic = preprocess_deseq(adata_train, get_hvgs = True, scale_and_hvgs = True)
test_dic = preprocess_deseq(adata_test)
print(len(intersection(adata_train.var.index, adata_test.var.index)))
list(adata_train.var.index)[0]

col= [i for i in train_dic['hvg'].index]

train_adata_pp =  train_dic['data']
print(train_dic['hvg'])
test_adata_pp =  test_dic['data'][:, intersection(col, test_dic['data'].var.index)]
train_adata_pp = train_dic['data'][:, intersection(col, train_dic['data'].var.index)]

train_df = train_adata_pp.to_df()
test_df = test_adata_pp.to_df()

## taking common genes
print("Taking common genes...")
final_columns = list(set(train_df.columns).intersection(set(test_df.columns)))
print('Common columns', len(final_columns))
final_columns = [i for i in final_columns if i != 'celltype']
train_df = train_df[final_columns]
test_df = test_df[final_columns]

y_train = train_adata_pp.obs.celltype.to_list()
y_test = test_adata_pp.obs.celltype.to_list()

X_train = train_df.to_numpy()
X_test = test_df.to_numpy()

labels = set(y_train)

mapping = get_mapping(y_train)

y_test = np.array(convert_y_to_mapping(y_test, mapping))

y_train = np.array(convert_y_to_mapping(y_train, mapping))

with open('dataset/np/X_train.pkl', 'wb') as fh:
    pickle.dump(X_train, fh)

with open('dataset/np/X_test.pkl', 'wb') as fh:
    pickle.dump(X_test, fh)

with open('dataset/np/y_test.pkl', 'wb') as fh:
    pickle.dump(y_test, fh)

with open('dataset/np/y_train.pkl', 'wb') as fh:
    pickle.dump(y_train, fh)