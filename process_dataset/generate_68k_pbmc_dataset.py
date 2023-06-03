

import pickle

import scanpy as sc
import os
from numpy.random import seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn import metrics
if __name__ == "__main__":
    import sys  
    sys.path.insert(0, 'scripts')
    
    from utils import * 
    dataset_train = 'dataset/68kPBMC_processed.h5ad'
    split = 0.80

    if len(sys.argv) > 1:
        dataset_train = sys.argv[1]
        split = float(sys.argv[2])
    adata_train, adata_test = train_test_split(sc.read(dataset_train), train_frac=split)

    print("Starting preprocessing...")
    train_dic = preprocess(adata_train, min_cells=0,min_genes=0, get_hvgs = True, scale_and_hvgs = True, calculate_hvg_and_log1p = False)
    test_dic = preprocess(adata_test, min_cells=0, min_genes=0)

    col= [i for i in train_dic['hvg'].index]
    print(col)

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

    mapped_data = get_data_mapping(X_train, y_train)

    for i in mapped_data:
        print(i, len(mapped_data[i]))
        
    mapping = get_mapping(y_train)

    y_test_lab = convert_y_to_mapping(y_test, mapping)
    y_test_lab = np.array(y_test_lab)

    y_train_lab = convert_y_to_mapping(y_train, mapping)
    y_train_lab = np.array(y_train_lab)

    with open('dataset/np/X_train.pkl', 'wb') as fh:
        pickle.dump(X_train, fh)

    with open('dataset/np/X_test.pkl', 'wb') as fh:
        pickle.dump(X_test, fh)
    
    with open('dataset/np/y_test.pkl', 'wb') as fh:
        pickle.dump(y_test_lab, fh)
    
    with open('dataset/np/y_train.pkl', 'wb') as fh:
        pickle.dump(y_train_lab, fh)