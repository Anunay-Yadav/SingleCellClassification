import scanpy as sc
import os
from numpy.random import seed
# from tensorflow import set_random_seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle 

if __name__ == "__main__":
    dataset_train = 'dataset/Bh.h5ad'
    dataset_test = 'dataset/smartseq2.h5ad'
    import sys  
    sys.path.insert(0, '')
    
    from scripts.utils import * 
    
    if len(sys.argv) > 1: 
        dataset_train = sys.argv[1]
        dataset_test = sys.argv[2]
        
    adata_train=sc.read(dataset_train)
    adata_test=sc.read(dataset_test)
    
    print("Starting preprocessing...")
    train_dic = preprocess(adata_train, min_cells=20,min_genes=50, get_hvgs = True, scale_and_hvgs = True)
    test_dic = preprocess(adata_test, min_cells=0, min_genes=0)
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