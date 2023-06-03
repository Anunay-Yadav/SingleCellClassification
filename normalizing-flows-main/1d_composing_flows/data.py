import torch
import torch.utils.data as data 
import numpy as np

import sys
 
sys.path.insert(0, '')

from scripts.utils import train_test_split, preprocess, get_data_mapping, intersection

import scanpy as sc

def generate_mixture_of_gaussians(num_of_points):
    n = num_of_points // 3
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=1.5, scale=0.35, size=(n,))
    gaussian3 = np.random.normal(loc=0.0, scale=0.2, size=(num_of_points-2*n,))
    return np.concatenate([gaussian1, gaussian2, gaussian3])

class NumpyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

def load_data(get_hvgs = True, scale_and_hvgs = True, calculate_hvg_and_log1p = True):
    dataset_train = '../../dataset/Baron_pancreatic_islet.h5ad'
    dataset_test = '../../dataset/smartseq2.h5ad'
    adata_train = sc.read(dataset_train)
    adata_test = sc.read(dataset_test)
    
    train_dic = preprocess(adata_train, min_cells=0,min_genes=0, get_hvgs = get_hvgs, scale_and_hvgs = scale_and_hvgs, calculate_hvg_and_log1p = calculate_hvg_and_log1p)
    test_dic = preprocess(adata_test, min_cells=0, min_genes=0)
    
    col= [i for i in train_dic['hvg'].index]
    print(col)
    
    train_adata_pp =  train_dic['data']
    test_adata_pp =  test_dic['data'][:, intersection(col, test_dic['data'].var.index)]
    train_adata_pp = train_dic['data'][:, intersection(col, train_dic['data'].var.index)]
    
    train_df = train_adata_pp.to_df()
    test_df = test_adata_pp.to_df()
    
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
    
    return X_train, X_test, y_train, y_test

def load_data_classwise(X, y):
    mapped_data = get_data_mapping(X, y)
    mapped_loader = {}
    for i in mapped_data:
        mapped_loader[i] = data.DataLoader(NumpyDataset(mapped_data[i]), batch_size=64, shuffle=True)
    return mapped_loader