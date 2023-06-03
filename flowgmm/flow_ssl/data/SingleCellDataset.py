import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from os.path import join
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from oil.utils.utils import Expression,export,Named
import subprocess
from sklearn.model_selection import train_test_split

class SingleCellDataset(Dataset,metaclass=Named):
    num_classes = 14
    class_weights=None
    ignored_index=-100
    stratify=True
    def __init__(self,train=True):
        super().__init__()
        with open('dataset/X_train.pkl', 'rb') as fh:
            self.trn = pickle.load(fh)
            # self.trn = self.trn[:, :200]
        with open('dataset/y_train.pkl', 'rb') as fh:
            self.y_trn = pickle.load(fh)
        with open('dataset/X_test.pkl', 'rb') as fh:
            self.tst = pickle.load(fh)
            # self.tst = self.tst[:, :200]
        with open('dataset/y_test.pkl', 'rb') as fh:
            self.y_tst = pickle.load(fh)

        self.X = torch.from_numpy(self.trn if train else self.tst).float()
        self.y = torch.from_numpy(self.y_trn if train else self.y_tst).long()
        self.dim = self.X.shape[1]
        print(len(self))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

    # def show_histograms(self, split, vars):

    #     data_split = getattr(self, split, None)
    #     if data_split is None:
    #         raise ValueError('Invalid data split')

    #     util.plot_hist_marginals(data_split.x[:, vars])
    #     plt.show()

