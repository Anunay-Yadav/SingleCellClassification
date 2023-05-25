import scanpy as sc
from sklearn.metrics import classification_report
import numpy as np
#/Utils/preprocess.py
def preprocess(adata_test, min_genes = 200, min_cells = 20, target_sum = 1e6, n_top_genes = 3000, max_value = 10, get_hvgs=False, scale_and_hvgs = False, calculate_hvg_and_log1p = True):
        """
        INPUT:
        file_path: path to .h5ad containing scRNA-seq
        """
        ## convert to h5ad
        # adata_test = sc.AnnData(genes, labels)

        ## make var names unique
        adata_test.obs_names_make_unique()
        adata_test.var_names_make_unique()

        ## filter cells with count less than 200
        sc.pp.filter_cells(adata_test, min_genes=min_genes)

        ## filter genes with count less than 20
        sc.pp.filter_genes(adata_test, min_cells=min_cells)

        ## normalise data
        sc.pp.normalize_total(adata_test, target_sum=target_sum)

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

def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

def rank_transform(feats):
        '''converts features to ranks'''
        rows = []
        cols = list(feats.columns)
#       ind = list(feats.index)
        for ind in feats.index.to_list():
                c_r = []
                for col in cols:
                        c_r.append(feats.loc[ind,col])
                rows.append([int(x) for x in rankdata(c_r)])
        feats = pd.DataFrame(rows)
        feats.columns = cols
#       feats.index = ind
        return feats
    
def get_data_mapping(X, y):
        mapped_data = {}
        for i in range(len(y)):   
                if y[i] not in mapped_data:
                        mapped_data[y[i]] = []
                mapped_data[y[i]].append(X[i, :])
        return mapped_data

def get_mapping(y):
        labels = set(y)
        mapping = {}
        cnt = 0
        for lab in set(y):
                if lab in mapping:
                        continue
                mapping[lab] = cnt
                cnt += 1
        return mapping

def convert_y_to_mapping(y, mapping):
        y_lab = []
        for i in y:
                if i in mapping:
                        y_lab.append(mapping[i])
                else:
                        y_lab.append(0)
        return y_lab

def intersection_with_values(dict_1, values):
        print(dict_1, values)
        target_name = []
        for key in dict_1:
                if dict_1[key] in values:
                        target_name.append(key)
        return target_name

def train_model(clf, X_train, y_train, X_test, y_test, mapping):
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=intersection(list(mapping.keys()), set(y_test))))
        
def calculate_freq(X):
        freq_count = {}
        for i in X:
                if(i not in freq_count):
                        freq_count[i] = 0
                freq_count[i]+=1
        return freq_count

def train_test_split(adata, train_frac=0.85):
    """
        Split ``adata`` into train and test annotated datasets.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        train_frac: float
            Fraction of observations (cells) to be used in training dataset. Has to be a value between 0 and 1.

        Returns
        -------
        train_adata: :class:`~anndata.AnnData`
            Training annotated dataset.
        valid_adata: :class:`~anndata.AnnData`
            Test annotated dataset.
    """
    np.random.seed(0)
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data