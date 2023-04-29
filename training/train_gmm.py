import os
import numpy as np
import pandas as pd
import argparse
import pickle
import tqdm
from sklearn import mixture, linear_model, svm, gaussian_process
from utils import *


base_dir = '/xfsdata/wpotosna'
data_dir = base_dir+'/disease_variant_prediction_language_model/data'

baseline_lm_data = pd.read_csv(data_dir+'/baseline_LM_evol_indices_dataset.csv')
eve_data = pd.read_csv(data_dir+'/EVE_evol_indices_dataset.csv')
lm_gnn_data = pd.read_csv(data_dir+'/LM_GNN_evol_indices_dataset.csv')

eve_data.set_index(['protein_name', 'mutations'], inplace=True)
baseline_lm_data.set_index(['protein_name', 'mutations'], inplace=True)
lm_gnn_data.set_index(['protein_name', 'mutations'], inplace=True)

common_proteins = np.array(list(set(eve_data.index.values) \
                                & set(baseline_lm_data.index.values)\
                                & set(lm_gnn_data.index.values)))
common_proteins = [tuple(i) for i in common_proteins]

indx = pd.MultiIndex.from_tuples(common_proteins, names=["first", "second"])
baseline_lm_data = baseline_lm_data.loc[indx].reset_index()
eve_data = eve_data.loc[indx].reset_index()
lm_gnn_data = lm_gnn_data.loc[indx].reset_index()


protein_GMM_weight = 0.3
for dataset_name in ['LM_Baseline', 'EVE', 'LM_GNN']:
    if dataset_name == 'LM_Baseline': dataset = baseline_lm_data
    elif dataset_name == 'EVE': dataset = eve_data
    elif dataset_name == 'LM_GNN': dataset = lm_gnn_data
    
    train_idxs, test_idxs = get_stratified_blocked_train_test_idxs(df=dataset, 
                                                               sample_id_label='protein_name', 
                                                               class_label='ClinVar_labels', 
                                                               test_size=0.20, 
                                                               random_state=0)

    X_train = dataset.iloc[train_idxs]
    X_test = dataset.iloc[test_idxs]
    
    dict_models, dict_pathogenic_cluster_index = get_gmm_parameters(dataset, X_train)
    train_results = compute_EVE_scores(X_train, dict_models, dict_pathogenic_cluster_index, protein_GMM_weight)
    test_results = compute_EVE_scores(X_test, dict_models, dict_pathogenic_cluster_index, protein_GMM_weight)

    train_results.to_csv(base_dir+f'/disease_variant_prediction_language_model/{dataset_name}_results/{dataset_name}_train_results.csv', index=False)
    test_results.to_csv(base_dir+f'/disease_variant_prediction_language_model/{dataset_name}_results/{dataset_name}_test_results.csv', index=False)

