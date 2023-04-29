import numpy as np
import pandas as pd

from sklearn import mixture, linear_model, svm, gaussian_process
import tqdm

def get_stratified_blocked_train_test_idxs(df, sample_id_label, 
                                           class_label, test_size=0.20, 
                                           random_state=0):
    np.random.seed(random_state)
    
    class_mean = df.groupby([sample_id_label]).ClinVar_labels.mean()
    class_quantiles = pd.qcut(df.groupby('protein_name').ClinVar_labels.mean(), 
                              10)
    
    quantile_df = {}
    for gi, quant in enumerate(set(class_quantiles.values.to_list())):
        quantile_df[quant] = gi
    class_quantiles = class_quantiles.replace(quantile_df)
    
    train_sample_ids = []
    test_sample_ids = []
    for q in set(class_quantiles.values):
        b = class_quantiles[class_quantiles==q].copy().index.values
        test_sample_ids.extend(b[:round(len(b)*test_size)])
        train_sample_ids.extend(b[round(len(b)*test_size):])
    
    train_idxs = np.where(np.in1d(df[sample_id_label], 
                                  np.array(train_sample_ids)))[0]   
    test_idxs = np.where(np.in1d(df[sample_id_label], 
                                 np.array(test_sample_ids)))[0] 
                                
    return train_idxs, test_idxs


def get_gmm_parameters(X, X_train):
    dict_models = {}
    dict_pathogenic_cluster_index = {}

    main_GMM = mixture.GaussianMixture(n_components=2, 
                                       covariance_type='full',
                                       max_iter=20000,
                                       n_init=30,
                                       tol=1e-4)
    main_GMM.fit(X_train.evol_indices.values.reshape(-1, 1))

    dict_models['main'] = main_GMM
    #The pathogenic cluster is the cluster with higher mean value
    pathogenic_cluster_index = np.argmax(np.array(main_GMM.means_).flatten()) 
    dict_pathogenic_cluster_index['main'] = pathogenic_cluster_index

    for protein in tqdm.tqdm(X.protein_name.unique(), "Training all protein GMMs"):
        X_train_protein = X[X.protein_name==protein].evol_indices.values.reshape(-1, 1)
        
        if len(X_train_protein) > 0: #We have evol indices computed for protein on file
            protein_GMM = mixture.GaussianMixture(n_components=2,
                                                  covariance_type='full',
                                                  max_iter=1000,tol=1e-4,
                                                  weights_init=main_GMM.weights_,
                                                  means_init=main_GMM.means_,
                                                  precisions_init=main_GMM.precisions_)
            protein_GMM.fit(X_train_protein)
            dict_models[protein] = protein_GMM
            dict_pathogenic_cluster_index[protein] = np.argmax(np.array(protein_GMM.means_).flatten())

    return dict_models, dict_pathogenic_cluster_index

def compute_weighted_score_two_GMMs(X_pred, main_model, protein_model, 
                                    cluster_index_main, cluster_index_protein, 
                                    protein_weight):
    
    preds = protein_model.predict_proba(X_pred)[:,cluster_index_protein] \
        * protein_weight + (main_model.predict_proba(X_pred)[:,cluster_index_main]) \
        * (1 - protein_weight)
    
    return preds

def compute_weighted_class_two_GMMs(X_pred, main_model, protein_model,
                                    cluster_index_main, cluster_index_protein, 
                                    protein_weight):
    """By construct, 1 is always index of pathogenic, 0 always that of benign"""
    proba_pathogenic = protein_model.predict_proba(X_pred)[:,cluster_index_protein] \
        * protein_weight + (main_model.predict_proba(X_pred)[:,cluster_index_main]) \
        * (1 - protein_weight)
    
    return (proba_pathogenic > 0.5).astype(int)

def compute_EVE_scores(X_test, dict_models, dict_pathogenic_cluster_index, protein_GMM_weight):
    all_scores = X_test.copy()
    if protein_GMM_weight > 0.0:
        for protein in tqdm.tqdm(X_test.protein_name.unique(),"Scoring all protein mutations"):
            
            X_test_protein = X_test[X_test.protein_name==protein].evol_indices.values.reshape(-1, 1)
            mutation_scores_protein = compute_weighted_score_two_GMMs(X_pred=X_test_protein, 
                                                                            main_model = dict_models['main'], 
                                                                            protein_model=dict_models[protein], 
                                                                            cluster_index_main = dict_pathogenic_cluster_index['main'], 
                                                                            cluster_index_protein = dict_pathogenic_cluster_index[protein], 
                                                                            protein_weight = protein_GMM_weight)
            gmm_class_protein = compute_weighted_class_two_GMMs(X_pred=X_test_protein, 
                                                                            main_model = dict_models['main'], 
                                                                            protein_model=dict_models[protein], 
                                                                            cluster_index_main = dict_pathogenic_cluster_index['main'], 
                                                                            cluster_index_protein = dict_pathogenic_cluster_index[protein], 
                                                                            protein_weight = protein_GMM_weight)

            gmm_class_label_protein = pd.Series(gmm_class_protein).map(lambda x: 'Pathogenic' if x == 1 else 'Benign')
                    
            all_scores.loc[all_scores.protein_name==protein, 'EVE_scores'] = np.array(mutation_scores_protein)
            all_scores.loc[all_scores.protein_name==protein, 'EVE_classes_100_pct_retained'] = np.array(gmm_class_label_protein)
            all_scores.loc[all_scores.EVE_classes_100_pct_retained=='Benign', 'EVE_classes_100_pct_retained'] = 0.0
            all_scores.loc[all_scores.EVE_classes_100_pct_retained=='Pathogenic', 'EVE_classes_100_pct_retained'] = 1.0

    else:
        all_scores = all_evol_indices.copy()
        mutation_scores = dict_models['main'].predict_proba(np.array(all_scores['evol_indices']).reshape(-1, 1))
        all_scores['EVE_scores'] = mutation_scores[:,dict_pathogenic_cluster_index['main']]
        gmm_class = dict_models['main'].predict(np.array(all_scores['evol_indices']).reshape(-1, 1))
        all_scores['EVE_classes_100_pct_retained'] = np.array(pd.Series(gmm_class).map(lambda x: 'Pathogenic' if x == dict_pathogenic_cluster_index['main'] else 'Benign'))
        
    return all_scores