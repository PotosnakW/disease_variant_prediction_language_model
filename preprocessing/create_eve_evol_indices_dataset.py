import pandas as pd
import os

base_dir = '/xfsdata/wpotosna'
data_dir = base_dir+'/disease_variant_prediction_language_model/data'

evol_ind_files = os.listdir(base_dir+'/EVE_results/evol_indices/')
labels = pd.read_csv(data_dir+'/labels/PTEN_ClinVar_labels.csv')

complete_dataset = pd.DataFrame()
for file in evol_ind_files:
    if '.csv' not in file:
        continue
    else:
        protein_name = '_'.join(file.split('_')[:2])
        f = pd.read_csv(base_dir+f'/EVE_results/evol_indices/{file}')
        f = f[f.mutations != 'wt'].copy()
        complete_dataset = pd.concat([complete_dataset, f], axis=0)
        
complete_dataset.reset_index(inplace=True, drop=True)

complete_dataset.set_index(['protein_name', 'mutations'], inplace=True)
clinvar_labels = labels.set_index(['protein_name', 'mutations']).loc[complete_dataset.index.values]

final_dataset = pd.concat([complete_dataset, clinvar_labels], axis=1)
final_dataset.reset_index(inplace=True, drop=False)
final_dataset.set_index(['protein_name', 'mutations'], inplace=True)
final_dataset = final_dataset[~final_dataset.index.duplicated(keep='first')]
final_dataset.reset_index(inplace=True, drop=False)

final_dataset.to_csv(base_dir+'/disease_variant_prediction_language_model/data/EVE_evol_indices_dataset.csv', index=False)
