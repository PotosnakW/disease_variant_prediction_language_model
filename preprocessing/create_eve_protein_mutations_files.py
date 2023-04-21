import pandas as pd
import numpy as np

base_dir = '/home/scratch/wpotosna'

proteins = pd.read_csv(base_dir+'/disease_variant_prediction_language_model/data/mappings/protein_msa_mapping.csv')
labels = pd.read_csv(base_dir+'/disease_variant_prediction_language_model/data/labels/PTEN_ClinVar_labels.csv')

protein_names = proteins.protein_name.values
for protein_name in np.unique(protein_names):
    mutations = pd.DataFrame(labels[labels.protein_name == protein_name].mutations)
    mutations.to_csv(base_dir+f'/disease_variant_prediction_language_model/data/eve_mutations/{protein_name}.csv', index=False)
    