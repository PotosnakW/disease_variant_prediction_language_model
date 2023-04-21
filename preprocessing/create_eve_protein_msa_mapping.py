import pandas as pd
import os

base_dir = '/home/scratch/wpotosna'
protein_msa_files = os.listdir(base_dir+'/protein_data/protein_msa')
protein_names = [('_').join(file.split('_')[:2]) for file in protein_msa_files]

df = pd.DataFrame([protein_names, protein_msa_files]).T
df.columns = ['protein_name', 'msa_location']
df['theta'] = 0.2

df.to_csv(base_dir+'/disease_variant_prediction_language_model/data/protein_msa_mapping.csv', index=False)