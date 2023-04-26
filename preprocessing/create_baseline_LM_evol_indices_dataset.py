import pandas as p

base_dir = '/xfsdata/wpotosna'
data_dir = base_dir+'/disease_variant_prediction_language_model/data'
logodds_dir = base_dir+'/protein_data'

mapping_file = pd.read_csv(data_dir+'/labels/ClinVar_labels_P53_PTEN_RASH_SCN5A.csv',low_memory=False)
protein_list = [i.replace('.txt', '') for i in os.listdir(data_dir+'/wildtype_protein_sequences/') if (i != 'ACHA_HUMAN')& (i!='.ipynb_checkpoints')]

labels = pd.read_csv(data_dir+'/labels/PTEN_ClinVar_labels.csv')
logodds = pd.read_csv(logodds_dir+'/single_logodds.csv', sep=';')

protein_names = ['_'.join(i.split('_')[:-1]) for i in logodds.SeqID_SAV.values]
mutation_names = [i.split('_')[-1] for i in logodds.SeqID_SAV.values]

df = pd.DataFrame(data=[protein_names, mutation_names, logodds.SAV_score.values]).T
df.columns = ['protein_name', 'mutations', 'evol_indices']
df.set_index(['protein_name', 'mutations'], inplace=True)

clinvar_labels = labels.set_index(['protein_name', 'mutations']).loc[df.index.values]

dataset = pd.concat([df, clinvar_labels], axis=1)
dataset.reset_index(inplace=True, drop=False)
dataset.drop(index=dataset[dataset.protein_name=='ALAT2_HUMAN'].index.values[0], inplace=True)

# EVE computes negative log ratio: -(log(Xv) - log(Xw))
dataset.evol_indices = dataset.evol_indices*(-1)
dataset.sort_values(by='protein_name', inplace=True)
dataset.reset_index(inplace=True)

dataset.to_csv(base_dir+'/disease_variant_prediction_langauge_model/data/Baseline_LM_evol_indices_dataset.csv', index=False)