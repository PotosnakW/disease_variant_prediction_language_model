import pandas as pd
import numpy as np
import os

data_dir = '/home/scratch/wpotosna/disease_variant_prediction_language_model/data'

labels = pd.read_csv(eve_data_dir+'/labels/ClinVar_labels_P53_PTEN_RASH_SCN5A.csv')

def get_protein_seq_from_txt(file_dir):
    f = open(file_dir,'r')
    txt = f.readlines()
    name = txt[0].split('|')[2].split(' ')[0].replace('_', '-')
    seq = txt[1:]
    seq = ' '.join(seq).replace('\n', '').replace(' ', '')
    
    return {name:seq}


# CREATE SEQUENCE FILE:
sequences = {}
for file in os.listdir(data_dir+'/protein_seq'):
    if file != '.ipynb_checkpoints':
        seq = get_protein_seq_from_txt(data_dir+'/protein_seq/'+file)
        sequences[list(seq.keys())[0]] = list(seq.values())[0]
        
        
output_file = open(data_dir+'/sequences.fasta','w')

for seq_id, seq in sequences.items():
    identifier_line = ">" + seq_id + "\n"
    output_file.write(identifier_line)
    sequence_line = seq + "\n"
    output_file.write(sequence_line)
    
output_file.close()

# Show results
# input_file = open(data_dir+'/sequences.fasta')
# for line in input_file:
#     print(line.strip())


# CREATE MUTATION FILE:
mlist = []
for name, mutation in zip(labels.protein_name, labels.mutations):
    mut_str = name.replace('_', '-')+'_'+mutation
    mlist.append(mut_str)
    
with open(data_dir+'/mutation.txt', 'w') as f:
    for line in mlist:
        f.write(f"{line}\n")