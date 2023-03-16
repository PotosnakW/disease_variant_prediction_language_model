import pandas as pd
import numpy as np
import os
import re

data_dir = '/home/scratch/wpotosna/data'

labels = pd.read_csv(data_dir+'/labels/PTEN_ClinVar_labels.csv')

def get_protein_seq_from_txt(file_dir, labels):
    f = open(file_dir,'r')
    txt = f.readlines()
    protein_name = txt[0].split('|')[2].split(' ')[0]
    seq = txt[1:]
    seq = ' '.join(seq).replace('\n', '').replace(' ', '')
    
    checked_seq, valid_mutations = check_sequence(protein_name, seq, labels)
    
    return protein_name.replace('_', '-'), checked_seq, valid_mutations
    
def check_sequence(protein_name, seq, labels):
    protein_mutations = labels[labels.protein_name==protein_name].mutations.values
    
    valid_mutations = []
    for mutation in protein_mutations:
        pos_idx = int(re.sub("[^0-9]","", mutation))-1
        orig_aa = mutation[0]

        if pos_idx > len(seq):
            continue
            
        elif seq[pos_idx] != orig_aa:
            continue
        else:
            valid_mutations.append(protein_name.replace('_', '-')+'_'+mutation) 
    
    if len(valid_mutations) >= 2:
        checked_seq = seq
    else:
        checked_seq = None

    return checked_seq, valid_mutations

# CREATE sequence.fasta and mutations.txt files:
sequences = {}
mutations = []
for file in os.listdir(data_dir+'/wildtype_protein_sequences'):
    if file != '.ipynb_checkpoints':
        pn, cs, vm = get_protein_seq_from_txt(data_dir+'/wildtype_protein_sequences/'+file, labels)
        if cs is not None:
            sequences[pn] = cs
            mutations.extend(vm)
            
output_file = open(data_dir+'/sequences.fasta','w')

for seq_id, seq in sequences.items():
    identifier_line = ">" + seq_id + "\n"
    output_file.write(identifier_line)
    sequence_line = seq + "\n"
    output_file.write(sequence_line)
output_file.close()

with open(data_dir+'/mutations.txt', 'w') as f:
    for line in mutations:
        f.write(f"{line}\n")
