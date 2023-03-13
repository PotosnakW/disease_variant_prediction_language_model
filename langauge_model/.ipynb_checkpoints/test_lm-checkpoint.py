import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc
import pickle

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

sequences_Example = ["A E T C Z A O","S K T Z P"]

sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    
embedding = embedding.last_hidden_state.cpu().numpy()

features = [] #[L, 1024] where L is the length of each sequence
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][:seq_len-1]
    features.append(seq_emd)
    
    
# if you want to derive a single representation (per-protein embedding) for the whole protein
# protein_embedding = embedding.mean(dim=0) # shape (1024)
    
with open("test", "wb") as fp:
    pickle.dump(features, fp)