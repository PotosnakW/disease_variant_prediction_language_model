import os

# Run code in same directory where 'data' folder is stored. model weights and embeddings will be stored in 'cache' and 'vespa_run_directory' folders that are created in same directory where 'data' folder is stored.

save_dir = '/home/scratch/wpotosna'
# IMPORTANT:  CHANGE 'CACHE_DIR' in '/VESPA/vespa/predict/config.py' file to same directory as 'save_dir' above

os.system(f'CUDA_VISIBLE_DEVICES=1,2,3 vespa_emb data/sequences.fasta -o {save_dir}/data/embeddings.h5  --prott5_weights_cache {save_dir}/data/cache')

os.system(f'CUDA_VISIBLE_DEVICES=1,2,3 vespa_conspred {save_dir}/data/embeddings.h5 -o {save_dir}/data/conspred.h5')

os.system(f'CUDA_VISIBLE_DEVICES=1,2,3 vespa_logodds {save_dir}/data/sequences.fasta -o {save_dir}/data/logodds.h5')

