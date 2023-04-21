import os

MSA_data_folder='/home/scratch/wpotosna/protein_data/protein_msa'
MSA_list='./data/mappings/protein_msa_mapping.csv'
MSA_weights_location='./data/eve_msa_weights'
VAE_checkpoint_location='./EVE_results/VAE_parameters'
model_name_suffix='vae_model'
model_parameters_location='./EVE_algorithm/EVE/default_model_params.json'
training_logs_location='./EVE_results/logs/'

for i in range(103):
    os.system(f'python ./EVE_algorithm/train_VAE.py \
            --MSA_data_folder {MSA_data_folder} \
            --MSA_list {MSA_list} \
            --protein_index {i} \
            --MSA_weights_location {MSA_weights_location} \
            --VAE_checkpoint_location {VAE_checkpoint_location} \
            --model_name_suffix {model_name_suffix} \
            --model_parameters_location {model_parameters_location} \
            --training_logs_location {training_logs_location}')
