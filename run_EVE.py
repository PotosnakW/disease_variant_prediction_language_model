import os

MSA_data_folder='./EVE/EVE_data/MSA'
MSA_list='./EVE/data/mappings/example_mapping.csv'
MSA_weights_location='./EVE/data/weights'
VAE_checkpoint_location='./EVE/results/VAE_parameters'
model_name_suffix='Jan1_PTEN_example'
model_parameters_location='./EVE/EVE/default_model_params.json'
training_logs_location='./EVE/logs/'
protein_index=0

os.system(f'python ./EVE/train_VAE.py \
        --MSA_data_folder {MSA_data_folder} \
        --MSA_list {MSA_list} \
        --protein_index {protein_index} \
        --MSA_weights_location {MSA_weights_location} \
        --VAE_checkpoint_location {VAE_checkpoint_location} \
        --model_name_suffix {model_name_suffix} \
        --model_parameters_location {model_parameters_location} \
        --training_logs_location {training_logs_location}')