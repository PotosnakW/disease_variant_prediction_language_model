import os

MSA_data_folder='/xfsdata/wpotosna/protein_data/protein_msa'
MSA_list='./data/mappings/protein_msa_mapping.csv'
MSA_weights_location='./data/eve_msa_weights'
VAE_checkpoint_location='/xfsdata/wpotosna/EVE_results/VAE_parameters'
model_name_suffix='vae_model'
model_parameters_location='./EVE_algorithm/EVE/default_model_params.json'
training_logs_location='/xfsdata/wpotosna/EVE_results/logs/'

computation_mode='input_mutations_list'
mutations_location='./data/eve_mutations'
output_evol_indices_location='/xfsdata/wpotosna/EVE_results/evol_indices'
num_samples_compute_evol_indices=20000
batch_size=256

for i in range(74, 80):
    print(i)
    print('Training VAE Model')
    os.system(f'python ./EVE_algorithm/train_VAE.py \
            --MSA_data_folder {MSA_data_folder} \
            --MSA_list {MSA_list} \
            --protein_index {i} \
            --MSA_weights_location {MSA_weights_location} \
            --VAE_checkpoint_location {VAE_checkpoint_location} \
            --model_name_suffix {model_name_suffix} \
            --model_parameters_location {model_parameters_location} \
            --training_logs_location {training_logs_location}')

    print('Computing Evolutationary Indices')
    os.system(f'python ./EVE_algorithm/compute_evol_indices.py \
        --MSA_data_folder {MSA_data_folder} \
        --MSA_list {MSA_list} \
        --protein_index {i} \
        --MSA_weights_location {MSA_weights_location} \
        --VAE_checkpoint_location {VAE_checkpoint_location} \
        --model_name_suffix {model_name_suffix} \
        --model_parameters_location {model_parameters_location} \
        --computation_mode {computation_mode} \
        --mutations_location {mutations_location} \
        --output_evol_indices_location {output_evol_indices_location} \
        --num_samples_compute_evol_indices {num_samples_compute_evol_indices} \
        --batch_size {batch_size}')
    
print('Complete')

#protein ind issues: #58, 65, 70, 73