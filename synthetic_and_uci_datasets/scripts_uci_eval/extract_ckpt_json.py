import os
import json
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Define the base directory
base_dir = os.path.join(os.getenv("PROJECT_ROOT"),'synthetic_and_uci_datasets','checkpoints_uci_amcl')
save_dir = os.path.join(os.getenv("PROJECT_ROOT"),'synthetic_and_uci_datasets','scripts_uci_eval')

# Define the datasets and methods
datasets = ['bostonHousing', 'concrete', 'energy', 'kin8nm', 'naval', 'power_plant', 'protein', 'wine', 'yacht', 'year']
methods = {
    'amcl': ['scheduler_linear', 'scheduler_exponential'],
    'relaxed_wta': ['epsilon_0.5', 'epsilon_0.1', 'epsilon_0.05', 'epsilon_0.3', 'epsilon_0.2', 'epsilon_annealed'],
    'vanilla_mcl': ['vwta']
}

# Initialize the JSON dictionary
json_data = {}

for method_name in os.listdir(base_dir):
    for dataset in os.listdir(os.path.join(base_dir, method_name)):
        for dir_name in os.listdir(os.path.join(base_dir, method_name, dataset)):
            for param in methods[method_name]:
                if param in ['epsilon_0.05', 'epsilon_0.3', 'epsilon_0.2'] and dataset not in ['protein', 'year']:
                    # the runs with epsilon={0.05, 0.3, 0.2} are only computed for the datasets 'protein' and 'year'
                    continue

                if 'epsilon_annealed' in dir_name and 'epsilon_annealed' not in param:
                    continue

                if param in dir_name:
                    checkpoint_path = os.path.join(base_dir, method_name, dataset, dir_name, 'checkpoints', 'best.ckpt')
                    if os.path.exists(checkpoint_path):
                        split_number = dir_name.split('split-')[1].split('_')[0]
                        key = f"{dataset}_{method_name}_{param}_split_{split_number}"
                        json_data[key] = checkpoint_path
                    else:
                        print(checkpoint_path)
                        split_number = dir_name.split('split-')[1].split('_')[0]
                        print(f"Checkpoint not found for {dataset}_{method_name}_{param}_split_{split_number}")
        # Print the number of keys in the json_data dictionary for this dataset
        keys = [e for e in json_data.keys() if dataset in e]
        print(f"Number of keys for {dataset}: {len(keys)}")

# Write the JSON data to a file
with open(os.path.join(save_dir, 'checkpoints_paths.json'), 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("JSON file created successfully.")