import os
from typing import List

import copy
from datetime import datetime
import json
import yaml

import random
import numpy as np
import torch

import itertools

import inspect

# def merge_subdicts(dicts_list:List[dict], key:str):
#     assert isinstance(dicts_list, list) and all( isinstance(d, dict) or isinstance(d.__dict__, dict) for d in dicts_list)
#     assert isinstance(key, str)
#     accepted_dicts = [ current_dict for current_dict in dicts_list if (key in current_dict)]
#     merged_dict = {}
#     for current_dict in accepted_dicts:
#         assert isinstance(current_dict[key], dict), "The entry '"+key+"' should be a dictionary."
#         merged_dict.update(copy.deepcopy(current_dict[key]))
#     return merged_dict


def save_dict_as_json(data_dict, file_path, create_folders=False, indent=4):
    """
    Saves a dictionary as a JSON file at the specified file path.
    Creates intermediate directories if they do not exist.

    Args:
    data_dict (dict): The dictionary to be saved.
    file_path (str): The path where the JSON file will be saved.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if (not os.path.exists(directory)) and create_folders:
        os.makedirs(directory)
    try:
        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=indent)
    except Exception as e:
        print(f"Error saving file: {e}")


def extract_arg_names(function):
    argspec = inspect.getfullargspec(function)
    return argspec.args


def set_all_seeds(seed:int, device='cpu'):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # torch.backends.cudnn.enabled       = False
    elif str(device)=='cpu':
        pass
    else:
        raise ValueError(f'Device {device} not supported.')


def set_reproducible_experiment(seed, detect_anomaly=False, device='cpu'):
    torch.use_deterministic_algorithms(mode=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    set_all_seeds(seed=seed, device=device)


def current_datetime():
    return datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%S")


def to_sweep_format(parameters:dict):
    sweep_parameters = {}
    for key,param in parameters.items():
        if isinstance(param, dict):
            raise Exception(f'Dict param not supported')
        elif isinstance(param,list):
            assert len(param) > 0
            if len(param) == 1:
                sweep_parameters[key] = {"value":param[0]}
            else: 
                sweep_parameters[key] = {"values":param}
        else:
            sweep_parameters[key] = {"value":param}
    return sweep_parameters


def remove_lists(dictionary:dict) -> dict:
    for key,val in dictionary.items():
        if isinstance(val, list):
            dictionary[key] = val[0]
    return dictionary


def adjust_for_quick_test(config:dict):
    config['step_per_epoch']   = 1
    config['step_per_collect'] = 2048
    config['episode_per_test'] = 4
    config['max_epoch']        = 3
    config['batch_size']       = 4
    return config



def leaves_to_lists(dictionary:dict):
    for key,val in dictionary.items():
        if not isinstance(val,list):
            dictionary[key] = [val]
    return dictionary


def dict_of_lists2list_of_dicts(input_dict):
    input_dict = leaves_to_lists(input_dict)
    # Extract keys and corresponding lists of values
    keys = input_dict.keys()
    values = input_dict.values()
    
    # Use itertools.product to compute the Cartesian product of lists
    product = itertools.product(*values)
    
    # Create a list of dictionaries, each representing a unique combination
    result = [dict(zip(keys, combo)) for combo in product]
    
    return result


def load_yaml_file(file_path:str):
    if not '.yaml' in file_path: file_path += '.yaml'
    with open(file_path, 'r') as file:
        data_dict = yaml.safe_load(file)
    return data_dict



def delete_files_in_directory(directory:str):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            else:
                print(f"Skipped {file_path} because it is a directory or a special file.")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")



import torch
import torch.nn as nn

def print_cuda_memory(label, tensor):
    print(f"{label} - Memory (MB): {tensor.element_size() * tensor.nelement() / 1024 ** 2:.2f}")

def estimate_memory_usage(model, input_size, dtype=torch.float32, device='cuda', optimizer=None):
    # Move the model to the specified device
    model = model.to(device)
    
    # Create a dummy input based on input_size and dtype
    input = torch.randn(*input_size, dtype=dtype, device=device)
    print_cuda_memory("Input", input)

    # Forward pass to calculate activations and memory used
    output = model(input)

    # Calculate total parameters and memory
    total_params = sum(p.numel() for p in model.parameters())
    element_size = torch.tensor([], dtype=dtype).element_size()
    params_memory = total_params * element_size
    print("Total parameters:", total_params/1000000, "M.")
    print("Memory for parameters (MB):", params_memory / 1024 ** 2)

    # Gradient memory (same as parameter memory)
    gradient_memory = params_memory  # Assuming all parameters require gradients
    print("Memory for gradients (MB):", gradient_memory / 1024 ** 2)

    if optimizer is not None:
        # Example for Adam: It keeps momentums and variances for each parameter
        optimizer = optimizer(model.parameters())
        optimizer_state_memory = sum((2 * p.numel() * element_size) for p in model.parameters())
        print("Memory for optimizer states (MB):", optimizer_state_memory / 1024 ** 2)

    # Calculating activation memory by registering a forward hook
    activation_memories = []

    def hook(module, input, output):
        memory = output.element_size() * output.nelement()
        activation_memories.append(memory)

    # Register hook for all layers
    handles = []
    for layer in model.modules():
        if isinstance(layer, (nn.ReLU, nn.Linear)):  # Can adjust based on layers used in the model
            handle = layer.register_forward_hook(hook)
            handles.append(handle)

    # Re-run forward pass to trigger hooks and calculate activation memory
    output = model(input)
    total_activation_memory = sum(activation_memories)
    print("Total activation memory (MB):", total_activation_memory / 1024 ** 2)

    # Clean up hooks
    for handle in handles:
        handle.remove()

# # Example usage
# model = nn.Sequential(
#     nn.Linear(4000, 4000),
#     nn.ReLU(),
#     nn.Linear(4000, 4000),
#     nn.ReLU(),
#     nn.Linear(4000, 4000),
#     nn.ReLU(),
#     nn.Linear(4000, 4000),
#     nn.ReLU(),
#     nn.Linear(4000, 4000),
#     nn.ReLU(),
#     nn.Linear(4000, 10)
# )

# estimate_memory_usage(model, (8, 4000), optimizer=torch.optim.Adam)



def extract_hidden_sizes_tuple(config):
    if not hasattr(config, 'hidden_size_critic1'):
        return ()
    elif config.hidden_size_critic1 is None:
        return ()
    else:
        if not hasattr(config, 'hidden_size_critic2'):
            return (config.hidden_size_critic1,)
        elif config.hidden_size_critic2 is None:
            return (config.hidden_size_critic1,)
        else:
            return (config.hidden_size_critic1, config.hidden_size_critic2)



def add_element_to_tree(tree:dict, path_str:str, element):
    """
    Create a path in a dictionary as a nested structure, and add an element to the list at the leaf.

    Parameters:
    d (dict): The dictionary to modify.
    path_str (str): A string indicating the path, with keys separated by '/'.
    element: The element to append to the list at the final leaf.
    """
    # Split the path string by '/' to get individual keys
    path = path_str.split('/')

    current = tree

    # Traverse the path and create dictionaries if necessary
    for key in path[:-1]:  # Traverse all keys except the last one
        if key not in current:
            current[key] = {}
        current = current[key]

    # Create the final leaf as an empty list if it does not exist
    if path[-1] not in current:
        current[path[-1]] = []

    # Append the element to the list at the final leaf
    current[path[-1]].append(element)



def flatten_dict(d, parent_key='', sep='/'):
    """
    Flattens a nested dictionary with arbitrary depth into a flat dictionary.
    
    Parameters:
    d (dict): The dictionary to flatten.
    parent_key (str): The base key used in recursive calls to track the key path.
    sep (str): The separator to use for joining keys.
    
    Returns:
    dict: A flattened dictionary with path-like keys.
    """
    flat_dict = {}
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat_dict[new_key] = v

    return flat_dict
