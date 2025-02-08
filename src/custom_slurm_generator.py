import os
from typing import List

import experiments_utils as exputils

from .path_manager import PathManager



class CustomSlurmGenerator(exputils.SlurmGenerator):

    SMALL_MODELS = ["gpt2_reduced_vocab_FT_3digits", "gpt2_reduced_vocab_FT_5digits", "gpt2_reduced_vocab_FT_7digits"]
    BIG_MODELS   = ["gpt2_reduced_vocab_FT_9digits", "gpt2_reduced_vocab_FT_9digits_20k", "gpt2_reduced_vocab_FT_11digits_20k", "gpt2_reduced_vocab_FT_13digits_20k", "gpt2_reduced_vocab_FT_15digits_20k"]
    MODEL_NAMES = SMALL_MODELS + BIG_MODELS

    # mandatory attributes
    PROJECT_PATH            = PathManager.PROJECT
    CONFIGS_PATH            = PathManager.CONFIGS
    SLURM_PATH              = PathManager.SLURM
    LOGFILES_PATH           = PathManager.LOGFILES
    SYNC_WANDB_PATH         = PathManager.SYNC_WANDB
    TRAIN_FILES_FOLDER_PATH = PathManager.SRC
    CONDA_ENV_NAME = ''
    EMAIL          = ''
    
    # mandatory attributes for RUCHE
    ANACONDA_MODULE_RUCHE = ''
    CUDA_MODULE_RUCHE     = ''
    REPO_PATH_RUCHE       = ''
    CONDA_ENV_PATH_RUCHE  = ''
    
    # mandatory attributes for JEAN-ZAY
    ANACONDA_MODULE_JEAN_ZAY = 'anaconda-py3/2023.09'
    

    @staticmethod
    def adjust_config_to_constraints(config:dict, slurm_kwargs:dict, cluster_name:str):
        model_name = config['model_path'].split('/')[-1]
        assert model_name in CustomSlurmGenerator.MODEL_NAMES
        if model_name in CustomSlurmGenerator.SMALL_MODELS:
            reduce_factor = 1
        elif model_name in CustomSlurmGenerator.BIG_MODELS:
            reduce_factor = 2
        else:
            raise ValueError()

        if cluster_name == CustomSlurmGenerator.CLUSTER_JEAN_ZAY:
            constraint = slurm_kwargs['constraint']
            if '32g' in constraint:
                config["num_envs"]   = 8//reduce_factor
                config["batch_size"] = 8//reduce_factor
            elif 'a100' in constraint:
                config["num_envs"]   = 16//reduce_factor
                config["batch_size"] = 16//reduce_factor
            else:
                raise ValueError(f'Constraint {constraint} not supported.')
        
        elif cluster_name == CustomSlurmGenerator.CLUSTER_RUCHE:
            partition = slurm_kwargs['partition']
            if partition == 'gpu':
                config["num_envs"]   = 8//reduce_factor
                config["batch_size"] = 8//reduce_factor
            elif partition == 'gpua100':
                config["num_envs"]   = 8//reduce_factor
                config["batch_size"] = 8//reduce_factor
            else:
                raise ValueError(f'Partition {partition} not supported.')
        else:
            raise ValueError(f'Cluster name {cluster_name} not supported.')
        

        if config["certainty_estimation"] is None:
            config["certainty_scaling_exponent"] = 1

        return config
    

    @staticmethod
    def shorten_cluster_name(cluster_name:str):
        if cluster_name == CustomSlurmGenerator.CLUSTER_JEAN_ZAY:
            return 'JZ'
        elif cluster_name == CustomSlurmGenerator.CLUSTER_RUCHE:
            return 'R'
        else:
            raise ValueError(f'Cluster name {cluster_name} not supported.')
        