import os

class PathManager:
    
    PROJECT = '.'
    RL_RESULTS   = os.path.join(PROJECT, 'rl_results')
    SAVED_MODELS = os.path.join(PROJECT, 'models')
    SLURM        = os.path.join(PROJECT, 'slurm')
    CONFIGS      = os.path.join(PROJECT, "configs")
    LOGFILES     = os.path.join(PROJECT, "logfiles")
    
    SYNC_WANDB = os.path.join(CONFIGS)
    
    SRC = os.path.join(PROJECT, "src")
    
    def __init__(self):
        pass