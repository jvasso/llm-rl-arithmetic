import copy
import pprint

import torch

# from tianshou.utils.logger.wandb import WandbLogger
from tianshou.utils.logger.base import BaseLogger, VALID_LOG_VALS_TYPE, TRestoredData, DataScope
import numpy as np

from typing import Callable, Dict, Optional, Tuple, Union

import wandb

from . import utils


class CustomWandbLogger(BaseLogger):

    TRAIN  = "train"
    TEST   = "test"
    STAGES = [TRAIN, TEST]

    STEP_COUNT = "step"
    EP_COUNT   = "episode"
    
    REW = "rew"
    LOSS_METRICS  = []
    SCORE_METRICS = [REW]

    INIT_LOG_DICT = {"train":{}, "test":{} }
    
    def __init__(
        self,
        train_interval_ep : int = 1,
        test_interval_ep  : int = 1,
        update_interval_ep: int = 1,
        use_wandb:bool=True
    ):
        super().__init__()
        
        self.optimizer          = None
        self.train_interval_ep  = train_interval_ep
        self.test_interval_ep   = test_interval_ep
        self.update_interval_ep = update_interval_ep

        self.use_wandb = use_wandb
        
        self.last_log_train_ep  = 0
        self.last_log_test_ep   = 0
        self.last_log_update_ep = 0

        self.train_episode_count = 0
        self.train_step_count    = 0

        # self.current_step_log = {'episode':0, 'train_step':0}
        self.current_step_log = {CustomWandbLogger.EP_COUNT: 0, CustomWandbLogger.STEP_COUNT: 0}
        self.logs_history = []

        self.critical_tokens_infos = {}
    

    def connect_optimizer(self, optimizer:torch.optim.Optimizer):
        self.optimizer = optimizer
    
    
    def write(self, step_type: str, step: int, new_log_data, is_final_update:bool):
        """Specify how the writer is used to log data.
        
        :param str step_type: namespace which the data dict belongs to. --> "train/env_ep", "test/env_ep"
        :param int step: stands for the ordinate of the data dict.
        :param dict data: the data to write with format ``{key: value}``.
        """
        if step_type != "update/gradient_step":
            
            if step_type=='train':
                self.current_step_log = {CustomWandbLogger.EP_COUNT: self.train_episode_count, CustomWandbLogger.STEP_COUNT: self.train_step_count}
            self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, new_log_data)
            
            # if step_type=='test':
            #     assert 'episode' in self.current_step_log.keys()
            #     # self.logs_history.append(copy.deepcopy(self.current_ep_log))
            #     # pprint.pprint(self.logs_history)

            #     if not 'current_lr' in self.current_step_log:
            #         current_lr = self.optimizer.param_groups[0]['lr']
            #         self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, {'current_lr':current_lr})

            #     if is_final_update and self.use_wandb:
            #         wandb.log(copy.deepcopy(self.current_step_log))
            
            if not 'current_lr' in self.current_step_log:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, {'current_lr':current_lr})
            
            # pprint.pprint(self.critical_tokens_infos)
            if self.use_wandb:
                critical_tokens_infos_flatten = utils.flatten_dict(d=self.critical_tokens_infos)
                critical_tokens_infos_flatten = {key:float(np.mean(val)) for key,val in critical_tokens_infos_flatten.items()}
                wandb.log(copy.deepcopy({**self.current_step_log, **critical_tokens_infos_flatten}))
            
            self.critical_tokens_infos = {}
        
        elif step_type == f'{DataScope.INFO}/epoch':
            print('should not happen')
            
        else:
            self.current_step_log = CustomWandbLogger.safe_update(self.current_step_log, new_log_data)
            
            if self.use_wandb:
                wandb.log(copy.deepcopy(self.current_step_log))

    
    @staticmethod
    def extract_relevant_log_data(collect_result: dict, env_type:str):
        log_data = {
                f"{env_type}/rew_mean": collect_result['returns_stat']["mean"],
                f"{env_type}/rew_std": collect_result['returns_stat']["std"],
                f"{env_type}/length_mean": collect_result['lens_stat']["mean"],
                f"{env_type}/length_std": collect_result['lens_stat']["std"],
                f"{env_type}/num_ep": collect_result["n_collected_episodes"],
                f"{env_type}/num_steps": collect_result["n_collected_steps"],
        }
        return log_data
    
    
    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.
        
        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result['n_collected_episodes'] > 0
        assert collect_result["n_collected_steps"] > 0
        self.train_episode_count += collect_result['n_collected_episodes']
        self.train_step_count    += collect_result["n_collected_steps"]
        log_data = CustomWandbLogger.extract_relevant_log_data(collect_result=collect_result, env_type='train')
        self.write("train", step, log_data, is_final_update=False)


    def log_test_data(self, collect_result: dict, step: int, env_type:str, is_final_update:bool) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n_collected_episodes"] > 0
        # assert sum(collect_result['lens'])==collect_result['n/st']
        
        log_data = CustomWandbLogger.extract_relevant_log_data(collect_result=collect_result, env_type=env_type)
        self.write("test", step, log_data, is_final_update=is_final_update)
    
    

    @staticmethod
    def safe_update(original_dict, new_entries):
        for key in new_entries:
            if key in original_dict:
                raise KeyError(f"Key '{key}' already exists in the dictionary.")
            original_dict[key] = new_entries[key]
        return original_dict



    def prepare_dict_for_logging(self, log_data: dict) -> dict[str, VALID_LOG_VALS_TYPE]:
        """Prepare the dict for logging by filtering out invalid data types.

        If necessary, reformulate the dict to be compatible with the writer.

        :param log_data: the dict to be prepared for logging.
        :return: the prepared dict.
        """
        # print('ok')
        # raise NotImplementedError()
        pass
    

    def log_update_data(self, losses, gradient_step) -> None:
        new_log_data = losses
        if 'loss' in new_log_data.keys():
            new_log_data['loss/loss'] = new_log_data['loss']
            del new_log_data['loss']
        self.write(step_type="update/gradient_step", step=gradient_step, new_log_data=new_log_data, is_final_update=None)
    

    def log_info_data(self, log_data: dict, step: int) -> None:
        """Use writer to log global statistics.

        :param log_data: a dict containing information of data collected at the end of an epoch.
        :param step: stands for the timestep the training info is logged.
        """
        # if (
        #     step - self.last_log_info_step >= self.info_interval
        # ):  # TODO: move interval check to calling method
        #     log_data = self.prepare_dict_for_logging(log_data)
        #     self.write(f"{DataScope.INFO}/epoch", step, log_data)
        #     self.last_log_info_step = step
        pass


    
    def finalize(self) -> None:
        """Finalize the logger, e.g., close writers and connections."""
        print('ok')
        pass



    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        pass
    
    
    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        print('ok')
        raise NotImplementedError()
    

    def restore_logged_data(
        self,
        log_path: str,
    ) -> TRestoredData:
        """Load the logged data from disk for post-processing.

        :return: a dict containing the logged data.
        """
        print('ok')
        raise NotImplementedError()

