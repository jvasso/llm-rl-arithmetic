from typing import List, TYPE_CHECKING

import re

from .prompt import Prompt
from . import utils
if TYPE_CHECKING:
    from .arithmetic_env import ArithmeticEnv


class RewardManager:

    SUCCESS_REWARD = 10 # the reward you get when you
    CORRECT_REWARD = 1  # the reward you get any time you write something correct

    INCORRECT_REWARD = -1

    MATH_SYMBOLS = set("+-*=/")
    
    def __init__(self,
                 reward_mode:str,
                 decomposition_step_bonus:float=None,
                 math_symbol_bonus:float=None,
                 correctness_bonus:float=None,
                 decomposition_step_penalty:float=None,
                 wrong_result_penalty:float=None,
                 time_penalty:float=None,
                 padding_penalty:float=None,
                 max_ep_length_penalty:float=None,
                 env_type:str='train'):
        
        self.reward_mode                = reward_mode
        self.decomposition_step_bonus   = decomposition_step_bonus
        self.math_symbol_bonus          = math_symbol_bonus
        self.correctness_bonus          = correctness_bonus
        
        self.time_penalty               = - abs(time_penalty)
        self.padding_penalty            = - abs(padding_penalty)
        self.max_ep_length_penalty      = - abs(max_ep_length_penalty)
        self.decomposition_step_penalty = - abs(decomposition_step_penalty)
        self.wrong_result_penalty       = - abs(wrong_result_penalty)

        self.env_type = env_type
        
        # if 'eval' in self.env_type:
        #     self.reward_mode = 'binary'
    

    def connect_env(self, env:"ArithmeticEnv"):
        self.env = env
    
    def compute_reward(self, action_token:int, action_str:str, terminated:bool, **args):
        # if self.env.llm_manager.is_padding_token(action_token):
        #     return RewardManager.PADDING_TOKEN_REWARD
        if self.env.llm_manager.special_token_mode is None:
            return self._compute_reward_without_special_token(action_token=action_token, action_str=action_str, terminated=terminated, **args)
        else:
            return self._compute_reward_with_special_token(action_token=action_token, action_str=action_str, terminated=terminated, **args)



    def _compute_reward_without_special_token(self, **args):
        if self.reward_mode=='binary':
            return self.binary_reward(**args)
        elif self.reward_mode=='ternary':
            return self.ternary_reward(**args)
        elif self.reward_mode=='composition':
            return self.composition_reward(**args)
        elif self.reward_mode=='step_by_step':
            return self.step_by_step_reward(**args)
        elif self.reward_mode=='correctness':
            return self.correctness_reward(**args)
        else:
            raise ValueError(f"Reward mode '{self.reward_mode}' not supported.")
    

    @staticmethod
    def is_math_symbol(char:str):
        if utils.is_number(char):
            return True
        if char in RewardManager.MATH_SYMBOLS:
            return True
        return False
    
    
    def is_correct_expression(self):
        prediction = self.env.get_full_arithmetic_expression()
        return utils.check_expr(prediction)
    

    def completed_decomposition_step_successfully(self):
        if self.env.is_end_of_decomposition_step and self.env.is_right_decomposition_step:
            return True
        else:
            return False


    # "composition"
    def composition_reward(self, action_str:str, action_token:int, terminated:bool) -> float:
        reward = 0
        if self.time_penalty is not None:
            reward += self.time_penalty
        
        if (self.padding_penalty is not None):
            if self.env.llm_manager.is_padding_token(token_id=action_token):
                reward += self.padding_penalty
        
        if self.math_symbol_bonus is not None:
            if RewardManager.is_math_symbol(char=action_str):
                reward += self.math_symbol_bonus
        
        if self.correctness_bonus is not None:
            if self.is_correct_expression():
                reward += self.correctness_bonus
        
        if self.decomposition_step_bonus is not None:
            if self.env.is_end_of_decomposition_step and self.env.is_right_decomposition_step:
                reward += self.decomposition_step_bonus

        if self.decomposition_step_penalty is not None:
            if self.env.is_end_of_decomposition_step and not self.env.is_right_decomposition_step:
                reward += self.decomposition_step_penalty

        if self.max_ep_length_penalty is not None:
            if self.env.max_episode_length_reached:
                reward += self.max_ep_length_penalty
        
        return reward
        
    
    # "binary"
    def binary_reward(self, action_str:str, action_token:int, terminated:bool):
        if not terminated: return 0
        if self.env.found_correct_result:
            return 1
        return 0
    
    # "ternary"
    def ternary_reward(self, action_str:str, action_token:int, terminated:bool):
        if not terminated: return 0
        if self.env.found_correct_result:
            return 1
        return self.wrong_result_penalty
    

    # "step-by-step"
    def step_by_step_reward(self, action_str:str, action_token:int, terminated:bool, expect_eos=True):
        reward = 0
        expected_results_list = self.env.prompt.step_by_step_expected_result
        if expect_eos: expected_results_list += [expected_results_list[-1]+self.env.llm_manager.eos_str]
        prediction = self.env.get_full_arithmetic_expression()
        if prediction in expected_results_list:
            reward += RewardManager.CORRECT_REWARD
            if terminated:
                if prediction == expected_results_list[-1]: reward += RewardManager.SUCCESS_REWARD
        else:
            reward += RewardManager.INCORRECT_REWARD
        return reward
    

    # "correct"
    def correctness_reward(self, action_str:str, action_token:int, terminated:bool):
        reward = 0
        prediction = self.env.get_full_arithmetic_expression()
        correct = utils.check_expr(prediction)
        if correct:
            reward += RewardManager.CORRECT_REWARD
            if terminated: reward += RewardManager.SUCCESS_REWARD
        else:
            reward += RewardManager.INCORRECT_REWARD
        return reward
    
    
    def _compute_reward_with_special_token(self, action_str:str, action_token:int, terminated:bool):
        raise NotImplementedError()
        answer = self.remove_prompt(self.state_str)
        if not terminated: return 0
        if not self.finished_result(answer): return 0
        start_index = answer.index(self.START_RESULT)
        end_index   = answer.index(self.END_RESULT)
        if start_index != -1 and end_index != -1:
            result_str = answer[start_index+1:end_index].strip()
        else:
            raise Exception("What happend?")
        if not result_str.isdigit(): return 0
        result_int = int(result_str)
        if result_int == self.prompt.expected_result: return 1
        return 0
    



    