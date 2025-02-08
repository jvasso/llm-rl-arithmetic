from typing import Union, List, Tuple
import copy

import torch
import gymnasium as gym
import string

from src.env.result_detector import ResultDetector

from ..llm_manager import LLMmanager
from .action_manager import ActionManager
from .prompt_generator import PromptGenerator
from .prompt import Prompt
from .reward_manager import RewardManager
from .template_manager import TemplateManager
from .types import Operation, Operator
from .utils import check_expr


class ArithmeticEnv(gym.Env):

    metadata = {"render_modes": ["human"]}
    
    STEP_COUNT    = 0
    EP_COUNT      = 0
    EVAL_EP_COUNT = 0

    NONE_CRITICAL_TOKEN_INFOS = {Prompt.POS:None, Prompt.VAL_STR:None, Prompt.VAL_TOKEN:None}
    
    def __init__(self,
                 llm_manager:LLMmanager,
                 action_manager:ActionManager,
                 prompt_generator:PromptGenerator,
                 reward_manager:RewardManager,
                 template_manager:TemplateManager,
                 max_episode_length:Union[int,str]='standard',
                 result_banner:str=None,
                 check_decomposition_steps:bool=True,
                 interrupt_wrong_decomposition_step:bool=False,
                 env_type:str='train',
                 state_mode:str='list',
                 steps_of_interest:List[str]=None,
                 seed:int=0,
                 verbose=2):
        self.llm_manager      = llm_manager
        self.action_manager   = action_manager
        self.prompt_generator = prompt_generator
        self.reward_manager   = reward_manager
        self.template_manager = template_manager
        self.reward_manager.connect_env(env=self)
        
        self.env_type           = env_type
        self.state_mode         = state_mode
        self.steps_of_interest  = steps_of_interest

        self.seed               = seed
        self.verbose            = verbose
        
        self.window_size = self.llm_manager.window_size
        
        self.check_decomposition_steps          = check_decomposition_steps
        self.interrupt_wrong_decomposition_step = interrupt_wrong_decomposition_step
        self.result_banner                      = result_banner

        self.max_episode_length = self.compute_max_episode_length(max_episode_length)

        self.decomposition_separator = self.template_manager.get_decomposition_separator()

        self.observation_space = gym.spaces.Box(low=0, high=self.llm_manager.vocab_size, shape=(self.window_size,), dtype=int)
        self.action_space = self.action_manager.action_space

        self.step_count = 0
        self.ep_count = 0

    
    def reset(self, seed=None):
        super().reset(seed=seed)

        if 'train' in self.env_type:
            ArithmeticEnv.EP_COUNT += 1
        elif 'eval' in self.env_type:
            ArithmeticEnv.EVAL_EP_COUNT += 1
        self.ep_count += 1
        
        self.step_count = 0
        self.rewards_list = []
        self.cumul_reward = 0
        
        # self._initialize_state_and_obs(seed)
        self._initialize_state(seed)

        # infos = {'step':self.step_count, 'obs_str':self.obs_str}
        infos = {'log_dict_path':None, 'env_type':self.env_type, 'critical_token_infos':self.NONE_CRITICAL_TOKEN_INFOS}
        self.print_reset()
        return copy.deepcopy(self.obs_token), infos
    

    def step(self, action:Union[str,int]):
        self.step_count += 1
        if 'train' in self.env_type:
            ArithmeticEnv.STEP_COUNT += 1
        
        action_token, action_str = self.action_manager.decode_action(action)
        
        # is_update_successful = self._maybe_update_state_and_obs(action_str, action_token)
        log_dict_path, critical_token_infos = self._update_state(action_str=action_str, action_token=action_token)
        terminated, cause  = self.is_terminated(action_token, action_str)
        reward             = self.reward_manager.compute_reward(action_token, action_str, terminated)

        self.rewards_list.append(reward)
        self.cumul_reward += reward
        
        self.preprocess_next_step()

        infos = {'log_dict_path':log_dict_path, 'env_type':self.env_type, 'critical_token_infos':critical_token_infos, 'context':self.full_text_str}
        self.print_step(action_token, action_str, terminated, reward, cause)
        return copy.deepcopy(self.obs_token), reward, terminated, False, infos
    

    def preprocess_next_step(self):
        if self.is_end_of_decomposition_step:
            self.decomposition_step_idx += 1
        if not self.has_result_started:
            if self.template_manager.text_before_result in self.answer_str:
                self.has_result_started = True
    

    def _initialize_state(self, seed):
        self.prompt = self.prompt_generator.generate_prompt()
        self.instruction_str = self.prompt.instruction_str
        self.full_text_str   = self.prompt.text
        self.full_text_token, first_pad_idx = self.llm_manager.encode_text(self.full_text_str, max_text_length=self.window_size)
        self.current_idx = first_pad_idx-1
        self.current_ep_max_length = min(self.max_episode_length, len(self.full_text_token)-self.current_idx)
        assert self.current_ep_max_length > 0
        self.answer_str = ''
        self.answer_token = []
        self.arithmetic_answer = ''

        self.true_decomposition_str = self.prompt.true_decomposition_str
        self.ground_truth_decomposition_list = self.prompt.ground_truth_decomposition_list
        self.decomposition_step_idx = 0

        self.has_result_started   = False
        self.found_correct_result = False
        
        max_window_size_exceeded = self.current_idx+1 > self.window_size
        self._update_obs(max_window_size_exceeded=max_window_size_exceeded)

        if self.steps_of_interest is not None:
            assert isinstance(self.steps_of_interest,list)
            self.steps2critical_tokens = self.prompt.extract_critical_tokens(steps_of_interest=self.steps_of_interest)

            self.reinitialize_operand_decomposition()
    

    def reinitialize_operand_decomposition(self):
        self.decomposition_steps_count = -1
        self.in_operand_decomposition  = None
        self.operand_decomposition_next_idx = None
        self.left_brackets_count2operand = {1:self.prompt.OPERAND1, 2:self.prompt.OPERAND2}
        self.left_brackets_count = 0
        self.is_operand_decomposition_correct = {self.prompt.OPERAND1: True, self.prompt.OPERAND2:True}


    def _update_state(self, action_str:str, action_token:int):

        if self.state_mode=='string':
            # self.full_text_str += action_str
            # self.answer_str    += action_str
            # self.arithmetic_answer += action_str
            # self.answer_token.append(action_token)
            # self.full_text_token, first_pad_idx = self.maybe_encode(text=self.full_text_str)
            # max_length_reached = first_pad_idx >= self.window_size
            raise NotImplementedError()
        
        elif self.state_mode=='list':
            self.current_idx += 1

            max_window_size_exceeded = self.current_idx+1 > self.window_size

            if max_window_size_exceeded:
                self.full_text_token.append(action_token)
            else:
                self.full_text_token[self.current_idx] = action_token
            
            self.answer_token.append(action_token)

            self.full_text_str     += action_str
            self.answer_str        += action_str
            self.arithmetic_answer += action_str
            
            self.max_episode_length_reached = self.is_max_episode_length_reached()
            
            log_dict_path        = None
            critical_token_infos = self.NONE_CRITICAL_TOKEN_INFOS
            if self.check_decomposition_steps:
                
                self.is_end_of_decomposition_step = self.template_manager.is_end_of_decomposition_step(token_int=action_token)
                decomposition_steps = self.answer_str.split(self.decomposition_separator)
                current_decomposition_step = decomposition_steps[-2] if decomposition_steps[-1]=='' else decomposition_steps[-1]

                if self.is_end_of_decomposition_step:
                    self.is_right_decomposition_step = self.check_right_decomposition_step(current_decomposition_step)
                else:
                    self.is_right_decomposition_step = None
                
                if self.decomposition_steps_count in self.steps_of_interest:
                    step_k = f'step_{self.decomposition_steps_count}'

                    if action_str=='[':
                        self.left_brackets_count += 1

                        if self.left_brackets_count in self.left_brackets_count2operand.keys():
                            self.in_operand_decomposition  = self.left_brackets_count2operand[self.left_brackets_count]
                            self.operand_decomposition_next_idx = 0
                        elif self.left_brackets_count==2:
                            self.in_operand_decomposition  = self.prompt.OPERAND2
                            self.operand_decomposition_next_idx = 0
                    
                    if self.in_operand_decomposition:
                        if self.prompt.operand2gap[self.in_operand_decomposition] > 0:
                            
                            if self.is_operand_decomposition_correct[self.in_operand_decomposition]:
                                gt_digit_decomposition = self.steps2critical_tokens[step_k][self.in_operand_decomposition][self.prompt.DIGIT_DECOMPOSITION][:self.operand_decomposition_next_idx+1]
                                current_digit_decomposition = '[' + self.answer_str.split('[')[-1]
                                self.is_operand_decomposition_correct[self.in_operand_decomposition] = gt_digit_decomposition==current_digit_decomposition

                                if self.is_operand_decomposition_correct[self.in_operand_decomposition]:
                                    self.operand_decomposition_next_idx += 1
                                    step_k = f'step_{self.decomposition_steps_count}'
                                    critical_tokens = self.steps2critical_tokens[step_k][self.in_operand_decomposition][self.prompt.CRITICAL_TOKENS]
                                    for critical_token_name, critical_token_infos in critical_tokens.items():
                                        pos = critical_token_infos[self.prompt.POS]
                                        if self.operand_decomposition_next_idx == pos:
                                            log_dict_path = f'{self.env_type}/{step_k}/{self.in_operand_decomposition}/{critical_token_name}'
                                            break
                
                if self.is_end_of_decomposition_step:
                    self.decomposition_steps_count += 1
                    self.left_brackets_count = 0
                
                if action_str==']':
                    self.in_operand_decomposition = None
                

                    
            # assert self.full_text_str == self.llm_manager.tokenizer.decode(self.full_text_token)
        else:
            raise ValueError(f'State mode {self.state_mode} not supported.')
        
        if self.has_result_started and self.is_end_of_decomposition_step:
            self.found_correct_result = self.check_result()
        
        self._update_obs(max_window_size_exceeded)

        return log_dict_path, critical_token_infos
    
    
    def check_right_decomposition_step(self, decomposition_step:str):
        if self.decomposition_step_idx > len(self.ground_truth_decomposition_list)-1:
            return False
        else:
            return decomposition_step == self.ground_truth_decomposition_list[self.decomposition_step_idx]
    
    
    def maybe_encode(self, text) -> Tuple[torch.Tensor, int]:
        encoded_text, first_pad_idx = self.llm_manager.encode_text(text, max_text_length=self.window_size)
        return encoded_text, first_pad_idx
    
    
    def _update_obs(self, max_window_size_exceeded:bool):
        self.obs_str   = self.full_text_str
        if max_window_size_exceeded:
            self.obs_token = self.full_text_token[-self.window_size:]
        else:
            self.obs_token = self.full_text_token
    
    
    def is_terminated(self, action_token:int, action_str:str):
        if self.interrupt_wrong_decomposition_step and self.is_end_of_decomposition_step and not self.is_right_decomposition_step:
            return True, 'wrong decomposition step'
        if self.has_result_started and self.is_end_of_decomposition_step:
            return True, 'ended result step'
        if self.llm_manager.is_eos(action_token):
            return True, "eos action"
        if self.max_episode_length_reached:
            return True, "max episode length"
        # if self.found_correct_result:                       return True, "found correct result"
        if self.llm_manager.is_padding_token(action_token):
            return True, "predicted padding token"
        return False, None
    
    
    def check_result(self):
        if self.llm_manager.result_detector.result_marker is not None:
            raise NotImplementedError()
            self.llm_manager.result_detector.check_result(prompt=self.prompt, answer=self.answer_str)
        else:
            predicted_result = self.template_manager.extract_result_default(text=self.answer_str)
            return self.prompt.is_right_result_template(predicted_result=predicted_result)
    

    def compute_max_episode_length(self, max_episode_length):
        if isinstance(max_episode_length, int):
            return max_episode_length
        elif max_episode_length=='standard':
            operand_max_size = self.prompt_generator.operation_generator.operand_max_size
            max_operand = int("9" * operand_max_size)
            operator = Operator(id=self.prompt_generator.operation_generator.operators_ids[0])
            operation = Operation(operand_one=max_operand, operand_two=max_operand, operator=operator)
            prompt = Prompt(operation=operation, example_operations=[], template_manager=self.template_manager)
            decomposition_str = prompt._true_decomposition_str
            max_episode_length = len(decomposition_str) + 100
            print(f'env {self.env_type} -> max episode length:', max_episode_length)
            return max_episode_length
        else:
            raise ValueError()


    
    @staticmethod
    def RESET_CLASS_DATA():
        ArithmeticEnv.STEP_COUNT    = 0
        ArithmeticEnv.EP_COUNT      = 0
        ArithmeticEnv.EVAL_EP_COUNT = 0

    
    def render(self):
        print(f"\nPrompt: {self.prompt.text}")
        print(f"Answer: {self.answer_str}")
    

    def is_max_episode_length_reached(self):
        return self.step_count >= self.max_episode_length
    

    # def max_text_length_reached(self, observation):
    #     if bool(observation[0,-1] != self.padding_token_id):
    #          return True
    #     return False
    def max_text_length_reached(self, observation:List[int]):
        last_token_id = observation[-1]
        if last_token_id == self.llm_manager.pad_token_id:
            return False
        return True
    
    def get_answer(self):
        return self.answer_str
    
    def get_full_arithmetic_expression(self):
        expression = self.prompt.raw_arithmetic_expression(answer=self.arithmetic_answer)
        return expression
    

    def print_reset(self):
        # if self.verbose >= 4:
        #     print(f"s_0: {self.prompt.prompt_str}")
        #     print(f"o_0: {self.obs_str}")
        if self.verbose >= 4:
            print(f"\nReset - STEP {ArithmeticEnv.STEP_COUNT} ({self.env_type}) | EPISODE: {ArithmeticEnv.EP_COUNT}")
            print(f"Prompt: {self.prompt.text}")
    
    
    def print_step(self, action_token, action_str, terminated, reward, cause):
        # if self.verbose >= 4:
        #     print(f"a_{self.step_count-1}: {action_str}")
        #     print(f"r_{self.step_count-1}: {reward}")
        #     print("-----------")
        #     print(f"s_{self.step_count}: {self.get_full_arithmetic_expression()}")
        #     print(f"o_{self.step_count}: {self.obs_str}")
        #     if terminated: print(f"Terminated ({cause})")
        #     if terminated: print(f"Cumul reward: {self.cumul_reward}")
        if self.verbose >= 4:
            text = f'Step: {self.step_count:>2} || Rew: {round(reward, ndigits=2):>4} || Act: {repr(action_str):>15} (token: {action_token:>5}) || State: {repr(self.get_full_arithmetic_expression())}'
            print(text)
            if terminated: print(f"Terminated. Cumul Rew: {self.cumul_reward}")
        elif self.verbose >= 3:
            if terminated:
                print(f"\n• {self.env_type} - Ep {ArithmeticEnv.EP_COUNT} (STEP: {ArithmeticEnv.STEP_COUNT}) | CUMUL R:{self.cumul_reward:>4} | LENGTH: {self.step_count} | TERMINATED CAUSE: {cause} | OBS:\n")
                examples_text = f'[{self.prompt.num_examples} examples]\n' if self.prompt.num_examples > 0 else ''
                print(f'({self.prompt.operation.num_of_digits_one}+{self.prompt.operation.num_of_digits_two} digits)')
                print(examples_text+self.instruction_str + self.answer_str)
        elif self.verbose >= 2:
            if terminated:
                print(f"• {self.env_type} - Ep {ArithmeticEnv.EP_COUNT} (STEP: {ArithmeticEnv.STEP_COUNT}) | CUMUL R:{self.cumul_reward:>4} | LENGTH: {self.step_count} | OBS: {repr(self.obs_str)}")