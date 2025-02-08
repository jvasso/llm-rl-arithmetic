from typing import Tuple, List, Union

import numpy as np
import gymnasium as gym

from ..llm_manager import LLMmanager


class ActionManager:

    def __init__(self,
                 llm_manager:LLMmanager=None,
                 delete_forbidden_logit:bool=True,
                 human_mode:bool=False):
        self.llm_manager = llm_manager
        self.delete_forbidden_logit = delete_forbidden_logit
        self.human_mode = human_mode

        self._build_action2token()
        self._set_action_space()
    

    def _build_action2token(self):
        if self.human_mode:
            self._action2token_id = self.llm_manager._token_str2id
        elif self.delete_forbidden_logit:
            self._action2token_id = { i:self.llm_manager.allowed_tokens_ids[i] for i in range(len(self.llm_manager.allowed_tokens_ids))}
        else:
            self._action2token_id = { idx:idx for idx in self.llm_manager.all_tokens_ids }
    
    
    def _set_action_space(self):
        if self.delete_forbidden_logit:
            self.action_space = gym.spaces.Discrete(self.llm_manager.num_allowed_token)
        else:
            self.action_space = gym.spaces.Discrete(self.llm_manager.vocab_size)
    
    
    def decode_action(self, action:Union[str,int]) -> Tuple[int,str]:
        if not (action in self._action2token_id):
            raise Exception(f"Action {action} not supported.\nAction dictionary: {self._action2token_id}")
        if self.human_mode:
            assert isinstance(action,str)
            token_id  = self.convert_human_action2token(action)
        else:
            assert isinstance(action, int) or np.issubdtype(type(action), np.integer)
            action = int(action)
            token_id = self.convert_action2token(action)
        token_str = self.llm_manager.token_id2str(token_id)
        return token_id, token_str
    

    def convert_action2token(self, action:int):
        if self.delete_forbidden_logit:
            return self._action2token_id[action]
        return action
    

    def convert_human_action2token(self, action_str:str):
        assert isinstance(action_str, str)
        action_token = self.llm_manager._token_str2id[action_str]
        return action_token
    

    def decode_human_action(self, action_str:str):
        assert isinstance(action_str, str)
        if action_str.isdigit():
            action_token = int(action_str)
        else:
            action_token = self.user_char2id(action_str)
        return action_token, action_str


    def user_char2id(self, char:str):
        char2id = {".":10, "=":11, "+":12, "-":13}
        return char2id[char]
    
    
    
