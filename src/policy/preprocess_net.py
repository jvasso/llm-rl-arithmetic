
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from typing import Dict, List
from types import SimpleNamespace

import pprint

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gymnasium import spaces

from ..llm_manager import LLMmanager
from ..custom_wandb_logger import CustomWandbLogger
from ..model_teaching_arithmetic import GPT
from .. import utils as utils


class PreprocessNet(nn.Module):

    def __init__(self,
                 llm_manager:LLMmanager,
                 is_sft_model:bool,
                 hidden_layer_idx:int=-1,
                 num_trainable_layers:int=None,
                 temperature:float=1,
                 temperature_weighted_kl:float=None,
                 delete_forbidden_logit:bool=False,
                 logger:CustomWandbLogger=None,
                 device:str='cpu',
                 verbose:int=2):
        super().__init__()
        self.call_count = 0
        
        self.llm_manager            = llm_manager
        self.hidden_layer_idx       = hidden_layer_idx
        self.num_trainable_layers   = num_trainable_layers
        self.temperature            = temperature
        self.delete_forbidden_logit = delete_forbidden_logit
        self.logger                 = logger
        self.device                 = device
        self.verbose                = verbose

        self.is_sft_model           = is_sft_model
        self.temp_weighted_kl_trick = temperature_weighted_kl
        
        self.language_model = self.llm_manager.model
        
        self.pad_token_id = self.llm_manager.pad_token_id

        self.total_num_layers = len(self.language_model.transformer.h)
        
        if self.num_trainable_layers is not None:
            self.freeze_layers(self.language_model, num_trainable_layers=self.num_trainable_layers)
        
        if isinstance(self.language_model, GPT):
            self.output_dim = self.language_model.transformer.h[-1].mlp.c_proj.out_features
        else:
            raise NotImplementedError()

    
    # step 1 | obs: 12 + 66 = <PAD> <PAD> <PAD> | action: 10
    # step 2 | obs: 12 + 66 = 10    <PAD> <PAD> | action: +

    def forward(self, observations_batch:np.ndarray, state: Any = None, info: Dict[str,Any] = {}):
        
        self.call_count += 1
        if self.verbose>=3: print(f"call foward {self.call_count}")

        if not isinstance(observations_batch, torch.Tensor):
            observations_batch = torch.tensor(observations_batch)
        observations_batch = observations_batch.to(device=self.device)
        
        attention_mask = (observations_batch != self.pad_token_id).long()
        last_non_pad_indices = attention_mask.sum(dim=1) - 1
        truncate_idx = torch.max(last_non_pad_indices).item() + 1

        observations_batch = observations_batch[:,:truncate_idx]
        
        if isinstance(self.language_model, GPT):
            output_tuple = self.language_model.custom_forward(observations_batch.to(torch.long))
            logits = output_tuple[0]
            hidden = output_tuple[1]
        else:
            attention_mask = (observations_batch != self.pad_token_id).long()
            llm_outputs = self.language_model(observations_batch.to(torch.long), attention_mask=attention_mask, output_hidden_states=False)
            logits = llm_outputs.logits # shape: (num_batchs, sentence_length, vocab_size)
        
        # self.decode_predictions(logits, observations_batch=observations_batch,sentence_idx=0)
        
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1)
        logits_target_token = logits[batch_indices, last_non_pad_indices.unsqueeze(1), :].squeeze(1)
        hidden_target_token = hidden[batch_indices, last_non_pad_indices.unsqueeze(1), :].squeeze(1)
        
        if self.delete_forbidden_logit:
            logits_target_token = logits_target_token[:, self.llm_manager.allowed_tokens_ids]
        else:
            logits_target_token[:, self.llm_manager.forbidden_token_ids] = float('-inf')
        
        probas = torch.softmax(logits_target_token/self.temperature, dim=-1)
        
        if self.is_sft_model:
            # logits_target_token = logits_target_token/self.temperature
            # logits_weighted_kl  = logits_target_token/self.temp_weighted_kl_trick
            # probas             = torch.softmax(logits_target_token/self.temperature, dim=-1)
            probas_weighted_kl = torch.softmax(logits_target_token/self.temp_weighted_kl_trick, dim=-1)
            return {'probas':probas, 'probas_weighted_kl':probas_weighted_kl}, hidden_target_token, state

        else:
            # if self.training:
            #     logits_target_token = logits_target_token/self.temperature
            
            # probas = torch.softmax(logits_target_token/self.temperature, dim=-1)
            
            # decoded_obs = self.decode_observation(observation=observations_batch[0])
            # if ('0,8,0,4,2,2' in decoded_obs) and self.training:
            #     top_k_predictions = self.get_top_k_predictions(probas=probas[0], k=5)
            #     pprint.pprint(top_k_predictions)
            #     print('ok')

            for env_idx in range(len(info)):
                log_dict_path = info['log_dict_path'][env_idx]
                if log_dict_path is not None:
                    critical_token = info['critical_token_infos']['val_token'][env_idx]
                    critical_token_proba = probas[env_idx, critical_token].detach().cpu().item()
                    
                    # debug
                    # if critical_token_proba <= 1e-9:
                    #     print("\n\nCritical token wrong. Context:\n\n"+info['context'][env_idx])
                    #     print("\nOutput probas:\n")
                    #     pprint.pprint(self.llm_manager.distribution2token_ranking(distrib=probas[env_idx, :])[:5])
                    # assert critical_token_proba > 1e-9
                    
                    utils.add_element_to_tree(tree=self.logger.critical_tokens_infos, path_str=log_dict_path, element=critical_token_proba)
            
            return probas, hidden_target_token, state
    

    

    def get_last_layer_dim_pi(self):
        if self.delete_forbidden_logit:
            return self.llm_manager.num_allowed_token
        else:
            return self.llm_manager.vocab_size
    
    
    def get_features_split_idx(self):
        return self.logits_shape

    def freeze_layers(self, model, num_trainable_layers=0):
        total_num_layers = len(model.transformer.h)
        for i, layer in enumerate(model.transformer.h):
            for param in layer.parameters():
                param.requires_grad = (i >= (total_num_layers - num_trainable_layers))
    

    def decode_observation(self, observation):
        context_token_list = []
        obs_str = ""
        for i in range(observation.shape[0]):
            context_token = self.llm_manager.tokenizer.convert_ids_to_tokens(observation[i].item())
            context_token_list.append(context_token)
            obs_str += context_token
        return obs_str

    
    def get_top_k_predictions(self, probas:torch.Tensor, k=5):
        values, indices = probas.topk(k, dim=0)
        result_dict = {}
        for i in range(values.shape[0]):
            val  = values[i]
            index = int(indices[i])
            token_str = self.llm_manager.token_id2str(token_id=index)
            result_dict[token_str] = val
        return result_dict
    
    
    def decode_predictions(self, logits, observations_batch, sentence_idx=0, sentence_start=0, sentence_end=25):
        predictions_list = []
        context_token_list = []
        for i in range(logits.shape[1]):
            context_token = self.llm_manager.tokenizer.convert_ids_to_tokens(observations_batch[sentence_idx, i].item())
            prediction    = self.llm_manager.tokenizer.convert_ids_to_tokens(logits[sentence_idx,i,:].argmax().item())
            predictions_list.append(prediction)
            context_token_list.append(context_token)
        
        max_length = max(max(len(item) for item in context_token_list), max(len(item) for item in predictions_list)) + 1
        # Prepare and print the strings
        formatted_list1 = ' '.join(f'{item:<{max_length}}' for item in context_token_list[sentence_start:sentence_end+1])
        formatted_list2 = ' '.join(f'{item:<{max_length}}' for item in predictions_list[sentence_start:sentence_end+1])

        print("CONTEXT    : ", formatted_list1)
        print("PREDICTION : ", formatted_list2)