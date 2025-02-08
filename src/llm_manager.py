from typing import List, Tuple, Union
import re
import warnings

import torch

from requests import get
from transformers import AutoTokenizer, GPT2Tokenizer, LlamaTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoConfig
from transformers import LlamaForCausalLM

from transformers import GPT2LMHeadModel, GPT2Model
from transformers import LlamaTokenizer, LlamaPreTrainedModel

from .model_teaching_arithmetic import GPT
from .env.result_detector import ResultDetector
from .path_manager import PathManager


class LLMmanager:
    
    LLAMA_WEIGHTS_PATH  = "./pretrained_llms/"
    LLAMA2_WEIGHTS_PATH = "./pretrained_llms/llama2"

    START_RESULT_TOKEN = "<RESULT>"
    END_RESULT_TOKEN   = "</RESULT>"

    DEFAULT_PAD_TOKEN_STR = '<PAD>'
    DEFAULT_PAD_TOKEN_ID  = -1

    ALL_TOKEN_MODE = 'all'
    ARITHMETIC_TOKEN_MODE = 'arithmetic'
    
    def __init__(self,
                 model_name:str='gpt2',
                 model_path:str=None,
                 result_marker:str=None,
                 special_token_mode:str=None,
                 pad_token:str=None,
                 allowed_token_mode:str='all'):
        
        self.model_name         = model_name
        self.model_path         = model_path
        self.special_token_mode = special_token_mode
        self.pad_token          = pad_token
        self.allowed_token_mode = allowed_token_mode

        self.human_mode         = model_name == 'human'
        
        self._load_tokenizer_and_model()
        self._set_model_pretrain_info()
        self._set_vocab_vars()
        self._set_eos_token()
        self._set_pad_token()
        self._set_forbidden_tokens()
        # self._set_special_tokens(result_marker)

        self.result_detector = ResultDetector(llm_manager=self, result_marker=result_marker)
    


    def _set_vocab_vars(self):
        if self.human_mode:
            human_vocab = self.build_human_vocab(eos_str=self._eos_str, padding_token_str=self._pad_token_str)
            self.vocab_size = len(human_vocab)
            self._token_str2id = { human_vocab[i]:i for i in range(len(human_vocab)) }
            self._eos_token_id = self._token_str2id["."]
            self._eos_str = "."
        else:
            self.vocab_size = len(self.tokenizer)
            self._token_str2id = self.tokenizer.get_vocab()
            self._eos_token_id = self.tokenizer.eos_token_id
            self._eos_str = self.tokenizer.decode(self._eos_token_id) if self._eos_token_id is not None else None
        
        self._token_id2str = {id:token for token,id in self._token_str2id.items()}

        self.all_tokens_str = list(self._token_str2id.keys())
        self.all_tokens_ids = list(self._token_id2str.keys())
        self.all_tokens_ids.sort()
    

    def _set_eos_token(self):
        if self.human_mode:
            self._eos_token_id = self._token_str2id["."]
            self._eos_str = "."
        else:
            if self.tokenizer.eos_token_id is not None:
                self._eos_token_id = self.tokenizer.eos_token_id
                self._eos_str = self.tokenizer.decode(self._eos_token_id)
                self.has_eos = True
            else:
                self.has_eos = False
    
    
    def _set_pad_token(self):
        
        if 'lecraquito' in self.model_path:
                assert '%' in self.tokenizer.get_vocab()
                self.tokenizer.pad_token = '%'
        
        if self.pad_token is None:
            self.has_pad_token = False
        elif self.pad_token=='default':
            if self.human_mode:
                self._pad_token_str = LLMmanager.DEFAULT_PAD_TOKEN_STR
                self._pad_token_id  = LLMmanager.DEFAULT_PAD_TOKEN_ID
            else:
                assert self.tokenizer.pad_token_id is not None
                self._pad_token_str = self.tokenizer.pad_token
                self._pad_token_id  = self.tokenizer.pad_token_id
                assert self._pad_token_id==self.token_str2id(self._pad_token_str)
            self.has_pad_token = True
        elif self.pad_token=='eos':
            assert self.tokenizer.eos_token_id is not None
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self._pad_token_str = self.tokenizer.pad_token
            self._pad_token_id  = self.token_str2id(self._pad_token_str)
            self.has_pad_token = True
        else:
            raise NotImplementedError()
                
    
    
    # def _set_special_tokens(self, result_marker) -> dict:
    #     self.special_tokens_str_list = []
    #     self.special_tokens_dict = {}
        
    #     if result_marker=='special_token':
    #         self.start_result_token, self.end_result_token = LLMmanager.START_RESULT_TOKEN, LLMmanager.END_RESULT_TOKEN
    #         result_tokens = [self.start_result_token, self.end_result_token]
    #         self.special_tokens_str_list += result_tokens
    #         self.special_tokens_dict.update({'additional_special_tokens':result_tokens})
    #     else:
    #         self.start_result_token, self.end_result_token = None, None

    #     if self.special_token_mode is not None:
    #         raise NotImplementedError()
    #         special_tokens_dict = {'pad_token': LLMmanager.DEFAULT_PAD_TOKEN_STR}
    #         additional_special_tokens = [self.START_RESULT_TOKEN, self.END_RESULT_TOKEN]
    #         self.special_tokens_str_list += additional_special_tokens
    #         special_tokens_dict.update({'additional_special_tokens':additional_special_tokens})
        
    #     if len(self.special_tokens_dict) > 0 and not self.human_mode:
    #         self.tokenizer.add_special_tokens(special_tokens_dict=self.special_tokens_dict, replace_additional_special_tokens=False)
    #         self.model.resize_token_embeddings(len(self.tokenizer))

    
    def _set_forbidden_tokens(self):
        allowed_tokens_str, allowed_tokens_ids = self.get_allowed_token()
        self.allowed_tokens_str = [token for _, token in sorted(zip(allowed_tokens_ids, allowed_tokens_str))]
        self.allowed_tokens_ids = sorted(allowed_tokens_ids)
        
        self.forbidden_token_ids = [token_id for token_id in self.all_tokens_ids if token_id not in self.allowed_tokens_ids]
        self.forbidden_token_str = [ self._token_id2str[token_id] for token_id in self.forbidden_token_ids]

        self._num_allowed_token = len(self.allowed_tokens_ids)
    

    def get_allowed_token(self) -> Tuple[List[str], List[int]]:
        if self.allowed_token_mode == LLMmanager.ALL_TOKEN_MODE:
            return self.get_all_token()
        elif self.allowed_token_mode == LLMmanager.ARITHMETIC_TOKEN_MODE:
            return self.get_arithmetic_token()
        else:
            raise Exception(f'Token mode {self.allowed_token_mode} not supported.')
    
    
    def get_all_token(self) -> Tuple[List[str], List[int]]:
        return self.all_tokens_str, self.all_tokens_ids


    def get_arithmetic_token(self) -> Tuple[List[str], List[int]]:
        # Find tokens relevant to arithmetic
        vocab = self.tokenizer.get_vocab()
        operators_list = self.operators_list()
        extra = [' ']
        arithmetic_tokens_str = []
        arithmetic_tokens_ids = []
        for token, token_id in vocab.items():
            # One or more digits, possibly separated by a dot for decimal numbers
            # if token.isdigit() or (token.replace('.', '', 1).isdigit() and '.' in token):
            if token in {"0","1","2","3","4","5","6","7","8","9"} or (token in operators_list) or (token in self.special_tokens_str_list) or (token in extra):
                arithmetic_tokens_str.append(token)
                arithmetic_tokens_ids.append(token_id)
        return arithmetic_tokens_str, arithmetic_tokens_ids
    

    def build_human_vocab(self, eos_str, padding_token_str):
        digits = [str(digit) for digit in list(range(10))]
        operators = ["=","+","-"]
        eos = [eos_str]
        pad = [padding_token_str]
        human_vocab = digits+operators+eos+pad+self.special_tokens_str_list
        return human_vocab


    def _load_tokenizer_and_model(self):
        print('Loading tokenizer and model...')
        if self.model_name=='gpt2':
            self._load_gpt2()
        elif self.model_name=='gpt2_special':
            self._load_gpt2_special()
        elif 'llama' in self.model_name:
            self._load_llama()
        else:
            raise Exception(f"Model name {self.model_name} not supported.")
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        print(f'Finished loading: num params = {self.num_model_parameters/1e6:.2f}M.')

        # self.model = self.model.half()
    

    def _set_model_pretrain_info(self):
        if 'lecraquito/gpt2_reduced_vocab_FT_' in self.model_path:
            digit_part = self.model_path.split('lecraquito/gpt2_reduced_vocab_FT_')[1] # gpt2_reduced_vocab_FT_7digits or gpt2_reduced_vocab_FT_11digits_20k
            num_digit_str = digit_part.split('digits')[0] # 7digits or 11digits_20k
            min_num_digit = 1
            max_num_digit = int(num_digit_str)
        else:
            raise NotImplementedError()
        
        self.min_confort_zone = min_num_digit
        self.max_confort_zone = max_num_digit

    
    def _load_gpt2(self):
        from transformers import GPT2Config
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.model     = GPT2LMHeadModel.from_pretrained('gpt2')
        self.config = GPT2Config.from_pretrained('gpt2')
        self.window_size = self.config.max_position_embeddings
        raise NotImplementedError() # implement line break!
    
    def _load_gpt2_special(self):
        if 'lecraquito' in self.model_path:
            self.tokenizer = AutoTokenizer.from_pretrained('lecraquito/gpt2_reduced_vocab_FT_3digits')
            self.model     = GPT.from_pretrained(self.model_path)
            self.config    = AutoConfig.from_pretrained('lecraquito/gpt2_reduced_vocab_FT_3digits')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.config = AutoConfig.from_pretrained(self.model_path)
        
        # remove dropout
        if isinstance(self.model, GPT):
            self.model = self.remove_gpt_dropout(model=self.model)
        
        self.line_break_token_id = self.tokenizer.encode('\n')[-1]
        self.line_break_token_str = '\n'
        assert self.tokenizer.decode(self.line_break_token_id)=='\n'

        self.window_size = self.config.max_position_embeddings

    
    def remove_gpt_dropout(self, model:GPT):
        for name, module in self.model.transformer.items():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
            elif isinstance(module, torch.nn.ModuleList):
                for block in module:
                    block.attn.attn_dropout.p  = 0
                    block.attn.resid_dropout.p = 0
                    block.mlp.dropout.p = 0
        return model
    

    def distribution2token_ranking(self, distrib:torch.Tensor):
        token_str2proba = {}
        for token_idx in range(distrib.shape[0]):
            token_str = self.token_id2str(token_idx)
            token_str2proba[token_str] = distrib[token_idx].item()
        ranking = sorted(token_str2proba.items(), key=lambda item: item[1], reverse=True)
        return ranking


        



    def _load_llama(self):
        if self.model_path is None:
            self.model_path = LLMmanager.LLAMA2_WEIGHTS_PATH if "llama2" in self.model_name else LLMmanager.LLAMA_WEIGHTS_PATH
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.model = LlamaForCausalLM.from_pretrained(self.model_path)
    
    
    def save_model(self, local_path:str=None, hugging_face:str=None):
        if local_path is not None:
            self.model.save_pretrained(local_path)
        if hugging_face is not None:
            self.model.push_to_hub(hugging_face)
    
    
    @staticmethod
    def operators_list():
        # return ['+', '-', '*', '/', '(', ')', '=', '>', '<', '>=', '<=', '!=', '^', '**', '%', 'sqrt', 'log', 'ln', 'exp', 'pi', 'e', '!', '[', ']', '{', '}']
        return ['+', '=']


    def is_eos(self, token_id:int):
        if not self.has_eos:
            return False
        else:
            return token_id==self._eos_token_id
    
    def is_padding_token(self, token_id:int):
        return token_id == self._pad_token_id
    
    def token_id2str(self, token_id:int):
        # return self.tokenizer.decode(token_id)
        # self.tokenizer.convert_tokens_to_ids()
        # print(self.tokenizer.convert_ids_to_tokens(token_id), repr(self.tokenizer.decode(token_id)))
        
        if isinstance(self.model, GPT):
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
        else:
            token_str = self.tokenizer.convert_ids_to_tokens(token_id) if token_id != 13 else '\n'
            token_str = token_str.replace('▁', ' ')
        return token_str

    def token_str2id(self, token_str:str, one_token_only:bool=False):
        if isinstance(self.model, GPT):
            id = self.tokenizer.convert_tokens_to_ids(token_str)
            if one_token_only:
                if isinstance(id,int):
                    return id
                else:
                    print('ok')
                    print(type(id))
            else:
                return id
        else:
            if token_str=='\n':
                return 13
            else:
                return self.tokenizer.convert_tokens_to_ids(token_str)

    

    def decode_obs(self, observation:Union[List[int],torch.Tensor]) -> str:
        if torch.is_tensor(observation):
            if len(observation.shape)==2:
                assert observation.shape[0] == 1
                observation = observation.squeeze()
            observation = observation.tolist()
        decoded_text = self.tokenizer.decode(observation)
        return decoded_text

    
    def encode_text(self, str_sequence:str, max_text_length:int=None) -> Tuple[List[int], bool]:
        if self.human_mode:
            assert max_text_length is not None
            token_seq = self.encode_without_tokenizer(str_sequence, max_text_length=max_text_length)
        else:
            token_seq = self.tokenizer.encode(str_sequence, return_tensors=None, padding='max_length', max_length=max_text_length)
        # if len(full_text_token) < self.max_text_length:
        #     self.obs_token = full_text_token + [-1]*(self.max_text_length-len(full_text_token))
        # is_size_ok = len(token_seq) <= max_text_length
        first_pad_idx = self.find_first_pad_token_idx(token_seq=token_seq)
        return token_seq, first_pad_idx
    

    def find_first_pad_token_idx(self, token_seq:List[int]):
        for idx in range(len(token_seq)):
            if token_seq[idx]==self._pad_token_id:
                return idx
        return -1

    
    def encode_without_tokenizer(self, str_sequence:str, max_text_length:int):
        tokens = []
        i = 0
        while i < len(str_sequence):
            matched = False
            for token in self.special_tokens_str_list:
                if str_sequence[i:].startswith(token):
                    tokens.append(self._token_str2id[token])  # Add the token id to the result
                    i += len(token)  # Skip the length of the token
                    matched = True
                    break
            if not matched:
                if str_sequence[i] in self._token_str2id:
                    tokens.append(self._token_str2id[str_sequence[i]])
                    i += 1
                else:
                    raise ValueError(f"Unknown character '{str_sequence[i]}' in input string.")
        if len(tokens) == max_text_length:
            warnings.warn(f"The string sequence {str_sequence} exceeds by 1 the max sentence length of the model.")
        elif len(tokens) > max_text_length:
            raise Exception(f"The string sequence {str_sequence} exceeds by more than 1 the max sentence length of the model.")
        else:
            remaining_size = max_text_length-len(tokens)
            tokens += [self._pad_token_id]*remaining_size
        return tokens
    

    def is_line_break(self, token_str:str=None, token_int:int=None):
        if token_int is not None:
            return token_int==self.line_break_token_id
        assert token_str is not None
        return token_str=='\n' or token_str=='<0x0A>'


    @property
    def hidden_state_shape(self):
        if isinstance(self.model, GPT2LMHeadModel):
            return self.model.base_model.embed_dim
        else:
            raise NotImplementedError()
    
    @property
    def num_allowed_token(self):
        return self._num_allowed_token
    @property
    def eos_token_id(self):
        return self._eos_token_id
    @property
    def eos_str(self):
        return self._eos_str
    @property
    def pad_token_id(self):
        return self._pad_token_id