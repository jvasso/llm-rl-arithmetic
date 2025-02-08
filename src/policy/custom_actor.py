

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
from ..model_teaching_arithmetic import GPT

from .preprocess_net import PreprocessNet


class CustomActor(nn.Module):
    """Simple actor network.
    """
    
    def __init__(self,
                 preprocess_net:PreprocessNet,
                 device:str='cpu',
                 verbose:int=2):
        super().__init__()
        self.call_count = 0
        
        self.preprocess_net = preprocess_net
        self.device         = device
        self.verbose        = verbose


    def forward(self, observations_batch:np.ndarray, state: Any = None, info: Dict[str,Any] = {}):
        self.call_count += 1
        probas, hidden, state = self.preprocess_net(observations_batch=observations_batch, state=state, info=info)
        return probas, state