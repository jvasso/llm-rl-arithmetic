from typing import Union, List, Tuple

import torch
from torch.distributions import Bernoulli
import numpy as np

from .types import Operator, Operation

from ..llm_manager import LLMmanager


class OperationGenerator:

    operators_id2sign = {"sum":"+", "sub":"-", "mul":"*"}
    
    def __init__(self,
                 llm_manager:LLMmanager,
                 operand_min_size:int,
                 operand_max_size:int,
                 env_type:str,
                 operators:Union[str, List[str]]="sum",
                 confort_zone_prop:float=None,
                 seed=None):

        self.llm_manager      = llm_manager
        self.env_type         = env_type

        self.min_confort_zone = self.llm_manager.min_confort_zone
        self.max_confort_zone = self.llm_manager.max_confort_zone

        self.operand_min_size, self.operand_max_size = self.preprocess_operand_size(operand_min_size=operand_min_size,
                                                                                    operand_max_size=operand_max_size)

        self.plus_mode = False
        
        if self.env_type in {'train','eval'}:
            self.confort_zone_prop = confort_zone_prop
        else:
            if 'confort' in self.env_type:
                self.confort_zone_prop = 1.
                self.operand_min_size = self.min_confort_zone
                self.operand_max_size = self.max_confort_zone
            elif 'plus' in self.env_type:
                self.confort_zone_prop = 0.
                num = int(self.env_type.split('plus')[1])
                self.operand_max_size = self.max_confort_zone + num
                self.plus_mode = True
            else:
                raise ValueError(f'Env type {self.env_type} not supported.')
        
        self.operators_ids = [operators] if isinstance(operators, str) else operators
        
        # self.random_generator = torch.manual_seed(seed=seed)
        self.random_generator = None
    
    
    def preprocess_operand_size(self, operand_min_size:Union[str,int], operand_max_size:Union[str,int]):
        if isinstance(operand_min_size, int) or isinstance(operand_max_size, int):
            assert isinstance(operand_min_size, int) and isinstance(operand_max_size, int)
            operand_min_size_int, operand_max_size_int = operand_min_size, operand_max_size
        else:
            assert isinstance(operand_min_size, str) and isinstance(operand_max_size, str)
            operand_min_size_int = self.max_confort_zone + int(operand_min_size.split('plus')[1])
            operand_max_size_int = self.max_confort_zone + int(operand_max_size.split('plus')[1])

        assert operand_min_size_int <= operand_max_size_int
        return operand_min_size_int, operand_max_size_int
    
    
    def generate_operations(self, num=1, seed=None) -> List[Operation]:
        operands_list  = self.generate_operands_couples(num=2*num, seed=seed)
        operators_list = self.generate_operator(num=num, seed=seed)
        operations_list = [ Operation(operand_one=operands_list[i][0], operand_two=operands_list[i][1], operator=operators_list[i]) for i in range(num) ]
        return operations_list
    
    
    def generate_operands_couples(self, num=1, seed=None) -> List[Tuple[int,int]]:
        if num==0:
            return []
        
        # generator = self.random_generator if (seed is None) else torch.manual_seed(seed)
        generator = None

        if self.confort_zone_prop is None:
            num_digits_list1 = torch.randint(low=self.operand_min_size, high=self.operand_max_size+1, size=(num,), generator=generator).tolist()
            num_digits_list2 = torch.randint(low=self.operand_min_size, high=self.operand_max_size+1, size=(num,), generator=generator).tolist()
        else:
            bernoulli_dist = Bernoulli(torch.tensor([float(self.confort_zone_prop)]))
            stay_in_confort_zone_list = bernoulli_dist.sample((int(num),)).squeeze().int().tolist()
            difficult_is_operand1_list = Bernoulli(torch.tensor([0.5])).sample((len(stay_in_confort_zone_list)-sum(stay_in_confort_zone_list),)).squeeze().int().tolist()
            difficult_is_operand1_list = [difficult_is_operand1_list] if not isinstance(difficult_is_operand1_list,list) else difficult_is_operand1_list
            
            num_digits_list1 = []
            num_digits_list2 = []
            idx = 0
            for stay_in_confort_zone in stay_in_confort_zone_list:
                if stay_in_confort_zone:
                    low = self.min_confort_zone # changed here (update)
                    high = min(self.max_confort_zone, self.operand_max_size)
                    num_digits_list1.append(torch.randint(low=low, high=high+1, size=(1,), generator=generator).item())
                    num_digits_list2.append(torch.randint(low=low, high=high+1, size=(1,), generator=generator).item())
                else:
                    if self.plus_mode:
                        difficult_number = self.operand_max_size
                    else:
                        difficult_number = torch.randint(low=self.operand_min_size, high=self.operand_max_size+1, size=(1,), generator=generator).item()
                    
                    # assert self.max_confort_zone+1 == self.operand_max_size

                    # the "standard" number is half of the time sampled from the confort zone, and half of the time sampled from the "inconfort" zone
                    is_standard_number_in_confort_zone = torch.rand(1).item() < 0.5
                    if is_standard_number_in_confort_zone:
                        standard_number = torch.randint(low=self.min_confort_zone, high=self.max_confort_zone+1, size=(1,), generator=generator).item()
                    else:
                        standard_number = torch.randint(low=self.max_confort_zone+1, high=self.operand_max_size+1, size=(1,), generator=generator).item()
                    
                    if difficult_is_operand1_list[idx]:
                        num_digits_list1.append(difficult_number)
                        num_digits_list2.append(standard_number)
                    else:
                        num_digits_list2.append(difficult_number)
                        num_digits_list1.append(standard_number)
                    
                    idx += 1
        
        operands = []
        for i in range(len(num_digits_list1)):
            num_digits1 = num_digits_list1[i]
            num_digits2 = num_digits_list2[i]
            lowest_int1  = 0 if num_digits1==1 else 10**(num_digits1-1)
            highest_int1 = 9 if num_digits1==1 else 10**(num_digits1)-1
            lowest_int2  = 0 if num_digits2==1 else 10**(num_digits2-1)
            highest_int2 = 9 if num_digits2==1 else 10**(num_digits2)-1
            operand1 = torch.randint(low=lowest_int1, high=highest_int1+1, size=(1,), generator=generator).item()
            operand2 = torch.randint(low=lowest_int2, high=highest_int2+1, size=(1,), generator=generator).item()
            operands.append((operand1, operand2))
        
        return operands
    

    def generate_operator(self, num=1, seed=None) -> List[Operator]:
        # generator = self.random_generator if (seed is None) else torch.manual_seed(seed)
        generator = None
        operators_idx_list = torch.randint(low=0, high=len(self.operators_ids), size=(num,), generator=generator).tolist()
        operators_ids_list = [ self.operators_ids[idx]  for idx in operators_idx_list]
        operators_list = [ Operator(id=operator_id) for operator_id in operators_ids_list]
        return operators_list



if __name__=='__main__':

    from ..llm_manager import LLMmanager
    from . import OperationGenerator

    num = 10

    operand_min_size = 1
    operand_max_size = 5

    confort_zone_prop = 0.1

    env_type = 'eval_plus2'
    
    llm_manager = LLMmanager(model_name='gpt2_special')
    operation_generator = OperationGenerator(llm_manager=llm_manager,
                                             operand_min_size=operand_min_size,
                                             operand_max_size = operand_max_size,
                                             env_type=env_type,
                                             operators = "sum",
                                             confort_zone_prop=confort_zone_prop)
    operations = operation_generator.generate_operations(num=num)
    for op in operations:
        print(op)