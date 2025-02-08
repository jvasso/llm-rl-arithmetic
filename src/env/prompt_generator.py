from typing import Union, List, Callable
import warnings
import torch

from ..llm_manager import LLMmanager

from .prompt import Prompt

from .template_manager import TemplateManager
from .operation_generator import OperationGenerator

from .types import Operator


TemplateFunc = Callable[[int,int,],str]


class PromptGenerator:
    
    operators_id2sign = {"sum":"+", "sub":"-", "mul":"*"}

    WARNING_RAISED = False
    
    def __init__(self,
                 operation_generator:OperationGenerator,
                 template_manager:TemplateManager,
                 min_num_examples:int,
                 max_num_examples:int):
        
        self.operation_generator = operation_generator
        self.template_manager    = template_manager
        self.min_num_examples = min_num_examples
        self.max_num_examples = max_num_examples
    
    
    def generate_prompt(self) -> Prompt:
        # if (seed is not None) and not PromptGenerator.WARNING_RAISED:
        #     PromptGenerator.WARNING_RAISED = True
        #     warnings.warn(f'\n\n!!! WARNING: A seed has been passed as an input to generate_prompt (risk of redundancy in the numbers generated) !!!\n\n')

        target_operation   = self.operation_generator.generate_operations(num=1, seed=None)[0]
        
        num_examples = self.generate_num_examples()
        example_operations = self.operation_generator.generate_operations(num=num_examples, seed=None)

        prompt = Prompt(operation=target_operation,
                        example_operations=example_operations,
                        template_manager=self.template_manager)
        return prompt
    

    def generate_num_examples(self) -> int:
        num_examples_tensor_list = torch.randint(low=self.min_num_examples, high=self.max_num_examples+1, size=(1,)).tolist()
        return num_examples_tensor_list[0]



if __name__=='__main__':

    from ..utils import set_reproducible_experiment

    num_env      = 2
    num_generate = 2
    num_examples = 2

    min_num_examples = 2
    max_num_examples = 3

    set_reproducible_experiment(seed=0)

    llm_manager = LLMmanager()

    for idx_env in range(num_env):
        print(f'\n\nEnv {idx_env+1}')
        operation_generator = OperationGenerator(operand_min_size=1, operand_max_size=2, operators='sum')
        template_manager    = TemplateManager(llm_manager=llm_manager, instruct_templates='nl0', decompose_templates='add_nl0')
        prompt_generator    = PromptGenerator(operation_generator=operation_generator, template_manager=template_manager, min_num_examples=min_num_examples, max_num_examples=max_num_examples)

        for idx_generate in range(num_generate):
            print(f'\n\nPrompt {idx_generate+1}')
            prompt = prompt_generator.generate_prompt()
            print(prompt)
