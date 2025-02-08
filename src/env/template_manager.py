from typing import List, Union, Tuple, TYPE_CHECKING

import copy

import torch

from .result_detector import ResultDetector

if TYPE_CHECKING:
    from ..llm_manager import LLMmanager
    from .prompt_generator import PromptGenerator

from .types import Operation


class TemplateManager:

    SCRATCHPAD0 = "scratchpad0"
    SCRATCHPAD0_BEFORE_RESULT = '</scratch>\n'
    SCRATCHPAD0_AFTER_RESULT  = '\n'
    SCRATCHPAD0_SEPARATOR     = '\n'
    
    def __init__(self,
                 llm_manager:"LLMmanager",
                 template_mode:str=None,
                 instruct_templates:Union[str, List[str]]=None,
                 decompose_templates:Union[str, List[str]]=None,
                 seed=None):

        self.llm_manager = llm_manager

        if template_mode==TemplateManager.SCRATCHPAD0:
            self.template_mode = template_mode
            self.instruct_templates  = ['scratchpad0']
            self.decompose_templates = ['add_scratchpad0']
        elif template_mode is None:
            self.template_mode = None
            self.instruct_templates  = [instruct_templates] if isinstance(instruct_templates, str) else instruct_templates
            self.decompose_templates = [decompose_templates] if isinstance(decompose_templates, str) else decompose_templates
        else:
            raise NotImplementedError()
        
        
        if self.template_mode == TemplateManager.SCRATCHPAD0:
            self.text_before_result = TemplateManager.SCRATCHPAD0_BEFORE_RESULT
            self.text_after_result  = TemplateManager.SCRATCHPAD0_AFTER_RESULT
            self.examples_separator = TemplateManager.SCRATCHPAD0_SEPARATOR
        else:
            raise NotImplementedError()


        # self.generator = torch.manual_seed(seed=seed)
        self.generator = None    
    
    
    def generate_text(self, operation:Operation, example_operations:List[Operation]=None) -> Tuple[str,str,str,List[int],str]:
        input = ""
        
        if (example_operations is not None) and len(example_operations)>0:
            input += self.generate_multiple_examples(example_operations=example_operations)
            input += self.examples_separator

        instruction_str, decompose_str, result_str = self.generate_example(operation=operation, with_decompose=True)
        input += instruction_str

        expected_text_token = self.llm_manager.encode_text(decompose_str, max_text_length=None)

        return input, instruction_str, decompose_str, expected_text_token, result_str
    

    def generate_multiple_examples(self, example_operations:List[Operation]):
        full_text = ""
        for idx in range(len(example_operations)):
            operation = example_operations[idx]
            instruction_str, decompose_str, result_str = self.generate_example(operation, with_decompose=True)
            example_str = instruction_str + '\n' + decompose_str
            full_text += example_str
            if idx < len(example_operations)-1: full_text += self.examples_separator
        return full_text
    
    
    def generate_example(self, operation:Operation, with_decompose:bool):
        
        instruct_template_idx  = int(torch.randint(low=0, high=len(self.instruct_templates) , size=(1,), generator=self.generator))
        if with_decompose: decompose_template_idx = int(torch.randint(low=0, high=len(self.decompose_templates), size=(1,), generator=self.generator))
        
        instruct_template  = self.instruct_templates[instruct_template_idx]
        if with_decompose: decompose_template = self.decompose_templates[decompose_template_idx]
        
        instruct_func  = getattr(self, f"instruct_{instruct_template}")
        if with_decompose: decompose_func = getattr(self, f"decompose_{decompose_template}") if (decompose_template is not None) else None
        
        instruction_str = instruct_func(operation=operation)
        if with_decompose: decompose_str, result_str = decompose_func(operation=operation) if decompose_func is not None else None
        
        if not with_decompose: decompose_str = None
        return instruction_str, decompose_str, result_str
    

    def is_end_of_decomposition_step(self, token_str:str=None, token_int:int=None):
        return self.llm_manager.is_line_break(token_str=token_str, token_int=token_int)
    
    def get_decomposition_separator(self) -> str:
        return self.llm_manager.line_break_token_str
    
    
    ### instruct templates
    
    def instruct_math0(self, operation:Operation) -> str:
        return f"{operation.operand_one}{operation.operator.sign}{operation.operand_two}="
    
    def instruct_math1(self, operation:Operation) -> str:
        return f"{operation.operand_one}{operation.operator.sign}{operation.operand_two}=?"
    
    def instruct_scratchpad0(self, operation:Operation) -> str:
        return f'Input:\n{operation.operand_one}{operation.operator.sign}{operation.operand_two}\nTarget:'
    
    def instruct_nl0(self, operation:Operation) -> str:
        return f"Let's compute {operation.operand_one}{operation.operator.sign}{operation.operand_two} step by step: "

    
    ### decomposition templates
    
    def decompose_add_nl0(self, operation:Operation) -> str:
        
        # Convert integers to strings for easier manipulation
        num1_str = str(operation.operand_one)
        num2_str = str(operation.operand_two)
        
        # Initialize variables for carry and result
        carry = 0
        result_str = ""
        final_result = 0
        
        # Reverse the input strings for easier iteration from right to left
        num1_str = num1_str[::-1]
        num2_str = num2_str[::-1]
        
        # Iterate through the digits of the numbers
        for i in range(max(len(num1_str), len(num2_str))):
            step_str = ""

            # Get the digits at the current position, or 0 if the number is shorter
            digit1 = int(num1_str[i]) if i < len(num1_str) else 0
            digit2 = int(num2_str[i]) if i < len(num2_str) else 0
            
            # Perform addition and handle carry
            sum_digits = digit1 + digit2 + carry
            result_digit = sum_digits % 10
            previous_carry = carry
            carry = sum_digits // 10
            
            # Construct the step-by-step decomposition string
            if i>0: step_str += "\n"
            step_str += f"{digit1} + {digit2} + {previous_carry} = {sum_digits}. Write down {result_digit}, carry over {carry}."
            result_str = result_str + step_str
            
            # Update the final result
            final_result += result_digit * (10 ** i)
        
        # If there's a final carry, add it to the final result
        if carry > 0:
            final_result += carry * (10 ** (i + 1))
            result_str = result_str + f"\nWrite {carry} (final carry)."
        
        # Add the final result to the string
        encapsulated_result = self.llm_manager.result_detector.encapsulate_result(final_result)
        result_str = result_str + f"\nSo, the sum of {operation.operand_one} and {operation.operand_two} is {encapsulated_result}."

        assert final_result==operation.result
        
        return result_str
    

    def decompose_add_scratchpad0(self, operation:Operation, space_results=True, reverse_A=True) -> str:
        list1 = copy.deepcopy(Operation.operand_to_list(operation.operand_one_str))
        list2 = copy.deepcopy(Operation.operand_to_list(operation.operand_two_str))
        
        text = '\n<scratch>\n'

        list1_str = f"[{','.join(str(x) for x in list1)}]"
        list2_str = f"[{','.join(str(x) for x in list2)}]"
        # text += f'{list1_str} has {len(list1)} digits.\n'
        # text += f'{list2_str} has {len(list2)} digits.\n'
        
        A = []
        C = 0
        while len(list1)>0 or len(list2)>0:
            num1 = 0 if len(list1)==0 else list1[-1]
            num2 = 0 if len(list2)==0 else list2[-1]
            result_step = num1+num2+C
            new_A = result_step%10
            new_C = result_step//10
            list1_str = f"[{','.join(str(x) for x in list1)}]"
            list2_str = f"[{','.join(str(x) for x in list2)}]"
            A_display = reversed(A) if reverse_A else A
            A_str     = f"[{','.join(str(a) for a in A_display)}]"
            text += f'{list1_str} + {list2_str} , A={A_str} , C={C} , {num1}+{num2}+{C}={result_step} , A->{new_A} , C->{new_C}\n'
            A.append(new_A)
            C = new_C
            if len(list1)>0: list1.pop()
            if len(list2)>0: list2.pop()
        assert len(list1)==0 and len(list2)==0
        
        list1_str = f"[{','.join(str(x) for x in list1)}]"
        list2_str = f"[{','.join(str(x) for x in list2)}]"
        A_display = reversed(A) if reverse_A else A
        A_str     = f"[{','.join(str(a) for a in A_display)}]"
        text += f'{list1_str} + {list2_str} , A={A_str} C={C} , END\n' # inconsistency in the template of "Teaching arithmetics to small transformers"
        
        if C != 0: A.append(C)

        final_result_list = A[::-1] if reverse_A else A

        join_char = ' ' if space_results else ''
        final_result_str = join_char.join(str(num) for num in final_result_list)
        final_result_int = int(''.join(str(num) for num in final_result_list))
        assert final_result_int==operation.result
        
        encapsulated_result = self.llm_manager.result_detector.encapsulate_result(final_result_str)
        text += f'{TemplateManager.SCRATCHPAD0_BEFORE_RESULT}{encapsulated_result}{TemplateManager.SCRATCHPAD0_AFTER_RESULT}'
        
        return text, final_result_str
        

    def extract_result_default(self, text:str) -> str:
        start = text.find(self.text_before_result)
        if start == -1:
            return ""
        start += len(self.text_before_result)
        end = text.find(self.text_after_result, start)
        if end == -1:
            return ""
        return text[start:end]


if __name__=='__main__':
    from . import PromptGenerator, LLMmanager, OperationGenerator
    
    llm_manager      = LLMmanager()
    template_manager = TemplateManager(llm_manager=llm_manager, instruct_templates='scratchpad0', decompose_templates='add_scratchpad0')
    operation_generator = OperationGenerator(operand_min_size = 3,
                                             operand_max_size = 5,
                                             operators = "sum")
    prompt_generator = PromptGenerator(operation_generator=operation_generator, template_manager=template_manager, min_num_examples=2, max_num_examples=2)

    prompt = prompt_generator.generate_prompt()

    print(prompt)
    
