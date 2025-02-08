from typing import TYPE_CHECKING, List
import pprint

from .types import Operation

if TYPE_CHECKING:
    from .template_manager import TemplateManager
    


class Prompt:

    OPERAND1 = 'operand_1'
    OPERAND2 = 'operand_2'

    POS = 'pos'
    VAL_STR   = "val_str"
    VAL_TOKEN = 'val_token'
    CRITICAL_TOKENS = 'critical_tokens'
    DIGIT_DECOMPOSITION = 'digit_decomposition'
    
    def __init__(self, operation:Operation, example_operations:List[Operation], template_manager:"TemplateManager"):
        
        self.operation          = operation
        self.example_operations = example_operations
        self.template_manager   = template_manager

        self.num_examples = len(self.example_operations)
        
        self._text, self._instruction_str, self._true_decomposition_str, self._expected_text_token, self._result_template = self.template_manager.generate_text(operation=self.operation, example_operations=self.example_operations)
        # self._expected_step_by_step = self.template_manager.
        self._true_decomposition_list = self._true_decomposition_str.split(self.template_manager.get_decomposition_separator())
        if self._true_decomposition_list[0]=='':
            self._true_decomposition_list = self._true_decomposition_list[1:]
        
        self.expected_result = self.operation.result
        self.expected_result_str = str(self.expected_result)
        self._step_by_step_expected_result = self._generate_step_by_step_result()
        self.correct_patterns = ["="+self.expected_result_str, "= "+self.expected_result_str]

        self.operand2num_digits = {self.OPERAND1:self.operation.num_of_digits_one, self.OPERAND2:self.operation.num_of_digits_two}
        self.operand2gap        = {operand:num_digits - self.template_manager.llm_manager.max_confort_zone for operand,num_digits in self.operand2num_digits.items()}
    
    

    def extract_critical_tokens(self, steps_of_interest:List[int]):
        steps_of_interest = [f'step_{step_int}' for step_int in steps_of_interest]

        steps2critical_tokens = {} # {step:{idx:token}}
        for step_k in steps_of_interest:
            k = int(step_k.split('_')[1])
            decomposition_step = self._true_decomposition_list[k]
            # print(f'{step_k}: {decomposition_step}')

            steps2critical_tokens[step_k] = {self.OPERAND1:None, self.OPERAND2:None}
            
            operand2bracket_idx = {}
            for operand, num_digits in self.operand2num_digits.items():
                num_digits_in_brackets = max(num_digits-(k-1),0)
                if num_digits_in_brackets >=2: # means there will be commas
                    operand2bracket_idx[operand] = num_digits_in_brackets*2
                elif num_digits_in_brackets in {0,1}:
                    operand2bracket_idx[operand] = num_digits_in_brackets + 1
                else:
                    raise ValueError()
            
            operand2_decomposition_digit = {self.OPERAND1:decomposition_step.split(']')[0]+']',
                                            self.OPERAND2:decomposition_step[int(operand2bracket_idx[self.OPERAND1]+4):].split(']')[0]+']'}
            for operand, gap_operand in self.operand2gap.items():
                if gap_operand > 0:
                    bracket_idx = operand2bracket_idx[operand]
                    decomposition_digit = operand2_decomposition_digit[operand]
                    assert decomposition_digit[0]=='['
                    assert decomposition_digit[bracket_idx]==']'
                    steps2critical_tokens[step_k][operand] = {self.DIGIT_DECOMPOSITION:decomposition_digit, self.CRITICAL_TOKENS:{}}
                    for i in range(2*gap_operand):
                        val_str   = decomposition_digit[bracket_idx-(i+1)]
                        val_token = self.template_manager.llm_manager.token_str2id(token_str=val_str, one_token_only=True)
                        critical_token_infos = {self.POS:bracket_idx-(i+1), self.VAL_STR:val_str, self.VAL_TOKEN:val_token}
                        if val_str ==',':
                            steps2critical_tokens[step_k][operand][self.CRITICAL_TOKENS][f'n+{gap_operand-i//2}_comma'] = critical_token_infos
                        elif val_str.isdigit():
                            steps2critical_tokens[step_k][operand][self.CRITICAL_TOKENS][f'n+{gap_operand-i//2}_digit'] = critical_token_infos
                        else:
                            raise ValueError()

        return steps2critical_tokens
    

# {'step_1': {'operand_1': {'n+1_comma': {'pos': 14,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+1_digit': {'pos': 15,
#                                         'val_str': '3',
#                                         'val_token': 20},
#                           'n+2_comma': {'pos': 16,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+2_digit': {'pos': 17,
#                                         'val_str': '1',
#                                         'val_token': 18}},
#             'operand_2': {'n+1_comma': {'pos': 14,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+1_digit': {'pos': 15,
#                                         'val_str': '2',
#                                         'val_token': 19}}},
#  'step_2': {'operand_1': {'n+1_comma': {'pos': 12,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+1_digit': {'pos': 13,
#                                         'val_str': '5',
#                                         'val_token': 22},
#                           'n+2_comma': {'pos': 14,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+2_digit': {'pos': 15,
#                                         'val_str': '3',
#                                         'val_token': 20}},
#             'operand_2': {'n+1_comma': {'pos': 12,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+1_digit': {'pos': 13,
#                                         'val_str': '5',
#                                         'val_token': 22}}},
#  'step_3': {'operand_1': {'n+1_comma': {'pos': 10,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+1_digit': {'pos': 11,
#                                         'val_str': '3',
#                                         'val_token': 20},
#                           'n+2_comma': {'pos': 12,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+2_digit': {'pos': 13,
#                                         'val_str': '5',
#                                         'val_token': 22}},
#             'operand_2': {'n+1_comma': {'pos': 10,
#                                         'val_str': ',',
#                                         'val_token': 13},
#                           'n+1_digit': {'pos': 11,
#                                         'val_str': '4',
#                                         'val_token': 21}}}}

    
    def raw_arithmetic_expression(self, answer:str=None):
        arithmetic_expression = f"{self.operation.operand_one}{self.operation.operator.sign}{self.operation.operand_two}"
        if answer is not None:
            arithmetic_expression += f"={answer}"
        return arithmetic_expression
    

    def is_right_result_template(self, predicted_result:str):
        return predicted_result == self._result_template
    
    
    def is_correct_result_in_answer(self, answer:str):
        for pattern in self.correct_patterns:
            if pattern in answer:
                return True
        return False
    

    def is_correct_result(self, result:str):
        return int(result)==self.expected_result

    def __str__(self):
        return self.text
    def __repr__(self):
        return self.text
    
    
    @property
    def text(self) -> str:
        return self._text
    
    @property
    def instruction_str(self) -> str:
        return self._instruction_str
    
    @property
    def true_decomposition_str(self) -> str:
        return self._true_decomposition_str
    
    @property
    def result_template(self) -> str:
        return self._result_template
    
    @property
    def ground_truth_decomposition_list(self) -> List[str]:
        return self._true_decomposition_list
    
    @property
    def step_by_step_expected_result(self):
        return self._step_by_step_expected_result
    
    def _generate_step_by_step_result(self):
        return [ self._text + self.expected_result_str[:i] for i in range(1, len(self.expected_result_str)+1) ]
