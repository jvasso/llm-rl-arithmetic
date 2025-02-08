from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import LLMmanager
    from . import Prompt



class ResultDetector:

    RESULT_BANNER_START = '<RESULT>'
    RESULT_BANNER_END   = '</RESULT>'


    def __init__(self,
                 llm_manager:"LLMmanager",
                 result_marker:str=None):
        
        self.llm_manager = llm_manager
        self.result_marker = result_marker

        if self.result_marker is None:
            self.encapsulate_result_func = getattr(self, 'encapsulate_result_with_none')
            self.detect_result_func      = getattr(self, 'check_result_with_none')
        elif self.result_marker == 'special_token':
            self.encapsulate_result_func = getattr(self, 'encapsulate_result_with_special_token')
            self.detect_result_func      = getattr(self, 'check_result_with_special_token')
        elif self.result_marker == 'banner':
            self.encapsulate_result_func = getattr(self, 'encapsulate_result_with_banner')
            self.detect_result_func      = getattr(self, 'check_result_with_banner')
        else:
            raise ValueError(f'Result marker {self.result_marker} not supported.')
    

    ### encapsulate result

    def encapsulate_result(self, result_str:str):
        return self.encapsulate_result_func(result_str)
    
    def encapsulate_result_with_none(self, result_str:str) -> str:
        return result_str
    
    def encapsulate_result_with_special_token(self, result_str:str) -> str:
        return self.llm_manager.start_result_token + result_str + result_str
    
    def encapsulate_result_with_banner(self, result_str:str) -> str:
        return self.RESULT_BANNER_START + result_str + self.RESULT_BANNER_END
    

    ### check result

    def check_result(self, prompt:"Prompt", answer:str) -> bool:
        """If a result is detected in answer, returns True or False. Otherwise, return None"""
        return self.detect_result_func(prompt, answer)
    
    def check_result_with_none(self, prompt:"Prompt", answer:str) -> bool:
        raise NotImplementedError()
        return prompt.is_correct_result_in_answer(answer=answer)
    
    def check_result_with_special_token(self, prompt:"Prompt", answer:str) -> bool:
        try:
            start_index = answer.index(self.llm_manager.start_result_token) + len(self.llm_manager.start_result_token)
            end_index   = answer.index(self.llm_manager.end_result_token, start_index)
            result = answer[start_index:end_index]
            return prompt.is_correct_result(result=result)
        except ValueError:
            return None
    
    def check_result_with_banner(self, prompt:"Prompt", answer:str) -> bool:
        try:
            start_index = answer.index(self.RESULT_BANNER_START) + len(self.RESULT_BANNER_START)
            end_index   = answer.index(self.RESULT_BANNER_END, start_index)
            result = answer[start_index:end_index]
            return prompt.is_correct_result(result=result)
        except ValueError:
            return None