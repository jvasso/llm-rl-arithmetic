from typing import List


class Operator:

    ID2FUNC = {"sum":int.__add__, "sub":int.__sub__, "mul":int.__mul__}
    ID2SIGN = {"sum":"+", "sub":"-", "mul":"*"}
    
    def __init__(self, id:str):
        self.id = id
        self.func = Operator.ID2FUNC[self.id]
        self.sign = Operator.ID2SIGN[self.id]
    
    def compute(self, operand_one:int, operand_two:int):
        return self.func(operand_one, operand_two)
    
    def __str__(self):
        return self.sign
    def __repr__(self):
        return self.sign



class Operation:
    
    def __init__(self, operand_one:int, operand_two:int, operator:Operator):
        self.operand_one = operand_one
        self.operand_two = operand_two
        self.operator    = operator

        self.operand_one_str = str(self.operand_one)
        self.operand_two_str = str(self.operand_two)

        self.num_of_digits_one = len(str(abs(self.operand_one)))
        self.num_of_digits_two = len(str(abs(self.operand_two)))
        
        self.result = self.operator.compute(operand_one=self.operand_one, operand_two=self.operand_two)
    
    @staticmethod
    def operand_to_list(operand:int) -> List[int]:
        return [ int(char) for char in list(operand) ]
    
    def __str__(self):
        return f'{self.operand_one}{self.operator}{self.operand_two}'
    def __repr__(self):
        return f'{self.operand_one}{self.operator}{self.operand_two}'
    


class Token:
    def __init__(self, token_str:str=None, token_id:int=None):
        assert not (token_str is None) and (token_id is None)
        self.token_str = token_str
        self.token_id  = token_id