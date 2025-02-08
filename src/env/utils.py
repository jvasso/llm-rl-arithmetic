
import re



def check_expr(expression:str):
    try:
        # Check if the expression has any invalid characters
        if not re.match("^[ 0-9+\-*/=()]*$", expression):
            return False
        
        # Split the expression by '=' into separate equations
        equations = expression.split("=")
        
        # Ensure each equation has numbers and operators
        if any(not re.search("[0-9]", eq) for eq in equations):
            return False
        
        # Evaluate each equation
        values = [eval(eq.strip()) for eq in equations]
        
        # Check if all evaluations are equal
        is_true = all(val == values[0] for val in values)
        
        return is_true
    except Exception as e:
        # Return False if there is any error in evaluating the expression
        return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False