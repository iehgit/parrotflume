import json
from ast import literal_eval
from datetime import datetime
from sympy import simplify, solve, sympify, Eq, integrate, diff
from sympy.parsing.sympy_parser import parse_expr

functions = [{"name": "literal_eval", "description": "Evaluate a simple Python or math expression with ast.literal_eval()",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "simple Python or math expression"}},
        "required": ["expression"]}},
    {"name": "get_current_date", "description": "Get the current (todays) date", "parameters": {"type": "object", "properties": {}}},
    {"name": "sympy_simplify", "description": "Simplify a mathematical expression using sympy.simplify", "parameters": {"type": "object",
        "properties": {"expression": {"type": "string", "description": "mathematical expression to simplify, using sympy syntax"}},
        "required": ["expression"]}}, {"name": "sympy_solve", "description": "Solve a mathematical expression using sympy.solve",
        "parameters": {"type": "object", "properties": {"equation": {"type": "string", "description": "mathematical expression to solve, using sympy syntax"},
            "variable": {"type": "string", "description": "variable to solve for, using sympy syntax"}}, "required": ["equation", "variable"]}},
    {"name": "sympy_integrate", "description": "Integrate a mathematical expression using sympy.integrate", "parameters": {"type": "object",
        "properties": {"expression": {"type": "string", "description": "mathematical expression to integrate, using sympy syntax"},
            "variable": {"type": "string", "description": "variable of integration, using sympy syntax"}}, "required": ["expression", "variable"]}},
    {"name": "sympy_differentiate", "description": "Differentiate a mathematical expression using sympy.diff", "parameters": {"type": "object",
        "properties": {"expression": {"type": "string", "description": "mathematical expression to differentiate, using sympy syntax"},
            "variable": {"type": "string", "description": "variable of differentiation, using sympy syntax"},
            "order": {"type": "integer", "description": "order of differentiation (optional, default is 1)"}}, "required": ["expression", "variable"]}}]


def handle_function_call(messages, function_call):
    if function_call.name == "literal_eval":
        try:
            result = literal_eval(function_call.arguments)
            messages.append({"role": "function", "name": "literal_eval", "content": str(result)})
        except (ValueError, SyntaxError) as e:
            messages.append({"role": "function", "name": "literal_eval", "content": str(e)})
    elif function_call.name == "get_current_date":
        result = datetime.now().strftime("%Y-%m-%d")
        messages.append({"role": "function", "name": "get_current_date", "content": result})
    elif function_call.name == "sympy_simplify":
        try:
            args = json.loads(function_call.arguments)
            expr = parse_expr(args["expression"])
            simplified_expr = simplify(expr)
            messages.append({"role": "function", "name": "sympy_simplify", "content": str(simplified_expr)})
        except Exception as e:
            messages.append({"role": "function", "name": "sympy_simplify", "content": str(e)})
    elif function_call.name == "sympy_solve":
        try:
            args = json.loads(function_call.arguments)
            equation = args["equation"]
            variable = args["variable"]

            # Parse the equation into a sympy Eq object
            if "=" in equation:
                left, right = equation.split("=", 1)  # Split on the first "="
                left_expr = parse_expr(left.strip())
                right_expr = parse_expr(right.strip())
                eq = Eq(left_expr, right_expr)
            else:
                # If no "=", assume the equation is already in the form "expression = 0"
                eq = parse_expr(equation)

            # Solve the equation
            solution = solve(eq, sympify(variable))
            messages.append({"role": "function", "name": "sympy_solve", "content": str(solution)})
        except Exception as e:
            messages.append({"role": "function", "name": "sympy_solve", "content": str(e)})
    elif function_call.name == "sympy_integrate":
        try:
            args = json.loads(function_call.arguments)
            expression = args["expression"]
            variable = sympify(args["variable"])

            # Parse the expression and perform indefinite integration
            expr = parse_expr(expression)
            result = integrate(expr, variable)
            messages.append({"role": "function", "name": "sympy_integrate", "content": str(result)})
        except Exception as e:
            messages.append({"role": "function", "name": "sympy_integrate", "content": str(e)})
    elif function_call.name == "sympy_differentiate":
        try:
            args = json.loads(function_call.arguments)
            expression = args["expression"]
            variable = sympify(args["variable"])
            order = args.get("order", 1)

            # Parse the expression and perform differentiation
            expr = parse_expr(expression)
            result = diff(expr, variable, order)
            messages.append({"role": "function", "name": "sympy_differentiate", "content": str(result)})
        except Exception as e:
            messages.append({"role": "function", "name": "sympy_differentiate", "content": str(e)})
