import json
import re
from ast import literal_eval
from datetime import datetime
from sympy import simplify, solve, sympify, Eq, integrate, diff
from sympy.parsing.sympy_parser import parse_expr

functions = [
    {
        "name": "literal_eval",
        "description": "Evaluate a simple Python or math expression with ast.literal_eval()",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "simple Python or math expression"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_date",
        "description": "Get the current (today's) date",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "sympy_simplify",
        "description": "Simplify a mathematical expression using sympy.simplify",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "mathematical expression to simplify, using sympy syntax"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "sympy_solve",
        "description": "Solve a mathematical expression using sympy.solve",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "mathematical expression to solve, using sympy syntax"
                },
                "variable": {
                    "type": "string",
                    "description": "variable to solve for, using sympy syntax"
                }
            },
            "required": ["expression", "variable"]
        }
    },
    {
        "name": "sympy_integrate",
        "description": "Integrate a mathematical expression using sympy.integrate",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "mathematical expression to integrate, using sympy syntax"
                },
                "variable": {
                    "type": "string",
                    "description": "variable of integration, using sympy syntax"
                }
            },
            "required": ["expression", "variable"]
        }
    },
    {
        "name": "sympy_differentiate",
        "description": "Differentiate a mathematical expression using sympy.diff",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "mathematical expression to differentiate, using sympy syntax"
                },
                "variable": {
                    "type": "string",
                    "description": "variable of differentiation, using sympy syntax"
                },
                "order": {
                    "type": "integer",
                    "description": "order of differentiation (optional, default is 1)"
                }
            },
            "required": ["expression", "variable"]
        }
    },
    {
        "name": "regex_match",
        "description": "Applies a regex pattern to a text and returns the matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to apply."
                },
                "text": {
                    "type": "string",
                    "description": "The text to search within."
                }
            },
            "required": ["pattern", "text"]
        }
}
]


def handle_literal_eval(messages, arguments):
    try:
        result = literal_eval(arguments)
        messages.append({"role": "function", "name": "literal_eval", "content": str(result)})
    except (ValueError, SyntaxError) as e:
        messages.append({"role": "function", "name": "literal_eval", "content": str(e)})


def handle_get_current_date(messages):
    result = datetime.now().strftime("%Y-%m-%d")
    messages.append({"role": "function", "name": "get_current_date", "content": result})


def handle_sympy_simplify(messages, arguments):
    try:
        expr = parse_expr(arguments["expression"])
        simplified_expr = simplify(expr)
        messages.append({"role": "function", "name": "sympy_simplify", "content": str(simplified_expr)})
    except Exception as e:
        messages.append({"role": "function", "name": "sympy_simplify", "content": str(e)})


def handle_sympy_solve(messages, arguments):
    try:
        equation = arguments["expression"]
        variable = arguments["variable"]

        if "=" in equation:
            left, right = equation.split("=", 1)
            left_expr = parse_expr(left.strip())
            right_expr = parse_expr(right.strip())
            eq = Eq(left_expr, right_expr)
        else:
            eq = parse_expr(equation)

        solution = solve(eq, sympify(variable))
        messages.append({"role": "function", "name": "sympy_solve", "content": str(solution)})
    except Exception as e:
        messages.append({"role": "function", "name": "sympy_solve", "content": str(e)})


def handle_sympy_integrate(messages, arguments):
    try:
        expression = arguments["expression"]
        variable = sympify(arguments["variable"])

        expr = parse_expr(expression)
        result = integrate(expr, variable)
        messages.append({"role": "function", "name": "sympy_integrate", "content": str(result)})
    except Exception as e:
        messages.append({"role": "function", "name": "sympy_integrate", "content": str(e)})


def handle_sympy_differentiate(messages, arguments):
    try:
        expression = arguments["expression"]
        variable = sympify(arguments["variable"])
        order = arguments.get("order", 1)

        expr = parse_expr(expression)
        result = diff(expr, variable, order)
        messages.append({"role": "function", "name": "sympy_differentiate", "content": str(result)})
    except Exception as e:
        messages.append({"role": "function", "name": "sympy_differentiate", "content": str(e)})


def handle_regex_match(messages, arguments):
    try:
        pattern = arguments["pattern"]
        text = arguments["text"]

        result = re.findall(pattern, text)
        messages.append({"role": "function", "name": "regex_match", "content": str(result)})
    except Exception as e:
        messages.append({"role": "function", "name": "regex_match", "content": str(e)})


def handle_function_call(messages, function_call):
    args = json.loads(function_call.arguments)

    if function_call.name == "literal_eval":
        handle_literal_eval(messages, function_call.arguments)
    elif function_call.name == "get_current_date":
        handle_get_current_date(messages)
    elif function_call.name == "sympy_simplify":
        handle_sympy_simplify(messages, args)
    elif function_call.name == "sympy_solve":
        handle_sympy_solve(messages, args)
    elif function_call.name == "sympy_integrate":
        handle_sympy_integrate(messages, args)
    elif function_call.name == "sympy_differentiate":
        handle_sympy_differentiate(messages, args)
    elif function_call.name == "regex_match":
        handle_regex_match(messages, args)
