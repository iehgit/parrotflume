import json
import re
from datetime import datetime
from sympy import simplify, solve, sympify, Eq, integrate, diff
from sympy.parsing.sympy_parser import parse_expr

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current (today's) date",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
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
        }
    },
    {
        "type": "function",
        "function": {
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
        }
    },
    {
        "type": "function",
        "function": {
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
        }
    },
    {
        "type": "function",
        "function": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "regex_match",
            "description": "Applies a regex pattern to a text and returns the matches",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "regex pattern to apply."
                    },
                    "text": {
                        "type": "string",
                        "description": "text to search within."
                    }
                },
                "required": ["pattern", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_chars",
            "description": "Counts the number of characters in a string",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "string whose characters are to be counted",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


def handle_get_current_date(messages):
    result = datetime.now().strftime("%Y-%m-%d")
    messages.append({"role": "tool", "name": "get_current_date", "content": result})


def handle_sympy_simplify(messages, arguments):
    try:
        expr = parse_expr(arguments["expression"])
        simplified_expr = simplify(expr)
        messages.append({"role": "tool", "name": "sympy_simplify", "content": str(simplified_expr)})
    except Exception as e:
        messages.append({"role": "tool", "name": "sympy_simplify", "content": str(e)})


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
        messages.append({"role": "tool", "name": "sympy_solve", "content": str(solution)})
    except Exception as e:
        messages.append({"role": "tool", "name": "sympy_solve", "content": str(e)})


def handle_sympy_integrate(messages, arguments):
    try:
        expression = arguments["expression"]
        variable = sympify(arguments["variable"])

        expr = parse_expr(expression)
        result = integrate(expr, variable)
        messages.append({"role": "tool", "name": "sympy_integrate", "content": str(result)})
    except Exception as e:
        messages.append({"role": "tool", "name": "sympy_integrate", "content": str(e)})


def handle_sympy_differentiate(messages, arguments):
    try:
        expression = arguments["expression"]
        variable = sympify(arguments["variable"])
        order = arguments.get("order", 1)

        expr = parse_expr(expression)
        result = diff(expr, variable, order)
        messages.append({"role": "tool", "name": "sympy_differentiate", "content": str(result)})
    except Exception as e:
        messages.append({"role": "tool", "name": "sympy_differentiate", "content": str(e)})


def handle_regex_match(messages, arguments):
    try:
        pattern = arguments["pattern"]
        text = arguments["text"]

        result = re.findall(pattern, text)
        messages.append({"role": "tool", "name": "regex_match", "content": str(result)})
    except Exception as e:
        messages.append({"role": "tool", "name": "regex_match", "content": str(e)})


def handle_count_chars(messages, arguments):
    try:
        text = arguments["text"]

        result = len(text)
        messages.append({"role": "tool", "name": "count_chars", "content": str(result)})
    except Exception as e:
        messages.append({"role": "tool", "name": "count_chars", "content": str(e)})


def handle_tool_call(messages, tool_call, do_print=False):
    args = json.loads(tool_call.function.arguments)
    if do_print:
        print(f"[{tool_call.function.name} called]")

    if tool_call.function.name == "get_current_date":
        result = datetime.now().strftime("%Y-%m-%d")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "get_current_date",
            "content": result
        })
    elif tool_call.function.name == "sympy_simplify":
        try:
            expr = parse_expr(args["expression"])
            simplified_expr = simplify(expr)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_simplify",
                "content": str(simplified_expr)
            })
        except Exception as e:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_simplify",
                "content": str(e)
            })
    elif tool_call.function.name == "sympy_solve":
        try:
            equation = args["expression"]
            variable = args["variable"]

            if "=" in equation:
                left, right = equation.split("=", 1)
                left_expr = parse_expr(left.strip())
                right_expr = parse_expr(right.strip())
                eq = Eq(left_expr, right_expr)
            else:
                eq = parse_expr(equation)

            solution = solve(eq, sympify(variable))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_solve",
                "content": str(solution)
            })
        except Exception as e:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_solve",
                "content": str(e)
            })
    elif tool_call.function.name == "sympy_integrate":
        try:
            expression = args["expression"]
            variable = sympify(args["variable"])

            expr = parse_expr(expression)
            result = integrate(expr, variable)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_integrate",
                "content": str(result)
            })
        except Exception as e:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_integrate",
                "content": str(e)
            })
    elif tool_call.function.name == "sympy_differentiate":
        try:
            expression = args["expression"]
            variable = sympify(args["variable"])
            order = args.get("order", 1)

            expr = parse_expr(expression)
            result = diff(expr, variable, order)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_differentiate",
                "content": str(result)
            })
        except Exception as e:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "sympy_differentiate",
                "content": str(e)
            })
    elif tool_call.function.name == "regex_match":
        try:
            pattern = args["pattern"]
            text = args["text"]

            result = re.findall(pattern, text)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "regex_match",
                "content": str(result)
            })
        except Exception as e:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "regex_match",
                "content": str(e)
            })
    elif tool_call.function.name == "count_chars":
        try:
            text = args["text"]

            result = len(text)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "count_chars",
                "content": str(result)
            })
        except Exception as e:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "count_chars",
                "content": str(e)
            })
