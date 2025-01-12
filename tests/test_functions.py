import unittest
import json
from src.parrotflume.functions import handle_function_call


class TestHandleFunctionCall(unittest.TestCase):
    def setUp(self):
        # Initialize a messages list for each test
        self.messages = []

    def test_handle_sympy_simplify(self):
        # Test handling a sympy_simplify function call
        function_call = type('FunctionCall', (), {'name': 'sympy_simplify', 'arguments': json.dumps({"expression": "x + x"})})

        # Call the function
        handle_function_call(self.messages, function_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "function")
        self.assertEqual(self.messages[0]["name"], "sympy_simplify")
        self.assertEqual(self.messages[0]["content"], "2*x")

    def test_handle_sympy_solve(self):
        # Test handling a sympy_solve function call
        function_call = type('FunctionCall', (), {'name': 'sympy_solve', 'arguments': json.dumps({"expression": "x**2 - 4", "variable": "x"})})

        # Call the function
        handle_function_call(self.messages, function_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "function")
        self.assertEqual(self.messages[0]["name"], "sympy_solve")
        self.assertEqual(self.messages[0]["content"], "[-2, 2]")

    def test_handle_sympy_integrate(self):
        # Test handling a sympy_integrate function call
        function_call = type('FunctionCall', (), {'name': 'sympy_integrate', 'arguments': json.dumps({"expression": "x**2", "variable": "x"})})

        # Call the function
        handle_function_call(self.messages, function_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "function")
        self.assertEqual(self.messages[0]["name"], "sympy_integrate")
        self.assertEqual(self.messages[0]["content"], "x**3/3")

    def test_handle_sympy_differentiate(self):
        # Test handling a sympy_differentiate function call
        function_call = type('FunctionCall', (), {'name': 'sympy_differentiate', 'arguments': json.dumps({"expression": "x**2", "variable": "x"})})

        # Call the function
        handle_function_call(self.messages, function_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "function")
        self.assertEqual(self.messages[0]["name"], "sympy_differentiate")
        self.assertEqual(self.messages[0]["content"], "2*x")

    def test_handle_regex_match(self):
        # Test handling a regex_match function call
        function_call = type('FunctionCall', (), {'name': 'regex_match', 'arguments': json.dumps({"pattern": r"\d+", "text": "123 abc 456"})})

        # Call the function
        handle_function_call(self.messages, function_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "function")
        self.assertEqual(self.messages[0]["name"], "regex_match")
        self.assertEqual(self.messages[0]["content"], "['123', '456']")


if __name__ == "__main__":
    unittest.main()
