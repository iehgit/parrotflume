import unittest
import json
from parrotflume.tools import handle_tool_call


class TestHandleToolCall(unittest.TestCase):
    def setUp(self):
        # Initialize a messages list for each test
        self.messages = []

    def test_handle_get_current_date(self):
        # Test handling a get_current_date tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'get_current_date',
                'arguments': json.dumps({})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "get_current_date")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

        # Verify the content is a valid date in the format YYYY-MM-DD
        from datetime import datetime
        try:
            datetime.strptime(self.messages[0]["content"], "%Y-%m-%d")
            date_is_valid = True
        except ValueError:
            date_is_valid = False
        self.assertTrue(date_is_valid, "The date format is invalid")

    def test_handle_sympy_simplify(self):
        # Test handling a sympy_simplify tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'sympy_simplify',
                'arguments': json.dumps({"expression": "x + x"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "sympy_simplify")
        self.assertEqual(self.messages[0]["content"], "2*x")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

    def test_handle_sympy_solve(self):
        # Test handling a sympy_solve tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'sympy_solve',
                'arguments': json.dumps({"expression": "x**2 - 4", "variable": "x"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "sympy_solve")
        self.assertEqual(self.messages[0]["content"], "[-2, 2]")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

    def test_handle_sympy_integrate(self):
        # Test handling a sympy_integrate tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'sympy_integrate',
                'arguments': json.dumps({"expression": "x**2", "variable": "x"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "sympy_integrate")
        self.assertEqual(self.messages[0]["content"], "x**3/3")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

    def test_handle_sympy_differentiate(self):
        # Test handling a sympy_differentiate tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'sympy_differentiate',
                'arguments': json.dumps({"expression": "x**2", "variable": "x"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "sympy_differentiate")
        self.assertEqual(self.messages[0]["content"], "2*x")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

    def test_handle_regex_match(self):
        # Test handling a regex_match tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'regex_match',
                'arguments': json.dumps({"pattern": r"\d+", "text": "123 abc 456"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "regex_match")
        self.assertEqual(self.messages[0]["content"], "['123', '456']")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

    def test_handle_count_chars(self):
        # Test handling a count_chars tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'count_chars',
                'arguments': json.dumps({"text": "1234567890"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "count_chars")
        self.assertEqual(self.messages[0]["content"], "10")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")

    def test_handle_count_substring_occurrences(self):
        # Test handling a count_substring_occurrences tool call
        tool_call = type('ToolCall', (), {
            'id': 'tool_call_123',
            'function': type('Function', (), {
                'name': 'count_substring_occurrences',
                'arguments': json.dumps({"string": "strawberry", "substring": "r"})
            })
        })

        # Call the function
        handle_tool_call(self.messages, tool_call)

        # Verify the result
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0]["role"], "tool")
        self.assertEqual(self.messages[0]["name"], "count_substring_occurrences")
        self.assertEqual(self.messages[0]["content"], "3")
        self.assertEqual(self.messages[0]["tool_call_id"], "tool_call_123")


if __name__ == "__main__":
    unittest.main()