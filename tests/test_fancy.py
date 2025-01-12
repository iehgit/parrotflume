import unittest
from io import StringIO
import contextlib
from src.parrotflume.fancy import print_fancy, print_reset

class TestPrintFancy(unittest.TestCase):
    @staticmethod
    def capture_output(func, *args, **kwargs):
        """Helper function to capture stdout from a function."""
        captured_output = StringIO()
        with contextlib.redirect_stdout(captured_output):
            func(*args, **kwargs)
        return captured_output.getvalue().strip()

    def test_markdown_formatting(self):
        # Test combined Markdown formatting (headers, bold, italic, code blocks)
        text = (
            "### Header H3\n"
            "#### Header H4\n"
            "**Bold text**\n"
            "*Italic text*\n"
            "`inline code`\n"
            "```bash\n"
            "echo \"code block\"\n"
            "```"
        )
        output = self.capture_output(print_fancy, text, do_markdown=True, do_latex=False, do_color=False, color="bright_yellow")

        # Expected output with ANSI escape codes for Markdown formatting
        expected_output = (
            "\033[7mHeader H3\033[27m\n"
            "\033[4mHeader H4\033[24m\n"
            "\033[1mBold text\033[22m\n"
            "\033[3mItalic text\033[23m\n"
            "\033[100minline code\033[49m\n"
            "\033[2mbash\033[22m\033[100m\n"
            "echo \"code block\"\n"
            "\033[49m"
        )
        self.assertEqual(expected_output, output)

    def test_latex_conversion(self):
        # Test complex LaTeX-to-Unicode conversion
        text = r"\sqrt{x} + \frac{a}{b} = \int_{0}^{1} f(x) \, dx"
        output = self.capture_output(print_fancy, text, do_markdown=False, do_latex=True, do_color=False, color="bright_yellow")

        # Expected output after LaTeX-to-Unicode conversion
        expected_output = "√(x) + a/b = ∫_0^1 f(x)   dx"
        self.assertEqual(expected_output, output)

    def test_color_formatting(self):
        # Test color formatting
        output = self.capture_output(print_fancy, "Hello, World!", do_markdown=False, do_latex=False, do_color=True, color="bright_yellow")

        # Expected output with ANSI escape codes for bright yellow
        expected_output = "\033[1;33mHello, World!\033[39m"
        self.assertEqual(expected_output, output)

    def test_print_reset(self):
        # Test print_reset function
        output = self.capture_output(print_reset)

        # Expected output: ANSI reset code
        expected_output = "\033[0m"
        self.assertEqual(expected_output, output)

if __name__ == "__main__":
    unittest.main()