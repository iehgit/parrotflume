import re
from pylatexenc.latex2text import LatexNodes2Text

# ANSI escape codes
BOLD = "\033[1m"
ITALIC = "\033[3m"
INVERT = "\033[7m"  # Inversion for h3 headers
UNDERLINE = "\033[4m"  # Underline for h4 headers
GREY_BG = "\033[100m"  # Grey background for code blocks
FAINT = "\033[2m"  # Dim for code block headers
YELLOW_FG = "\033[93m"

RESET_BOLD = "\033[22m"
RESET_ITALIC = "\033[23m"
RESET_INVERT = "\033[27m"
RESET_UNDERLINE = "\033[24m"
RESET_GREY_BG = "\033[49m"
RESET_FAINT = "\033[22m"
RESET_YELLOW_FG = "\033[39m"

RESET_GENERIC = "\033[0m"


def print_fancy(text, do_markdown, do_latex, do_color):
    """
    Processes the given text with Markdown, LaTeX, and color formatting.
    Supports h3, h4, italic (*), bold (**), code blocks, inline code, and LaTeX to Unicode conversion.
    """
    result = []
    in_code_block = False

    for line in text.splitlines():
        # Handle code blocks
        if line.startswith("\x60\x60\x60"):
            if not in_code_block:
                in_code_block = True
                result.append(FAINT + line[3:] + RESET_FAINT + GREY_BG + "\n")
            else:
                in_code_block = False
                result.append(line[3:] + RESET_GREY_BG + "\n")
            continue

        if not in_code_block:
            if do_markdown:
                # Render header h3
                if line.startswith("### "):
                    line = INVERT + line[4:] + RESET_INVERT
                # Render header h4
                elif line.startswith("#### "):
                    line = UNDERLINE + line[5:] + RESET_UNDERLINE

                # Render inline bold (**)
                try:
                    line = re.sub(r"\*\*(.*?)\*\*", lambda m: BOLD + (m.group(1) or m.group(2)) + RESET_BOLD, line)
                except IndexError:
                    pass  # ignore

                # Render inline italic (*)
                try:
                    line = re.sub(r"\*(.*?)\*", lambda m: ITALIC + (m.group(1) or m.group(2)) + RESET_ITALIC, line)
                except IndexError:
                    pass  # ignore

                # Render inline code (`)
                try:
                    line = re.sub(r"`(.*?)`", lambda m: GREY_BG + (m.group(1) or m.group(2)) + RESET_GREY_BG, line)
                except IndexError:
                    pass  # ignore

            if do_latex and not re.search(r"`.*?`", line):
                # Convert LaTeX to Unicode
                latex2text = LatexNodes2Text(keep_comments=True)
                line = latex2text.latex_to_text(line, tolerant_parsing=True)

            if do_color:
                # Apply yellow foreground color
                line = YELLOW_FG + line + RESET_YELLOW_FG

        result.append(line + "\n")

    print("".join(result), end=None)


def print_reset():
    print(RESET_GENERIC, end=None)