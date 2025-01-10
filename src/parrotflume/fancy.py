import re

from pylatexenc import latexwalker, latex2text
from pylatexenc.macrospec import MacroSpec

#  Overwrite sqrt macros from pylatexenc, the original wrongly renders n-th roots as square roots
lw_context_db = latexwalker.get_default_latex_context_db()
lw_context_db.add_context_category(
    'roots',
    prepend=True,
    macros=[
        MacroSpec("sqrt", "[{")
    ],
)

l2t_context_db = latex2text.get_default_latex_context_db()
l2t_context_db.add_context_category(
    'roots',
    prepend=True,
    macros=[
        latex2text.MacroTextSpec("sqrt", simplify_repl="%(1)s√(%(2)s)"),
    ],
)

# ANSI escape codes
BOLD = "\033[1m"
ITALIC = "\033[3m"
INVERT = "\033[7m"  # Inversion for h3 headers
UNDERLINE = "\033[4m"  # Underline for h4 headers
GREY_BG = "\033[100m"  # Grey background for code blocks
FAINT = "\033[2m"  # Dim for code block headers

RESET_BOLD = "\033[22m"
RESET_ITALIC = "\033[23m"
RESET_INVERT = "\033[27m"
RESET_UNDERLINE = "\033[24m"
RESET_GREY_BG = "\033[49m"
RESET_FAINT = "\033[22m"

RESET_FG = "\033[39m"
RESET_GENERIC = "\033[0m"

fg_color_map = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[1;30m",
    "bright_red": "\033[1;31m",
    "bright_green": "\033[1;32m",
    "bright_yellow": "\033[1;33m",
    "bright_blue": "\033[1;34m",
    "bright_magenta": "\033[1;35m",
    "bright_cyan": "\033[1;36m",
    "bright_white": "\033[1;37m",
}


def print_fancy(text, do_markdown, do_latex, do_color, color):
    """
    Processes the given text with Markdown, LaTeX, and color formatting.
    Supports h3, h4, italic (*), bold (**), code blocks, inline code, and LaTeX to Unicode conversion.
    """
    result = []
    in_code_block = False

    for line in text.splitlines():
        # Handle code blocks
        if line.lstrip().startswith("\x60\x60\x60"):
            line = line.lstrip()
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
                line = custom_latex_to_text(line)

            if do_color:
                # Apply yellow foreground color
                line = fg_color_map.get(color, RESET_FG) + line + RESET_FG

        result.append(line + "\n")

    print("".join(result), end=None)


def print_reset():
    print(RESET_GENERIC, end=None)


def custom_latex_to_text(input_latex):
    # the latex parser instance with custom latex_context
    lw_obj = latexwalker.LatexWalker(input_latex, latex_context=lw_context_db, tolerant_parsing=True)
    # parse to node list
    nodelist, pos, length = lw_obj.get_latex_nodes()
    # initialize the converter to text with custom latex_context
    l2t_obj = latex2text.LatexNodes2Text(latex_context=l2t_context_db, keep_comments=True)
    # convert to text
    return l2t_obj.nodelist_to_text(nodelist)
