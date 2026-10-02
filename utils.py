"""
Tiny ANSI colour helper shared by the model scripts.

Colour is off when NO_COLOR is set to a non-empty value (https://no-color.org)
or when stdout is not a terminal, so piped or redirected output stays clean.
Pad text to its column width BEFORE colouring it -- escape codes count
towards str.format widths but take no space on screen.
"""

import os
import sys


def colour_enabled(stream=sys.stdout):
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def _style(code):
    def apply(text):
        text = str(text)
        return f"\033[{code}m{text}\033[0m" if colour_enabled() else text
    return apply


green = _style("32")    # matches the paper
red = _style("31")      # does not match
dim = _style("2")       # derived or non-paper figure
bold = _style("1")


def match(ok):
    """green if ok else red -- for colouring a value by whether it matched."""
    return green if ok else red
