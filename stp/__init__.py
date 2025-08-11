#!/usr/bin/env python3
"""Top-level init file to make package available for import.

**Author: Jonathan Delgado**

Handles importing and setting project-wide fields.

"""

__author__ = 'Jonathan Delgado'
__version__ = '0.0.1.2'

# Make all functions from stochastic accessible on import of stp
from .stochastic import *

#======================== Rich ========================#
#------------- Imports -------------#
import rich.theme
import rich.progress
import rich.console
import rich.prompt # override input with Prompt.ask
# Improved tracebacks
import rich.traceback; rich.traceback.install()
#------------- Settings -------------#
# Store colors as variables for use with library objects
jd_blue = '#0675BB'
jd_green = '#ADEBBB'
# Custom theme
theme = rich.theme.Theme({
    # Syntax highlighting for numbers, light mint
    "repr.number": "#9DFBCC",
    #--- Colors ---#
    'green': jd_green,
    #--- Semantic colors ---#
    'success': jd_green,
    # Emphasis
    'emph': 'blue',
    # Softer red than a failure
    'warning': 'red',
    # Amaranth red
    'failure': '#E03E52'
})
#--- Input and printing ---#
console = rich.console.Console(theme=theme)
# Override
print = console.print
input = console.input
# New prompts
ask = lambda text : rich.prompt.Prompt.ask(text, console=console)
confirm = lambda text, default=True : rich.prompt.Confirm.ask(
    text, default=default, console=console
)
# Provide a rich status function for indeterminate progress
status = lambda text: console.status(
    text, spinner='dots', spinner_style=jd_blue
)
#--- Progress bar ---#
def Progress(label='Progress'):
    """ Overload constructor for generating progress bars. """
    return rich.progress.Progress(
        rich.progress.SpinnerColumn('dots', style=jd_blue),
        rich.progress.TextColumn(f'{label}:', style=jd_blue),
        rich.progress.BarColumn(complete_style=jd_green, finished_style=jd_blue),
        rich.progress.MofNCompleteColumn(),
        console=console
    )
#======================== End Rich ========================#