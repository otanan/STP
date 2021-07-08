#!/usr/bin/env python3
"""Top-level init file to make package available for import.

**Author: Jonathan Delgado**

Handles importing and setting project-wide fields.

"""

__author__ = 'Jonathan Delgado'
__version__ = '0.0.1.0'

# Make all functions from stochastic accessible on import of stp
from .stochastic import *