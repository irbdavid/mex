"""Python routines for MEX/MARSIS analsyis (and some MEX/ASPERA)"""

__all__ = ['mex_sc', 'maRs', 'ais']

# Generic time stuff
import celsius

# mex position stuff
from .mex_sc import *

# Init shit
data_directory = locate_data_directory()

import spiceypy

# from time import *
if not 'orbits' in locals(): orbits = read_all_mex_orbits()

from .maRs import *

from . import ais
from . import aspera

from .orbit_plots import plot_planet, plot_bs, plot_mpb

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "david.andrews@irfu.se"
