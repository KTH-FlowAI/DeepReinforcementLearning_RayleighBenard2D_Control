#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# configuration.py: Path to the binaries used for the DRL tool
#
# Pol Suarez, Arnau Miro
#
# 29/09/2022

from __future__ import print_function, division

import os


## PATHS
# Absolute path to the DRL directory
DRL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# Absolute path to the shenfun_files folder
ALYA_PATH = os.path.join(DRL_PATH,'shenfun_files')
# Absolute path to the binaries folder
BIN_PATH  = os.path.join(DRL_PATH,'shenfun_files','bin')


## FILE NAMES
NODELIST   = 'nodelist'

