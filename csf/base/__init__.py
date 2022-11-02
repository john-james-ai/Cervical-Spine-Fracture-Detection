#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /__init__.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday November 1st 2022 11:24:14 pm                                               #
# Modified   : Wednesday November 2nd 2022 05:18:31 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Constants used throughout the package."""

file_formats = ["csv", "yml", "yaml", "pickle", "pkl", "nii", "nib", "dcm", "h5"]
# ------------------------------------------------------------------------------------------------ #

IMMUTABLE_TYPES: tuple = (str, int, float, bool, type(None))
SEQUENCE_TYPES: tuple = (list, tuple)
