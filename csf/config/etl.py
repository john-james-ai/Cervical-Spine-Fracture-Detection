#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /etl.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday November 1st 2022 03:36:45 pm                                               #
# Modified   : Tuesday November 1st 2022 03:42:49 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Configuration Module for ETL"""
from dataclasses import dataclass
from csf.base.config import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class SegmentationVertebraeExtractorConfig(Config):
    name: str = "segmentation_label_extractor"
    source: str = "input/segmentations/*.nii"
    target: str = "working/segmentation_metadata.csv"
    n_jobs: int = 12
    verbose: int = 10
    force: bool = False
