#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /file.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 07:35:27 am                                                #
# Modified   : Thursday October 27th 2022 02:36:34 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""File Configuration: Contains paths for all files."""
import os
from dataclasses import dataclass

from csf.base.config import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class FileConfig(Config):
    """File Configuration. Contains the paths to all files and directories for the given runtime environment.

    All files used must be included in the PATHS dictionary. In addition, there must be a base
    directory entry in the BASEDIR dictionary for each environment.

    Args:
        environment (str): One of ['local','kaggle']
    """

    PATHS = {
        "segmentations": "input/segmentations",
        "train_images": "input/train_images",
        "test_images": "input/test_images",
        "train": "input/train.csv",
        "test": "input/test.csv",
        "train_bbox_data": "input/train_bounding_boxes.csv",
        "segmentations_labels": "working/segmentations_labels.csv",
        "train_metadata": "working/train_metadata.csv",
    }
    BASEDIR = {"local": "data", "kaggle": "kaggle"}
    environment: str = "local"

    def __post_init__(self) -> None:
        """Adds filepaths for the designated environment to the class"""
        if self.environment not in self.BASEDIR.keys():
            raise ValueError(
                "Environment {} requires a base directory entry in BASEDIR.".format(
                    self.environment
                )
            )
        else:
            for name, path in self.PATHS.items():
                relpath = os.path.join(self.BASEDIR[self.environment], path)
                setattr(self, name, relpath)
