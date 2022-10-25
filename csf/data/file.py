#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /file.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 25th 2022 02:05:17 am                                               #
# Modified   : Tuesday October 25th 2022 08:12:10 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Data Artifact Definition Module

This module has been loosely adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""
import os
from dataclasses import dataclass
import tensorflow as tf
from glob import glob
from typing import Union, Optional

from csf.config.base import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class FileConfig(Config):
    """File Configuration

    Args:
        root_dir (str): The root directory for the file. This is set based upon the value of 'on_kaggle'
        kaggle_base_input_folder (str): The read-only Kaggle folder containing all input data
        kaggle_base_working_folder (str): The Kaggle folder in which working files are stored.
        local_base_input_folder (str): The local folder containing all input data
        local_base_working_folder (str): The local folder containing working files.
        path (str): Path or path specification for a file or file set, not including root_dir
        full_path (str): Full path or path specification to the file or file set.
        on_kaggle (bool): True if running in a Kaggle notebook, False otherwise. Default = False
        input_data (bool): True if the File is part of the provided input data, False otherwise. Default = True
        path_is_glob (bool): True if the path should be treated as a glob, False otherwise. Default = False
    """

    KAGGLE_BASE_INPUT_FOLDER: str = "kaggle/input/rsna-2022-cervical-spine-fracture-detection"
    KAGGLE_BASE_WORKING_FOLDER: str = "kaggle/working/"
    LOCAL_BASE_INPUT_FOLDER = "data/input/"
    KAGGLE_BASE_WORKING_FOLDER = "data/working/"

    root_dir: str = ""
    path: str = ""
    full_path: str = ""
    on_kaggle: bool = False
    input_data: bool = True
    path_is_glob: bool = False

    def __post_init__(self) -> None:
        """Sets the root_dir into which the file resides"""
        if self.on_kaggle and self.input_data:
            self.root_dir = self.kaggle_base_input_folder
        elif self.on_kaggle and not self.input_data:
            self.root_dir = self.kaggle_base_working_folder
        elif not self.on_kaggle and self.input_data:
            self.root_dir = self.local_base_input_folder
        else:
            self.root_dir = self.local_base_working_folder

        if self.path_is_glob:
            self.full_path = glob(pathname=self.path, root_dir=self.root_dir)
        else:
            self.full_path = os.path.join(self.root_dir, self.path)
