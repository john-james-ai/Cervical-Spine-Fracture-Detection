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
# Created    : Monday October 24th 2022 01:29:06 pm                                                #
# Modified   : Monday October 24th 2022 02:37:52 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Configuration Module for the Package"""
from csf.config.data import FileConfig, DatasetConfig

# ------------------------------------------------------------------------------------------------ #
#                                        FILE CONFIGS                                              #
# ------------------------------------------------------------------------------------------------ #
# Input File Configs
sample_submission_file_config = FileConfig(
    path="sample_submission.csv", path_is_glob=False, on_kaggle=False, input_data=True
)

test_file_config = FileConfig(path="test.csv", path_is_glob=False, on_kaggle=False, input_data=True)

train_file_config = FileConfig(
    path="train.csv", path_is_glob=False, on_kaggle=False, input_data=True
)

train_bounding_boxes_file_config = FileConfig(
    path="train.csv", path_is_glob=False, on_kaggle=False, input_data=True
)

train_images_file_config = FileConfig(
    path="train_images/**/*.dcm", path_is_glob=True, on_kaggle=False, input_data=True
)

test_images_file_config = FileConfig(
    path="test_images/**/*.dcm", path_is_glob=True, on_kaggle=False, input_data=True
)

segmentations_file_config = FileConfig(
    path="segmentations/*.nii", path_is_glob=True, on_kaggle=False, input_data=True
)

# ------------------------------------------------------------------------------------------------ #
# Working File Configs
segmentation_metadata_file_config = FileConfig(
    path="segmentation.metadata.csv", path_is_glob=False, on_kaggle=False, input_data=False
)
train_metadata_file_config = FileConfig(
    path="train_metadata.csv", path_is_glob=False, input_data=False, on_kaggle=False
)
