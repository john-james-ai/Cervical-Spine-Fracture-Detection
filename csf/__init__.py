#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /__init__.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday September 5th 2022 04:51:01 pm                                               #
# Modified   : Sunday October 23rd 2022 11:10:11 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-Clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
# Constants used throughout the package
import os

# ------------------------------------------------------------------------------------------------ #
#                                         ENVIRONMENT                                              #
# ------------------------------------------------------------------------------------------------ #
IS_KAGGLE = False

# ------------------------------------------------------------------------------------------------ #
#                                 BASE ENVIRONMENT DIRECTORIES                                     #
# ------------------------------------------------------------------------------------------------ #
KAGGLE = {
    "INPUT": "../input/rsna-2022-cervical-spine-fracture-detection",
    "METADATA": "../working/metadata/rsna-2022-cervical-spine-fracture-detection",
    "PROCESSED": "../working/processed/rsna-2022-cervical-spine-fracture-detection",
    "MODELS": "../working/models/rsna-2022-cervical-spine-fracture-detection",
}
LOCAL = {
    "INPUT": "data/input",
    "METADATA": "data/metadata",
    "PROCESSED": "data/processed",
    "MODELS": "models",
}

if IS_KAGGLE:
    BASE_DIRS = KAGGLE
else:
    BASE_DIRS = LOCAL

# ------------------------------------------------------------------------------------------------ #
#                                    INPUT IMAGE DATA                                              #
# ------------------------------------------------------------------------------------------------ #
# Key Image Subdirectories
TRAIN_IMAGES_DIR = f"{BASE_DIRS['INPUT']}/train_images"
TEST_IMAGES_DIR = f"{BASE_DIRS['INPUT']}/test_images"
SEGMENTATION_DIR = f"{BASE_DIRS['INPUT']}/segmentations"

# Filepaths
TRAIN_IMAGES_FILEPATHS = f"{TRAIN_IMAGES_DIR}/**/*.dcm"
TEST_IMAGES_FILEPATHS = f"{TEST_IMAGES_DIR}/**/*.dcm"
SEGMENTATION_FILEPATHS = f"{SEGMENTATION_DIR}/*.nii"


# ------------------------------------------------------------------------------------------------ #
#                                   INPUT TABULAR DATA                                             #
# ------------------------------------------------------------------------------------------------ #
# Tabular Filenames
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"
TRAIN_BBOXES_FILENAME = "train_bounding_boxes.csv"


# Tabular Filepaths
TRAIN_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_FILENAME}"
TEST_FILEPATH = f"{BASE_DIRS['INPUT']}/{TEST_FILENAME}"
SAMPLE_SUBMISSION_FILEPATH = f"{BASE_DIRS['INPUT']}/{SAMPLE_SUBMISSION_FILENAME}"
TRAIN_BBOX_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_BBOXES_FILENAME}"


# ------------------------------------------------------------------------------------------------ #
#                                        TRAIN METADATA                                            #
# ------------------------------------------------------------------------------------------------ #
# Filenames
TRAIN_SLICE_METADATA_FILENAME = "train_slice_metadata.csv"
TRAIN_SCAN_METADATA_FILENAME = "train_scan_metadata.csv"


# Filepaths
TRAIN_SLICE_METADATA_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_SLICE_METADATA_FILENAME}"
TRAIN_SCAN_METADATA_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_SCAN_METADATA_FILENAME}"


# ------------------------------------------------------------------------------------------------ #
#                                        SEGMENTATION METADATA                                     #
# ------------------------------------------------------------------------------------------------ #
# Labels
TRAIN_SEGMENTATION_METADATA_FILENAME = "train_segmentation_metadata.csv"
TRAIN_SEGMENTATION_METADATA_FILEPATH = (
    f"{BASE_DIRS['INPUT']}/{TRAIN_SEGMENTATION_METADATA_FILENAME}"
)


# ------------------------------------------------------------------------------------------------ #
#                                           MULTIPROCESSING                                        #
# ------------------------------------------------------------------------------------------------ #
N_JOBS = 12

# ------------------------------------------------------------------------------------------------ #
#                                           VISUALIZATION                                          #
# ------------------------------------------------------------------------------------------------ #
FIG_SIZE = (12, 4)

# ------------------------------------------------------------------------------------------------ #
#                                           IMAGE PROCESSING                                       #
# ------------------------------------------------------------------------------------------------ #
WINDOW_DEFAULT = (1200, 250)
DICOM_IMAGE_SHAPE = [512, 512]
EFFICIENTNET_INPUT_SHAPES = {
    "B0": [224, 224, 3],
    "B1": [240, 240, 3],
    "B2": [260, 260, 3],
    "B3": [300, 300, 3],
    "B4": [380, 380, 3],
    "B5": [456, 456, 3],
    "B6": [528, 528, 3],
    "B7": [600, 600, 3],
}


# ------------------------------------------------------------------------------------------------ #
#                                             MODELS                                               #
# ------------------------------------------------------------------------------------------------ #
LABELING_MODEL = {"name": "EfficientNetB3", "code": "B3"}
EFFICIENTNET_INPUT_SHAPE = EFFICIENTNET_INPUT_SHAPES[LABELING_MODEL["code"]]
KFOLDS = 5
BATCH_SIZE = 32
NO_CLASSES = 7
NO_EPOCHS = 25
