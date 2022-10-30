#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /test_segment.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 27th 2022 06:43:32 pm                                              #
# Modified   : Saturday October 29th 2022 06:55:24 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import pandas as pd
import inspect
import pytest
import logging
import logging.config

# Enter imports for modules and classes being tested here
from csf.data.etl import SegmentationVertebraeExtractor

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.vertebrae
class TestVertebraeExtractor:
    def test_vertebrae_extractor(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        input_path = "data/input/segmentations/*.nii"
        output_path = "tests/data/preprocess/vertebrae.csv"
        study_id = "1.2.826.0.1.3680043.12281"
        slice_number = 116

        ext = RSNASegmentationVertebraeExtractor(
            input_path=input_path, output_path=output_path, force=False
        )
        ext.execute()

        assert os.path.exists(output_path)

        df = pd.read_csv(output_path, index_col=False)
        print(df.head())
        n_patients = df["StudyInstanceUID"].nunique()
        n_images = df.shape[0]
        print(
            "There are {} patients with {} images for an average of {} per patient.".format(
                n_patients, n_images, n_images / n_patients
            )
        )

        assert df.shape[0] > 1000
        assert df.shape[1] == 9

        slice = df.loc[(df["StudyInstanceUID"] == study_id) & (df["SliceNumber"] == slice_number)]
        print("\n")
        print(slice)
        vertebrae = [0, 1, 0, 0, 0, 0, 0]
        for i in range(1, 8):
            assert slice[f"C{i}"].values == vertebrae[i - 1]

        assert isinstance(ext.started, str)
        assert isinstance(ext.ended, str)
        assert isinstance(ext.duration, str)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
