#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday September 13th 2022 08:33:33 pm                                             #
# Modified   : Saturday September 17th 2022 01:06:30 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pytest
import pandas as pd
from csf.data.studies import CSFStudies
from csf.data.scans import CSFScan

# ------------------------------------------------------------------------------------------------ #
TRAINING_METADATA_FILEPATH = "data/raw/train.csv"
# ------------------------------------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def training_metadata():
    return pd.read_csv(TRAINING_METADATA_FILEPATH, index_col=False)


@pytest.fixture(scope="module")
def csf_test():
    return CSFStudies(TRAINING_METADATA_FILEPATH)


@pytest.fixture(scope="module")
def scan():
    studies = CSFStudies(filepath=TRAINING_METADATA_FILEPATH)
    study = studies.get_sample_study_by_fracture_count(n=2)
    scan = CSFScan(study=study)
    return scan
