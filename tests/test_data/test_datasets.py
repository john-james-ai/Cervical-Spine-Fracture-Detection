#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /test_datasets.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 23rd 2022 09:26:01 pm                                                #
# Modified   : Monday October 24th 2022 01:10:10 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pandas as pd
import tensorflow as tf
import pytest
import logging
import logging.config

from csf.data.dataset import RSNALabelDatasetBuilder
from csf import SEGMENTATION_METADATA_FILEPATH

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.dataset
class TestDatasetBuilder:
    def test_dataset_builder(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        seg_meta = pd.read_csv(SEGMENTATION_METADATA_FILEPATH, index_col=None)
        builder = RSNALabelDatasetBuilder(metadata=seg_meta, seed=22, eager=True)
        builder.build()
        print(builder.spec)
        ds = builder.dataset
        assert isinstance(ds, tf.data.Dataset)
        print(ds.element_spec)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
