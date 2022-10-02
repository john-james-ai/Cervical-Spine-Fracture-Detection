#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /test_process.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 1st 2022 03:24:06 pm                                               #
# Modified   : Saturday October 1st 2022 03:43:14 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import numpy as np
import pytest
import logging
import logging.config

# Enter imports for modules and classes being tested here
from csf.data.process import HounsfieldTransformer  # , Windower, Denoiser

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.process
class TestProcess:
    def test_hu_transformer(self, image):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        name = "HU Transformer"
        params = {"display": True}
        xformer = HounsfieldTransformer(name=name, params=params)
        hu_image = xformer.execute(image)
        assert isinstance(hu_image, np.array)
        assert hu_image.shape == (512, 512)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
