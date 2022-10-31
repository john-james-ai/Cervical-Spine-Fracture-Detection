#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /test_etl.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 29th 2022 09:05:44 am                                              #
# Modified   : Sunday October 30th 2022 07:03:18 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pytest
import logging
import logging.config

from csf.base.pipeline import DataPipeBuilder, DataPipe
from csf.base.io import IOFactory

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.etl
class TestETL:
    def test_download(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        basedir = ""
        filepath = "csf/config/etl.yml"
        factory = IOFactory(basedir)
        io = factory.create("yml")
        config = io.read(filepath)

        basedir = "data"
        factory = IOFactory(basedir)

        builder = DataPipeBuilder(config=config, io_factory=factory)
        builder.build()

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
