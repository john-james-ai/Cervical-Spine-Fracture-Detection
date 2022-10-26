#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /test_config.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 25th 2022 01:08:13 pm                                               #
# Modified   : Tuesday October 25th 2022 07:57:30 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pytest
import logging
import logging.config

from csf.data.entity import Fileset
from csf.data.config import FilesetConfig

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.config
class TestBuildFileRepo
    def test_build_file_repo(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))




        fsc = FilesetConfig(
            name="test_fileset_config", storage_uri="data/input", event_name="build_metadata_event"
        )

        assert fsc.name == "test_fileset_config"
        assert fsc.storage_uri == "data/input"
        assert fsc.event_name == "build_metadata_event"

        fsc.description = "FSC Description"
        config = fsc.as_dict()

        assert isinstance(config, dict)
        assert isinstance(config[""], _{_name__} or tuple)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
