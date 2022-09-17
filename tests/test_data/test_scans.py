#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /test_scans.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday September 15th 2022 12:37:38 pm                                            #
# Modified   : Saturday September 17th 2022 04:01:35 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pytest
import logging
import logging.config
from csf.data import Study

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.scan
class TestScan:
    def test_scan_properties(self, caplog, scan):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        assert isinstance(scan.study, Study)
        assert isinstance(scan.basedir, str)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_load_slices(self, caplog, scan):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        assert isinstance(scan.img_shape, list)
        # assert isinstance(scan.patient_name, tuple(None, str))
        # assert isinstance(scan.patient_id, tuple(None, str))
        # assert isinstance(scan.modality, tuple(None, str))
        # assert isinstance(scan.study_date, tuple(None, str))

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_print_slice(self, caplog, scan):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        scan.print_slice()
        assert True

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_slice_viewer(self, caplog, scan):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        assert True

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
