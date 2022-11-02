#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /test_io.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday November 1st 2022 05:20:09 pm                                               #
# Modified   : Tuesday November 1st 2022 08:50:38 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import inspect
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import logging.config
from csf.base.io import IOFactory

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.io
class TestIO:
    def test_csv(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # Read
        fp1 = "tests/data/io/test1.csv"
        io = IOFactory.create("csv")
        df1 = io.read(fp1)
        assert isinstance(df1, pd.DataFrame)

        # Read Filenot found
        fp2 = "tests/data/io/testnot.csv"
        with pytest.raises(FileNotFoundError):
            df1 = io.read(fp2)

        # Write
        fp2 = "tests/data/io/test2.csv"
        os.remove(fp2)
        io.write(fp2, df1)
        assert os.path.exists(fp2)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_yaml(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # Read
        fp1 = "tests/data/io/test1.yml"
        io = IOFactory.create("yml")
        d1 = io.read(fp1)
        assert isinstance(d1, dict)

        # Read Filenot found
        fp2 = "tests/data/io/testnot.yml"
        with pytest.raises(FileNotFoundError):
            d1 = io.read(fp2)

        # Write
        fp2 = "tests/data/io/test2.yml"
        os.remove(fp2)
        io.write(fp2, d1)
        assert os.path.exists(fp2)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_pickle(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # Read
        fp1 = "tests/data/io/test1.pickle"
        io = IOFactory.create("pickle")
        df1 = io.read(fp1)
        assert isinstance(df1, pd.DataFrame)

        # Read Filenot found
        fp2 = "tests/data/io/testnot.pickle"
        with pytest.raises(FileNotFoundError):
            df1 = io.read(fp2)

        # Write
        fp2 = "tests/data/io/test2.pickle"
        os.remove(fp2)
        io.write(fp2, df1)
        assert os.path.exists(fp2)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_dcm(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # Read
        fp1 = "tests/data/io/test1.dcm"
        io = IOFactory.create("dcm")
        d1 = io.read(fp1)
        assert isinstance(d1, tf.Tensor)

        # Read Filenot found
        fp2 = "tests/data/io/testnot.dcm"
        with pytest.raises(FileNotFoundError):
            d1 = io.read(fp2)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_nii(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
        # Read
        fp1 = "tests/data/io/test1.nii"
        io = IOFactory.create("nii")
        d1 = io.read(fp1)
        assert isinstance(d1, np.ndarray)

        # Read Filenot found
        fp2 = "tests/data/io/testnot.dcm"
        with pytest.raises(FileNotFoundError):
            d1 = io.read(fp2)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
