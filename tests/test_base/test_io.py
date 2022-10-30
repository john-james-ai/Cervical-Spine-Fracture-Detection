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
# Created    : Saturday October 29th 2022 06:26:39 am                                              #
# Modified   : Saturday October 29th 2022 08:49:12 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest
import logging
import logging.config

# Enter imports for modules and classes being tested here
from csf.base.io import IOFactory, CSVIO

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.io
class TestIO:
    def test_csv(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        basedir = "tests/data"
        csv_filepath1 = "train.csv"
        csv_filepath2 = "train2.csv"
        filepath = os.path.join(basedir, csv_filepath1)
        assert os.path.exists(filepath)

        iof = IOFactory(basedir=basedir)
        io = iof.create(ftype="csv")
        assert isinstance(io, CSVIO)
        data = io.read(csv_filepath1)
        print(data.head())
        io.write(filepath=csv_filepath2, data=data)

        assert isinstance(data, pd.DataFrame)
        filepath = os.path.join(basedir, csv_filepath2)
        assert os.path.exists(filepath)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_yaml(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        basedir = ""
        yaml1 = "config/data.yml"
        yaml2 = "tests/data/data.yml"
        filepath = os.path.join(basedir, yaml2)

        iof = IOFactory(basedir=basedir)
        io = iof.create("yml")
        data = io.read(yaml1)
        io.write(filepath=yaml2, data=data)

        assert isinstance(data, dict)
        assert os.path.exists(filepath)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_pickle(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        filepath = "tests/data/train.csv"
        df1 = pd.read_csv(filepath)

        basedir = "tests/data"
        df1path = "io/df1.pickle"
        filepath = os.path.join(basedir, df1path)

        iof = IOFactory(basedir=basedir)
        io = iof.create("pkl")
        io.write(filepath=df1path, data=df1)
        assert os.path.exists(filepath)

        df2 = io.read(df1path)
        assert isinstance(df2, pd.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_nii(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        basedir = "tests/data/"
        nii1 = "test1.nii"
        nii2 = "test2.nii"

        iof = IOFactory(basedir=basedir)
        io = iof.create("nii")
        data = io.read(nii1)
        assert isinstance(data, np.ndarray)

        io.write(filepath=nii2, data=data)
        filepath = os.path.join(basedir, nii1)
        assert os.path.exists(filepath)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_dicom(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        basedir = "tests/data"
        dcm1 = "test1.dcm"

        iof = IOFactory(basedir=basedir)
        io = iof.create("dcm")
        data = io.read(dcm1)
        assert isinstance(data, tf.Tensor)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
