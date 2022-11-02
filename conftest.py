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
# Modified   : Wednesday November 2nd 2022 05:04:32 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pytest
from csf.base.entity import Table
import pandas as pd

# ------------------------------------------------------------------------------------------------ #


@pytest.fixture
def table():
    data = pd.read_csv("tests/data/io/test1.csv", index_col=False)
    return Table(name="test_table", data=data, type="dataset", description="test_table_described")
