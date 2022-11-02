#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /test_repo.py                                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday November 2nd 2022 04:53:42 am                                             #
# Modified   : Wednesday November 2nd 2022 05:40:49 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import inspect
import pytest
import logging
import logging.config
from datetime import datetime
import wandb

from csf.base.repo import Registry  # , TableRepo
from csf.base.io import IOFactory

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.registry
class TestTableRegistry:
    def entity_test(self, entity):
        assert entity.name in entity.id
        assert entity.data is None
        assert isinstance(entity.created, datetime)
        assert isinstance(entity.modified, datetime)
        assert isinstance(entity.obj, wandb.data_types.Table)
        assert isinstance(entity.size, int)
        assert isinstance(entity.rows, int)
        assert isinstance(entity.cols, int)
        assert isinstance(entity.nulls, int)

    def test_create_read(self, table, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        io_factory = IOFactory()
        repo_name = "TableRepo"

        reg = Registry(io_factory, repo_name)
        reg.create(entity=table)
        assert os.path.exists(reg.registry_filepath)

        table_reg = reg.read(name=table.name)
        self.test_entity(table_reg)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_update(self, table, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        io_factory = IOFactory()
        repo_name = "TableRepo"

        table.name = "2nd_test_table"
        table.path = "tests/data/repo/input/2nd_test_table.csv"

        reg = Registry(io_factory, repo_name)
        reg.create(entity=table)

        table_reg = reg.read(name=table.name)
        self.entity_test(table_reg)
        assert table_reg.name == "2nd_test_table"
        assert table_reg.path == "tests/data/repo/input/2nd_test_table.csv"

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_delete(self, table, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        io_factory = IOFactory()
        repo_name = "TableRepo"

        table.name = "3rd_test_table"
        table.path = "tests/data/repo/input/3rd_test_table.csv"

        reg = Registry(io_factory, repo_name)
        reg.create(entity=table)
        reg.delete(entity=table.name)

        with pytest.raises(FileNotFoundError):
            reg.read(name=table.name)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_exists(self, table, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        io_factory = IOFactory()
        repo_name = "TableRepo"

        table.name = "2nd_test_table"
        table.path = "tests/data/repo/input/2nd_test_table.csv"

        reg = Registry(io_factory, repo_name)
        assert reg.exists(entity=table.name)

        table.name = "3rd_test_table"
        assert not reg.exists(entity=table.name)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_count(self, table, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        io_factory = IOFactory()
        repo_name = "TableRepo"

        reg = Registry(io_factory, repo_name)
        assert reg.count() == 2

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_as_list(self, table, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        io_factory = IOFactory()
        repo_name = "TableRepo"

        reg = Registry(io_factory, repo_name)
        assert len(reg.as_list()) == 2

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
