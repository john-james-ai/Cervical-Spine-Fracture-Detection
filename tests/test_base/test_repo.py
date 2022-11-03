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
# Modified   : Thursday November 3rd 2022 12:37:23 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import inspect
import pytest
import pandas as pd
from glob import glob
import shutil
import logging
import logging.config
from datetime import datetime
import wandb

from csf.base.repo import Registry, TableRepo
from csf.base.io import IOFactory
from csf.base.entity import Table

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.registry
class TestTableRegistry:

    io_factory = IOFactory()
    repo_name = "TableRepo"
    reg = Registry(io_factory=io_factory, repo_name=repo_name)

    def entity_test(self, entity):
        assert entity.name in entity.id
        assert entity.data is None
        assert isinstance(entity.created, datetime)
        assert isinstance(entity.obj, wandb.data_types.Table)
        assert isinstance(entity.size, int)
        assert isinstance(entity.rows, int)
        assert isinstance(entity.cols, int)
        assert isinstance(entity.nulls, int)

    def test_filepath(self, caplog):
        assert TestTableRegistry.reg.registry_filepath == os.path.join(
            "tests/data/repo/repo_registries/tablerepo.pkl"
        )

    def test_reset(self, caplog):
        TestTableRegistry.reg.reset()
        assert TestTableRegistry.reg.count() == 0

    def test_create_read(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for table in tables:
            TestTableRegistry.reg.create(entity=table)
        assert os.path.exists(TestTableRegistry.reg.registry_filepath)

        for table in tables:
            with pytest.raises(FileExistsError):
                TestTableRegistry.reg.create(entity=table)

        for table in tables:
            entity = TestTableRegistry.reg.read(name=table.name)
            self.entity_test(entity)

        TestTableRegistry.reg.print()

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_update(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for i, table in enumerate(tables):
            if i % 2 == 0:
                table.description = table.description + "updated description #" + str(i)
                TestTableRegistry.reg.update(entity=table)
                table = TestTableRegistry.reg.read(name=table.name)

        for i, table in enumerate(tables):
            if i % 2 == 0:
                table = TestTableRegistry.reg.read(name=table.name)
                assert isinstance(table, Table)
                assert "updated_description" in table.description

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_delete(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for i, table in enumerate(tables):
            if i % 2 == 0:
                TestTableRegistry.reg.delete(name=table.name)
                with pytest.raises(KeyError):
                    TestTableRegistry.reg.read(name=table.name)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_exists(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for i, table in enumerate(tables):
            if i % 2 == 0:
                assert not TestTableRegistry.reg.exists(name=table.name)
            else:
                assert TestTableRegistry.reg.exists(name=table.name)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_count(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        assert TestTableRegistry.reg.count() == 2

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))


@pytest.mark.repo
class TestTableRepo:

    io_factory = IOFactory()
    repo_name = "TableRepo"
    repo = TableRepo(io_factory=io_factory, registry=Registry)

    def test_setup(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        working = "tests/data/repo/working"
        registry = "tests/data/repo/repo_registries"

        shutil.rmtree(working, ignore_errors=True)
        shutil.rmtree(registry, ignore_errors=True)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_load(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        TestTableRepo.repo.reset()

        tables = glob("tests/data/repo/*.csv")

        TestTableRepo.repo.load(directory="tests/data/repo")

        for table in tables:
            name = os.path.splitext(os.path.basename(table))[0]
            table = TestTableRepo.repo.read(name)
            assert isinstance(table, Table)
            assert isinstance(table.data, pd.DataFrame)
        assert TestTableRepo.repo.count() == 4

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_create(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for table in tables:
            TestTableRepo.repo.create(table)
            TestTableRepo.repo.print()
            table = TestTableRepo.repo.read(table.name)
            assert isinstance(table, Table)
            assert isinstance(table.data, pd.DataFrame)
        assert TestTableRepo.repo.count() == 9

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_find(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        table = tables[0]
        table.name = "number_3"
        table.path = None
        table = TestTableRepo.repo.find(table)
        assert table.path == "tests/data/repo/working/test_table_number_3.csv"

        table.name = "number_1"
        table = TestTableRepo.repo.find(table)
        assert table.path == "tests/data/repo/working/test_table_number_1.csv"

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_delete(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for i, table in enumerate(tables):
            if i % 2 == 0:
                TestTableRepo.repo.delete(table.name)

        for i, table in enumerate(tables):
            if i % 2 == 0:
                with pytest.raises(FileNotFoundError):
                    TestTableRepo.repo.read(table.name)
            else:
                table = TestTableRepo.repo.read(table.name)
                assert isinstance(table, Table)
                assert isinstance(table.data, pd.DataFrame)

        assert TestTableRepo.repo.count() == 6

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_archive(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for i, table in enumerate(tables):
            if i % 2 != 0:
                table = TestTableRepo.repo.find(table)
                print(table.path)
                table = TestTableRepo.repo.archive(table)

                assert table.is_archived

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_exists(self, tables, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        for i, table in enumerate(tables):
            if i % 2 == 0:
                assert not TestTableRepo.repo.exists(table.name)
            else:
                assert TestTableRepo.repo.exists(table.name)

        assert TestTableRepo.repo.count() == 6

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
