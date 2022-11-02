#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /repo.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 29th 2022 01:00:32 pm                                              #
# Modified   : Sunday October 30th 2022 06:07:10 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Repository Module"""
import os
from dotenv import load_dotenv
from typing import Any
from glob import glob

from csf.base.io import IOFactory

# ------------------------------------------------------------------------------------------------ #


class RepoRegistry(type):
    """Maintains the collection of Repos, not to be confused with an inventory of Repo contents."""

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RepoRegistry but in child classes will another type of Repo
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


# ------------------------------------------------------------------------------------------------ #
class Repo(metaclass=RepoRegistry):
    """Abstract base class for Repositories."""

    def add(self, name, *args, **kwargs) -> None:
        """Add an item to the repository."""
        raise NotImplementedError()

    def get(self, name) -> Any:
        """Return an item by name from the repo"""
        raise NotImplementedError()

    def update(self, name, *args, **kwargs) -> None:
        """Updates an existing member of the repo."""
        raise NotImplementedError()

    def remove(self, name, *args, **kwargs) -> None:
        raise NotImplementedError()


# ------------------------------------------------------------------------------------------------ #
class FileRepo(Repo):
    """Handles File and Persistence Operations."""

    def __init__(self, io_factory: IOFactory, *args, **kwargs) -> None:
        load_dotenv(".env")

        self._base_dir = os.getenv("BASE_DATA_DIR")
        self._io_factory = io_factory(self._base_dir)
        self._csv_io = self._io_factory.create("csv")
        self._nib_io = self._io_factory.create("nii")
        self._dcm_io = self._io_factory.create("dcm")

    def add(self, name: str, type: str, data: Any) -> None:
        """Add an item to the repository."""

    def get(self, name) -> Any:
        """Return an item by name from the repo"""

    def update(self, name, *args, **kwargs) -> None:
        """Updates an existing member of the repo."""

    def remove(self, name, *args, **kwargs) -> None:
        """Removes a file from the file repository."""

    def listfiles(self, pattern: str) -> list:
        return glob(pattern)

    def _get_path(self, name: str) -> str:
        """Returns a path to a resource, given its name."""
        pattern = name + "*"
        result = glob(pathname=pattern, root_dir=self._base_dir)
