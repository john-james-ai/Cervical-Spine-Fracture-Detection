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
# Created    : Thursday October 27th 2022 08:58:37 pm                                              #
# Modified   : Friday October 28th 2022 08:09:36 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Package Repositories."""
import os
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any

# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    def __init__(self, basedir: str) -> None:
        self._basedir = basedir

    @abstractmethod
    def create(self, name: str, **kwargs) -> None:
        pass

    @abstractmethod
    def read(self, name: str) -> Any:
        pass

    @abstractmethod
    def update(self, name: str, data: Any) -> None:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass

    def _get_path(self, path: str) -> str:
        return os.path.join(self._basedir, path)

# ------------------------------------------------------------------------------------------------ #
class FileRepo(Repo):


    __registry = 

    def __init__(self, basedir: str) -> None:
        super().__init__(basedir=basedir)
        self._load_registry()

    def create(self, name: str, path: str, description: str = None) -> None:
        """Creates a file or directory with the given name, path, and optional description."""

    def list_filepath(self, name: str) -> list:
        """Provides a list of files in the named directory."""


