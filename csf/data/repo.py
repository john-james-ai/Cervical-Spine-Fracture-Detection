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
# Created    : Tuesday October 25th 2022 08:38:51 am                                               #
# Modified   : Tuesday October 25th 2022 07:57:39 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
from datetime import datetime
from csf.base.entity import Entity
from csf.base.config import Config
from csf.base.repo import Repo, RepoBuilder


# ------------------------------------------------------------------------------------------------ #
class FileRepo(Repo):
    """FileRepo class encapsulating persistence for Filesets

    Args:
        name (str): Human readible name for the registry
        description (str): Brielfy describes the entity and its purpose.
        type (str): One of ['data','analysis','experiment','inference']
    """


    def load(self, name) -> Fileset:
        pass

    def getall(self) -> [Entity]:
        pass

    def add(self, entity: Entity) -> None:
        pass

    def update(self, entity: Entity) -> None:
        pass

    def delete(self, entity: Entity) -> None:
        pass

    def registry_lookup(self, name: str) ->


# ------------------------------------------------------------------------------------------------ #
class FileRepoBuilder(RepoBuilder):
    """Abstract builder for object repositories.

    Args:
        config (Config): Repository builder config

    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._repo = self.reset()

    @property
    @abstractmethod
    def reset(self) -> FileRepo:
        return FileRepo()

    @abstractmethod
    def build_access(self) -> None:
        pass

    @abstractmethod
    def build_storage(self) -> None:
        self._config.uri

    @abstractmethod
    def build_io(self) -> None:
        pass

    @abstractmethod
    def build_registry(self) -> None:
        pass
