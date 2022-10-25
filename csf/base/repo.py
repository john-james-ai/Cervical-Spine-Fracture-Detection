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
# Modified   : Tuesday October 25th 2022 12:07:16 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
from abc import ABC, abstractmethod
from datetime import datetime
from csf.base.entity import Entity
from csf.base.config import Config


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Base class for Domain Repository objects, such as Data,Analyses, Configs, and Experiments.

    Args:
        name (str): Human readible name for the registry
        description (str): Brielfy describes the entity and its purpose.
        type (str): One of ['data','analysis','experiment','inference']
    """

    def __init__(self) -> None:
        self._name = None
        self._type = None
        self._description = None
        self._entity = os.get_env("WANDB_ENTITY")
        self._created = datetime.now()
        self._n_items = 0
        self._registry_filepath = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, type: str) -> None:
        self._type = type

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, description: str) -> None:
        self._updated = datetime.now()
        self._description = description

    @property
    def created(self) -> datetime:
        return self._created

    @abstractmethod
    def get(self, name: str) -> Entity:
        pass

    @abstractmethod
    def getall(self) -> [Entity]:
        pass

    @abstractmethod
    def add(self, entity: Entity) -> None:
        pass

    @abstractmethod
    def update(self, entity: Entity) -> None:
        pass

    @abstractmethod
    def delete(self, entity: Entity) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
class RepoBuilder(ABC):
    """Abstract builder for object repositories.

    Args:
        config (Config): Repository builder config

    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._repo = None
        self._force = False

    @property
    @abstractmethod
    def repo(self) -> Repo:
        pass

    @abstractmethod
    def build_authorization(self) -> None:
        pass

    @abstractmethod
    def build_io(self) -> None:
        pass

    @abstractmethod
    def build_registry(self) -> None:
        pass

    @property
    def force(self) -> bool:
        return self._force

    @force.setter
    def force(self, force: bool) -> None:
        self._force = force
