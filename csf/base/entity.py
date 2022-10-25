#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /entity.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 25th 2022 08:39:21 am                                               #
# Modified   : Tuesday October 25th 2022 11:33:13 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
from abc import ABC
from datetime import datetime


# ------------------------------------------------------------------------------------------------ #
class Entity(ABC):
    """Base class for the domain / entity objects (Filesets, Datasets, Analyses, Trials, & Experiments)

    Args:
        name (str): Human readible name for the entity.
        description (str): Brielfy describes the entity and its purpose.
        type (str): One of ['fileset','dataset','analysis','model']
    """

    def __init__(self, name: str, type: str, description: str = None) -> None:
        self._name = name
        self._type = type
        self._description = description
        self._created = datetime.now()
        self._updated = datetime.now()
        self._entity = os.get_env("WANDB_ENTITY")

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

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

    @property
    def updated(self) -> datetime:
        return self._updated

    def generate_artifact(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in [None, ""]}
