#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /event.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 25th 2022 08:40:24 am                                               #
# Modified   : Tuesday October 25th 2022 12:06:02 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

from csf.utils.time import time_format
from csf.base.entity import Entity
from csf.base.operator import Operator


# ------------------------------------------------------------------------------------------------ #


class Event(ABC):
    """Base class for events, such as Projects, Experiments and Trials

    Args:
        name (str): Human readible name for the entity.
        description (str): Brielfy describes the entity and its purpose.
        type (str): One of ['project','trial','experiment']
    """

    def __init__(self) -> None:
        self._name = None
        self._type = None
        self._description = None
        self._created = datetime.now()
        self._started = None
        self._completed = None
        self._return_code = 0
        self._entity = os.get_env("WANDB_ENTITY")
        self._input = None
        self._output = None
        self._operators = []
        self._force = False

    def add_operator(self, operator: Operator) -> None:
        self._operators.append(operator)

    def reset(self) -> None:
        self._operators = []
        self._name = None
        self._type = None
        self._description = None
        self._input = None
        self._output = None
        self._force = False

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
    def input(self) -> Entity:
        return self._input

    @input.setter
    def input(self, input: Entity) -> None:
        self._input = input

    @property
    def output(self) -> Entity:
        return self._output

    @output.setter
    def output(self, output: Entity) -> None:
        self._output = output

    @property
    def force(self, force: bool) -> None:
        self._force = force

    @force.setter
    def force(self, force: bool) -> None:
        self._force = force

    @property
    def result(self) -> Union[Entity]:
        return self._result

    @property
    def created(self) -> datetime:
        return self._created

    @property
    def started(self) -> datetime:
        return self._started

    @property
    def completed(self) -> datetime:
        return self._completed

    @property
    def duration(self) -> str:
        return self._duration

    @property
    def return_code(self) -> datetime:
        return self._return_code

    def generate_artifact(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in [None, ""]}

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        pass

    def run(self) -> None:
        if self._force or not self.is_complete:

            self._setup()
            self._run()
            self._teardown()

    def _setup(self) -> None:
        self._started = datetime.now()
        self._setup_wandb()

    @abstractmethod
    def _setup_wandb(self) -> None:
        pass

    @abstractmethod
    def _run(self) -> None:
        pass

    def _teardown(self) -> None:
        self._teardown_wandb()
        self._completed = datetime.now()
        self._duration = time_format((self._completed - self._started).total_seconds())
        self._completed = True

    @abstractmethod
    def _teardown_wandb(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #


class EventBuilder(ABC):

    """Builds an event, data, analysis, modeling, or inference event from configuration.

    Args:
        config (Config): Event configuration
    """

    def __init__(self, config_name: str) -> None:
        self._config_name = config_name
        self._config = None
        self._event = None

    @abstractmethod
    @property
    def event(self) -> Event:
        pass

    @abstractmethod
    def build_config(self) -> None:
        pass

    @abstractmethod
    def build_inputs(self) -> None:
        pass

    @abstractmethod
    def build_outputs(self) -> None:
        pass

    @abstractmethod
    def build_tasks(self) -> None:
        pass
