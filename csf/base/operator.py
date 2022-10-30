#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /operator.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 27th 2022 02:34:47 pm                                              #
# Modified   : Saturday October 29th 2022 04:26:04 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Abstract base class for tasks."""
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime, timedelta

from csf.base.io import IOFactory

# ------------------------------------------------------------------------------------------------ #


class Operator(ABC):
    """Abstract class for task classes

    Args:
        name (str): The name for the operator that distinguishes it in the pipeline.
        force (str): True suppresses execution if
    """

    def __init__(self, name: str, params: dict, *args, **kwargs) -> None:
        self._name = name
        self._params = params

        self._created = datetime.now()
        self._started = None
        self._ended = None
        self._duration = None
        self._skipped = False

    def __str__(self) -> str:
        return f"Task: {self.__class__.__name__}\n\tName: {self._name}\n\tParams: {self._params}"

    def __repr__(self) -> str:
        return f"{self.__class__}(name={self._name}, params={self._params})"

    def execute(self, data: Any = None, context: dict = None) -> Any:
        """Call setup() and teardown() before and after in subclasses."""
        pass

    @abstractmethod
    def _execute(self, data: Any = None, context: dict = None) -> Any:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def force(self) -> bool:
        return self._force

    @property
    def created(self) -> datetime:
        return self._created

    @property
    def started(self) -> datetime:
        return self._started.strftime("%m/%d/%Y at %H:%M:%S")

    @property
    def ended(self) -> datetime:
        return self._ended.strftime("%m/%d/%Y at %H:%M:%S")

    @property
    def duration(self) -> str:
        return str(timedelta(seconds=self._duration))

    @property
    def status(self) -> bool:
        return self._status

    @property
    def skipped(self) -> bool:
        return self._skipped

    def setup(self) -> None:
        self._started = datetime.now()

    def teardown(self) -> None:
        self._ended = datetime.now()
        self._duration = (self._ended - self._started).total_seconds()


# ------------------------------------------------------------------------------------------------ #
class DataOperator(Operator):
    """Operators used to process data."""

    def __init__(self, name: str, params: dict, io_factory: IOFactory, *args, **kwargs) -> None:
        super().__init__(name=name, params=params)
        self._io_factory = io_factory
