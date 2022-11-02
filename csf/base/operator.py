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
# Modified   : Tuesday November 1st 2022 03:48:16 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Abstract base class for tasks."""
from abc import ABC, abstractmethod
from typing import Any, Union

# ------------------------------------------------------------------------------------------------ #


class Operator(ABC):
    """Abstract base class for Operator sub-classes. Adds kwargs to subclass as attributes

    Setup and teardown methods are provided for pipeline classes which track metadata.

    Args:
        name (str): The name for the operator that distinguishes it in the pipeline.
        operation (Operation): The class the performs the operation.
    """

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._skipped = False

    def __str__(self) -> str:
        return f"Operation: {self.__class__.__name__}\n\tAttributes: {self.__dict__.items()}"

    def __repr__(self) -> str:
        return f"Operation: {self.__class__.__name__}\n\tAttributes: {self.__dict__.items()}"

    @abstractmethod
    def execute(self, data: Any = None, context: dict = None) -> Any:
        """Call setup() and teardown() before and after in subclasses."""
        pass

    @abstractmethod
    def return_code(self) -> Union[bool, str]:
        """Returns a return code as bool or string"""
        pass
