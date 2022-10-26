#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /dao.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 25th 2022 01:53:40 pm                                               #
# Modified   : Tuesday October 25th 2022 07:57:49 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Union, Any

# ------------------------------------------------------------------------------------------------ #

class IO(ABC):
    """IO Base class"""

    @abstractmethod
    @classmethod
    def load(clf, filepath: str) -> Union[Any]:
        pass

    @abstractmethod
    @classmethod
    def save(clf, data: Union[Any], filepath: str) -> None:
        pass

# ------------------------------------------------------------------------------------------------ #

class CSV_IO(IO):
    """IO for Tabular Files"""
    @classmethod
    def load(cls, filepath: str) ->