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
# Created    : Tuesday October 25th 2022 08:40:14 am                                               #
# Modified   : Tuesday October 25th 2022 12:06:56 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from typing import Union, Any
from datetime import datetime
import pandas as pd

# ------------------------------------------------------------------------------------------------ #


class Operator(ABC):
    """Base class for operators and objects that perform atomic units of work."""

    def __init__(self, name: str, params: dict) -> None:
        self._name = name
        self._params = params
        self._created = datetime.now()
        self._started = None
        self._completed = None
        self._return_code = None

    @abstractmethod
    def execute(self, data: Union[pd.DataFrame, dict], context: Any) -> Union[pd.DataFrame, dict]:
        pass
