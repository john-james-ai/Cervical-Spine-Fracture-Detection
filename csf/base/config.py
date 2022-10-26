#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /config.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 07:36:55 am                                                #
# Modified   : Tuesday October 25th 2022 01:04:13 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Configuration Base Class

This module has been adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
import wandb

# ------------------------------------------------------------------------------------------------ #


@dataclass
class Config:
    """
    Base class for all configuration objects.

    Args:
        name (str): A human readible name for the configuration.
        event_name (str): The name of the event to which the configuration is applied
        id (str): A unique identifier for the configuration.
        type (str): Type of config
        entity (str): The username associated with the configuration.
        description (str): Brief description of the configuration.
        created (datetime): Datetime the configuration was created.

    """

    name: str
    event_name: str
    id: str = None
    entity: str = None
    description: Optional[float] = None
    created: datetime = field(default_factory=datetime.now)

    IMMUTABLE_TYPES: tuple = (str, int, float, bool, type(None))
    SEQUENCE_TYPES: tuple = (list, tuple)

    def __post_init__(self) -> None:
        self.id = self.name + "-" + wandb.util.generate_id()
        self.entity = os.get_env("WANDB_ENTITY")

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Config object."""
        return {k: self._export_config(v) for k, v in self.__dict__.items()}

    @classmethod
    def _export_config(cls, v):
        """Returns v with Configs converted to dicts, recursively."""
        if isinstance(v, cls.IMMUTABLE_TYPES):
            return v
        elif isinstance(v, cls.SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, dict):
            return {k2: cls._export_config(v2) for k2, v2 in v.items()}
        else:
            raise TypeError("Unknown type: {!r}".format(type(v)))
