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
# Modified   : Tuesday October 25th 2022 02:21:54 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Data Task Configurator

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
class FileRepoConfig:
    """
    FileRepoConfig configuration object.

    Args:
        name (str): A human readible name for the FileRepo.
        uri (str): URI to the base folder in the storage platform.
        id (str): A unique identifier for the FileRepo
        type (str): Type of Repo
        entity (str): The username associated with the FileRepo
        description (str): Brief description of the FileRepo.
        created (datetime): Datetime the configuration was created.

    """

    name: str
    uri: str
    id: str = None
    description: Optional[float] = None
    entity: str = None
    created: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.id = self.name + "-" + wandb.util.generate_id()
        self.entity = os.get_env("WANDB_ENTITY")

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Config object."""
        return {k: v for k, v in self.__dict__.items()}
