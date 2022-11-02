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
# Created    : Tuesday November 1st 2022 10:28:59 pm                                               #
# Modified   : Wednesday November 2nd 2022 05:45:38 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Module defining the base Entity Type."""
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
import wandb
from typing import Any

from csf.base import IMMUTABLE_TYPES, SEQUENCE_TYPES

# ------------------------------------------------------------------------------------------------ #


@dataclass(frozen=True)
class Entity(ABC):
    """Entity. Base class for File, Model, and all entities that have a lifecycle.

    Args:
        id (str): The Artifact's id.
        name (str): A human readible name for a fileset that doesnt contain the file extension
        data (Any): The content of the Entity object.
        obj (Any): The wandb data type
        type (str): The type of the artifact, which is used to organize and differentiate artifacts.
            Common types include dataset or model, but you can use any string containing letters, numbers,
            underscores, hyphens, and dots. Set by the Entity sub-class.
        description (str):  Optional free text that offers a description of the artifact. The description is
            markdown rendered in the UI, so this is a good place to place tables, links, etc.
        created (datetime): The datetime when the entity was created.
        modified (datetime): The datetime when the entity was updated.
        path: (str): The relative path to the entity, including base directory. This is assigned by the repo.
    """

    name: str
    data: Any
    id: str = None
    obj: wandb.data_types = None
    type: str = None
    description: str = None
    created: datetime = None
    modified: datetime = None
    path: str = None

    def __post_init__(self) -> None:
        self.id = self.name + "-" + wandb.util.generate_id()
        self.created = datetime.now()
        self.modified = datetime.now()

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Config object."""
        return {k: self._export_config(v) for k, v in self.__dict__.items()}

    @classmethod
    def _export_config(cls, v):
        """Returns v with Configs converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, datetime):
            return v.strftime("%m/%d/%Y, %H:%M")
        elif isinstance(v, dict):
            return {kk: cls._export_config(vv) for kk, vv in v}
        else:
            return "Mutable Object"


@dataclass
class Table(Entity):
    """Table Entity.

    Args:
        id (str): The Artifact's id.
        name (str): A human readible name for a fileset that doesnt contain the file extension
        data (pd.DataFrame): The table content as a pandas DataFrame object.
        obj (Any): The wandb data type
        type (str): The type of the artifact, which is used to organize and differentiate artifacts.
            Common types include dataset or model, but you can use any string containing letters, numbers,
            underscores, hyphens, and dots. Set by the Entity sub-class.
        description (str):  Optional free text that offers a description of the artifact. The description is
            markdown rendered in the UI, so this is a good place to place tables, links, etc.
        created (datetime): The datetime when the entity was created.
        modified (datetime): The datetime when the entity was updated.
        path: (str): The relative path to the entity, including base directory. This is assigned by the repo.
        size: (int): The size of the Table in bytes
        rows: (int): The number of rows in the Table
        cols: (int): The number of columns in the Table
    """

    size: int = 0
    rows: int = 0
    cols: int = 0
    nulls: int = 0

    def __post_init__(self) -> None:
        self.id = self.name + "-" + wandb.util.generate_id()
        self.created = datetime.now()
        self.modified = datetime.now()
        self.obj = wandb.WBValue = wandb.data_types.Table
        self.size = self.data.memory_usage(index=True, deep=True).sum()
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
        self.nulls = self.data.isna().sum().sum()
