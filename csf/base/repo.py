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
# Created    : Saturday October 29th 2022 01:00:32 pm                                              #
# Modified   : Wednesday November 2nd 2022 05:41:54 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Repository Module"""
import os
from datetime import datetime
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Any
from glob import glob

from csf.base.entity import Entity, Table
from csf.base.io import IOFactory


# ------------------------------------------------------------------------------------------------ #


class Registry:
    """Registry for the Generic Repo."""

    def __init__(self, io_factory: IOFactory, repo_name: str) -> None:
        self._io_factory = io_factory
        self._io = io_factory.create("pkl")

        # Set registry filepath
        load_dotenv()
        base_dir = os.getenv("BASE_DATA_DIR")
        registry_dir = "repo_registries"
        filename = repo_name.lower() + ".pkl"
        self._registry_filepath = os.path.join(base_dir, registry_dir, filename)

    @property
    def registry_filepath(self) -> str:
        return self._registry_filepath

    def create(self, entity: Entity) -> None:
        if os.path.exists(self._registry_filepath):
            registry = self._io.read(filepath=self._registry_filepath)
        else:
            registry = {}

        if entity.name in registry.items():
            raise FileExistsError(
                "{}:{} already exists.".format(entity.__class__.__name__, entity.name)
            )
        else:
            registry[entity.name] = entity
            self._io.write(filepath=self._registry_filepath, data=registry)

    def read(self, name: str) -> Entity:
        try:
            registry = self._io.read(filepath=self._registry_filepath)
            return registry[name]
        except KeyError:
            raise FileNotFoundError("Entity named {} not found.".format(name))

    def update(self, entity: Entity) -> None:
        registry = self._io.read(filepath=self._registry_filepath)
        entity.modified = datetime.now()
        registry[entity.name] = entity
        self._io.write(filepath=self._registry_filepath, data=registry)

    def delete(self, name: str) -> None:
        registry = self._io.read(filepath=self._registry_filepath)
        try:
            del registry[name]
            self._io.write(filepath=self._registry_filepath, data=registry)
        except KeyError:
            raise FileNotFoundError("Entity named {} not found.".format(name))

    def exists(self, name) -> bool:
        registry = self._io.read(filepath=self._registry_filepath)
        return name in registry.keys()

    def count(self) -> int:
        registry = self._io.read(filepath=self._registry_filepath)
        return len(registry)

    def as_list(self) -> list:
        registry = self._io.read(filepath=self._registry_filepath)
        return [name for name in registry.keys()]


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Generic Repository."""

    @abstractmethod
    def load(self, directory: str) -> None:
        pass

    def create(self, entity: Entity, *args, **kwargs) -> None:
        if self._is_valid_for_create(entity):
            self._write_entity(entity)

    def read(self, name) -> Any:
        """Return a entity object by name from the repo"""
        entity = self._registry.read(name)
        return self._io.read(filepath=entity.path)

    def update(self, entity: Entity, *args, **kwargs) -> None:
        """Updates an existing member of the repo."""
        if self._is_valid_for_update_delete(entity):
            self._write_entity(entity)

    def delete(self, name, *args, **kwargs) -> None:
        """Deletes the item with the designated name."""
        entity = self._registry.read(name)
        os.remove(entity.path)
        self._registry.delete(name)

    def exists(self, name) -> bool:
        """Returns True if the named entity's data exists."""
        return self._registry.exists(name)

    def as_list(self) -> list:
        """Returns a list of the entities in the Repo by name"""
        return self._registry.as_list()

    def _write_entity(self, entity: Entity) -> None:
        # Extract the data from the entity object
        data = entity.data
        entity.data = None
        # Set the path
        entity.path = self._set_path(entity.name)
        # Register the object
        self._registry.create(entity)
        # Write the data to file
        self._io.write(filepath=entity.path, data=data)

    def _is_valid_for_create(self, entity: Entity) -> bool:
        """Ensures the entity can be created."""
        return self._file_not_exist(entity) and self._registration_not_exist(entity)

    def _is_valid_for_update_delete(self, entity: Entity) -> bool:
        return self._file_exist(entity) and self._registration_exists(entity)

    @abstractmethod
    def _set_path(self, name: str) -> str:
        pass

    # -------------------------------------------------------------------------------------------- #
    #                                       VALIDATION                                             #
    # -------------------------------------------------------------------------------------------- #
    def _file_exist(self, entity: Entity) -> bool:
        if not os.path.exists(entity.path):
            raise FileNotFoundError(
                "{}:{} does not exist.".format(entity.__class__.__name__, entity.name)
            )
        else:
            return True

    def _file_not_exist(self, entity: Entity) -> bool:
        if os.path.exists(entity.path):
            raise FileExistsError(
                "{}:{} already exists.".format(entity.__class__.__name__, entity.name)
            )
        else:
            return True

    def _registration_exists(self, entity: Entity) -> bool:
        if not self._registry.exists(entity.name):
            raise FileNotFoundError(
                "{}:{} does not exist in the registry.".format(
                    entity.__class__.__name__, entity.name
                )
            )
        else:
            return True

    def _registration_not_exist(self, entity: Entity) -> bool:
        if self._registry.exists(entity.name):
            raise FileExistsError(
                "{}:{} already exists in the registry.".format(
                    entity.__class__.__name__, entity.name
                )
            )
        else:
            return True


# ------------------------------------------------------------------------------------------------ #
class TableRepo(Repo):
    """Repository for Table or DataFrame objects.

    Args:
        io_factory (IOFactory): The class responsible for providing access to IO services.
        registry (Registry): Class responsible for registration of TableRepo objects.

    """

    __input = "input"
    __working = "working"
    __format = "csv"

    def __init__(self, io_factory: IOFactory, registry: Registry, *args, **kwargs) -> None:
        self._io_factory = io_factory.create(TableRepo.__format)
        self._registry = registry(io_factory=io_factory, repo_name=self.__class__.__name)

        load_dotenv(".env")
        self._base_dir = os.getenv("BASE_DATA_DIR")
        self._input_dir = os.path.join(self._base_dir, TableRepo.__input)
        self._working_dir = os.path.join(self._base_dir, TableRepo.__working)

    def load(self, directory: str) -> None:
        """Loads existing entity objects into the registry"""
        pattern = directory + "\\*." + TableRepo.__format
        files = glob(pattern)
        for file in files:
            entity = Table(
                name=os.path.splitext(os.path.basename(file))[0], type="dataset", path=file
            )
            self._registry.create(entity)

    def _set_path(self, name: str) -> str:
        """Sets the path to the working directory. All new objects go in working directory"""
        filename = name + "." + TableRepo.__format
        return os.path.join(self._base_dir, TableRepo.__working, filename)
