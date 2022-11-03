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
# Modified   : Thursday November 3rd 2022 12:41:46 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Repository Module"""
import os
import numpy as np
from difflib import SequenceMatcher
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import pandas as pd
from glob import glob
import shutil
import logging
import logging.config

from csf.base.entity import Entity, Table
from csf.base.io import IOFactory

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class Registry:
    """Registry for the Generic Repo."""

    def __init__(self, io_factory: IOFactory, repo_name: str) -> None:
        self._io_factory = io_factory
        self._io = io_factory.create("pkl")

        # Set registry filepath
        load_dotenv()
        base_dir = os.getenv("BASE_DATA_DIR")
        registry_dir = "registries"
        filename = repo_name.lower() + ".pkl"
        self._registry_filepath = os.path.join(base_dir, registry_dir, filename)

    @property
    def registry_filepath(self) -> str:
        return self._registry_filepath

    def reset(self) -> None:
        """Resets the registry by deleting it."""
        if os.path.exists(self._registry_filepath):
            os.remove(self._registry_filepath)
            logger.info("Registry at {} has been deleted.".format(self._registry_filepath))

    def create(self, entity: Entity, force: bool = False) -> None:
        """Saves the entity in the registry, after removing the data."""
        entity.data = None  # Remove data before storing registration info.

        if force or not self.exists(name=entity.name):
            registry = self._read_registry()
            registry[entity.name] = entity
            self._write_registry(registry)
        elif self.exists(name=entity.name):
            msg = "Entity {} already exists.".format(entity.name)
            logger.error(msg)
            raise FileExistsError(msg)

    def read(self, name: str) -> Entity:
        """Reads an entity from the registry"""
        registry = self._read_registry()
        try:
            return registry[name]
        except KeyError:
            msg = "Entity named {} not found.".format(name)
            logger.error(msg)
            raise FileNotFoundError(msg)

    def update(self, entity: Entity) -> None:
        """Updates the existing registry with the given entity."""
        entity.data = None
        registry = self._read_registry()
        registry[entity.name] = entity
        self._write_registry(registry)

    def delete(self, name: str) -> None:
        registry = self._read_registry()

        try:
            del registry[name]
            self._write_registry(registry)
        except KeyError:
            msg = "Entity named {} does not exist in registry.".format(name)
            logger.error(msg)
            raise FileNotFoundError(msg)

    def exists(self, name: str) -> bool:
        """Evaluates existence  of a named entity in the registry."""
        registry = self._read_registry()

        try:
            registry[name]
            return True
        except KeyError:
            return False

    def count(self) -> int:
        """Counts the number of elements in the registry."""
        df = self.as_df()
        return int(df.shape[0])

    def as_dict(self, name: str = None) -> list:
        """Returns a list of entity objects from the register"""
        registry = self._read_registry()
        if name is not None:
            try:
                return registry[name]
            except KeyError:
                return {}
        else:
            return registry

    def as_df(self) -> pd.DataFrame:
        """Returns the registry as a DataFrame"""
        entities = []
        registry = self._read_registry()
        for name, entity in registry.items():
            entities.append(entity.as_dict())
        return pd.DataFrame(entities)

    def print(self) -> None:
        """Prints the registry as a pandas DataFrame."""
        print(self.as_df())

    def _read_registry(self) -> dict:
        """Returns the registry's contents."""
        if os.path.exists(self._registry_filepath):
            registry = self._io.read(filepath=self._registry_filepath)
        else:
            registry = {}
        return registry

    def _write_registry(self, registry: dict) -> None:
        """Writes the registry to file."""
        os.makedirs(os.path.dirname(self._registry_filepath), exist_ok=True)
        self._io.write(filepath=self._registry_filepath, data=registry)


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    """Generic Repository.

    Args:
        io_factory (IOFactory): The class responsible for providing access to IO services.
        registry (Registry): Class responsible for registration of TableRepo objects.

    """

    __input = "input"
    __working = "working"
    __format = "csv"

    def __init__(self, io_factory: IOFactory, registry: Registry, *args, **kwargs) -> None:
        self._io = io_factory.create(Repo.__format)

        self._registry = registry(io_factory=io_factory, repo_name=self.__class__.__name__)

        load_dotenv(".env")
        self._base_dir = os.getenv("BASE_DATA_DIR")
        self._archive_dir = os.getenv("ARCHIVE_DATA_DIR")
        self._input_dir = os.path.join(self._base_dir, Repo.__input)
        self._working_dir = os.path.join(self._base_dir, Repo.__working)

    def reset(self, force: bool = False) -> None:
        """Deletes the registry. If force is True, permanently deletes all files in the repo."""
        reset = input("This will wipe out the registry. Do you want to proceed? [y/n] ")
        if reset in ["y", "Y", "yes", "Yes", "YES"]:
            entities = self._registry.as_dict()
            if force:
                force = input(
                    "Force delete will permanently delete all files in this repo. Are you SURE you wish to proceed? [y/n]"
                )
                if force in ["y", "Y", "yes", "Yes", "YES"]:
                    for name in entities.keys():
                        self.delete(name)

            self._registry.reset()

    def load(self, directory: str) -> None:
        """Loads existing entity objects into the registry"""
        files = self._get_paths(directory)
        for file in files:
            # Add the data, even though it will be removed prior to writing to registry.
            # Metadata, such as size, rows, and columns are added during instantiation
            # based upon the data attribute.
            data = self._io.read(filepath=file)
            entity = Table(
                name=os.path.splitext(os.path.basename(file))[0],
                type="dataset",
                path=file,
                data=data,
            )
            self._registry.create(entity)

    def create(self, entity: Entity, force: bool = False) -> Entity:
        """Adds an entity to the repository, if it doesn't already exist."""
        # Set the path
        entity.path = self._set_path(entity)
        if self._is_valid_for_create(entity) or force:
            self._write_entity(entity)
            self._registry.create(entity)
        else:
            logger.error(
                "File {} already exists. Set force = True to overwrite".format(entity.name)
            )
        return entity

    def read(self, name: str) -> Entity:
        """Return a entity object by name from the repo"""
        return self._read_entity(name=name)

    def find(self, entity: Entity) -> Entity:
        """Finds the file by name and updates the entity path and registry if found."""
        pattern = self._get_file_search_pattern(entity.name)
        # Get files that match the name pattern
        files = glob(pattern, root_dir=self._base_dir, recursive=True)
        # Return the single match or the closest match based upon % similarity of the file and the entity name
        if len(files) == 1:
            entity.path = os.path.join(self._base_dir, files[0])

        elif len(files) > 1:
            ratio = []
            for file in files:
                ratio.append(SequenceMatcher(a=file, b=entity.name).ratio())
            idx = np.random.choice(np.argmax(np.array(ratio)))
            entity.path = os.path.join(self._base_dir, files[idx][0])
        return entity

    def archive(self, entity: Entity, *args, **kwargs) -> Entity:
        """Moves the named file to archive."""

        if entity.path is not None:
            if os.path.exists(entity.path):
                source = entity.path
                entity.path = self._get_archive_path(entity.path)
                target = entity.path
                filename = os.path.basename(entity.path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(source, target)
                entity.is_archived = True
                self._registry.update(entity)
                logger.info(
                    "Moved {} from {} to {}.".format(
                        filename, os.path.dirname(source), os.path.dirname(target)
                    )
                )
        else:
            raise ValueError(
                "{} {} could not be archived. The path is None or the file does not exist.".format(
                    entity.__class__.__name__, entity.name
                )
            )
        return entity

    def delete(self, name: str) -> None:
        entity = self._registry.read(name=name)
        os.remove(entity.path)
        self._registry.delete(name=entity.name)
        logger.info("Deleted {}: {}.".format(entity.__class__.__name__, entity.path))

    def exists(self, name) -> bool:
        """Returns True if the named entity's data exists."""
        return self._registry.exists(name)

    def count(self) -> bool:
        """Returns True if the named entity's data exists."""
        return self._registry.count()

    def print(self) -> None:
        self._registry.print()

    def _read_entity(self, name: str) -> Entity:
        """Reads an entity from the registry."""

        entity = self._registry.read(name=name)
        entity.data = self._io.read(filepath=entity.path)
        return entity

    def _write_entity(self, entity: Entity) -> None:
        # Write the data to file
        self._io.write(filepath=entity.path, data=entity.data)

    def _is_valid_for_create(self, entity: Entity) -> bool:
        """Ensures the entity can be created."""
        return self._file_not_exist(entity) and self._registration_not_exist(entity)

    def _is_valid_for_update_delete(self, entity: Entity) -> bool:
        return self._file_exist(entity) and self._registration_exists(entity)

    def _get_archive_path(self, path: str) -> str:
        return os.path.join(self._archive_dir, os.path.basename(path))

    @abstractmethod
    def _get_paths(self, directory: str) -> list:
        pass

    @abstractmethod
    def _set_path(self, entity: Entity) -> str:
        pass

    @abstractmethod
    def _get_file_search_pattern(self, name: str) -> str:
        pass

    # -------------------------------------------------------------------------------------------- #
    #                                       VALIDATION                                             #
    # -------------------------------------------------------------------------------------------- #
    def _file_exist(self, entity: Entity) -> bool:
        return os.path.exists(entity.path)

    def _file_not_exist(self, entity: Entity) -> bool:
        return not os.path.exists(entity.path)

    def _registration_exists(self, entity: Entity) -> bool:
        return self._registry.exists(name=entity.name)

    def _registration_not_exist(self, entity: Entity) -> bool:
        return not self._registry.exists(name=entity.name)


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
        self._io = io_factory.create(TableRepo.__format)

        self._registry = registry(io_factory=io_factory, repo_name=self.__class__.__name__)

        load_dotenv(".env")
        self._base_dir = os.getenv("BASE_DATA_DIR")
        self._archive_dir = os.getenv("ARCHIVE_DATA_DIR")
        self._input_dir = os.path.join(self._base_dir, TableRepo.__input)
        self._working_dir = os.path.join(self._base_dir, TableRepo.__working)

    def _get_paths(self, directory: str) -> list:
        """Loads existing entity objects into the registry"""
        pattern = directory + "/*." + TableRepo.__format
        return glob(pattern)

    def _set_path(self, entity: Entity) -> str:
        """Sets the path to the working directory. All new objects go in working directory"""
        filename = entity.name + "." + TableRepo.__format
        return os.path.join(self._working_dir, filename)

    def _get_file_search_pattern(self, name: str) -> str:
        """Converts a name to a file search pattern."""
        return "**/*" + name + "*." + TableRepo.__format
