#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine FractureDetection                                                    #
# Version    : 0.1.0                                                                               #
# Filename   : /config.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/CervicalSpineFractureDetection                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday July 29th 2022 12:41:04 am                                                   #
# Modified   : Thursday September 8th 2022 11:32:57 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Configuration Module"""
from abc import ABC
import os
from dotenv import load_dotenv

from atelier.data.io import YamlIO

# ------------------------------------------------------------------------------------------------ #


class Config(ABC):
    """Base class defining read / write access to configuration files."""

    def __init__(self, name: str) -> None:
        self._filepath = None
        self._io = YamlIO
        self._config = None
        self._initialize(name)

    @property
    def config(self) -> dict:
        return self._config

    @config.setter
    def config(self, config: dict) -> None:
        self._config = config
        self.save()

    def load(self) -> list:
        """Reads data from a yaml file."""
        self._config = self._io.read(self._filepath)

    def save(self) -> None:
        """Writes config data to a yaml file."""
        self._io.write(data=self._config, filepath=self._filepath)

    def _initialize(self, name: str) -> None:
        """Initializes the Config object with an io object, a filepath, and the config data."""

        # Config filepaths are stored in the environment variables.
        load_dotenv()
        self._filepath = os.getenv(name)

        # Load the configuration data
        self._config = self._io.read(self._filepath)


# ------------------------------------------------------------------------------------------------ #


class LogConfig(Config):

    __CONFIG_NAME = "CONFIG_LOG"

    def __init__(self) -> None:
        config = LogConfig.__CONFIG_NAME
        super(LogConfig, self).__init__(config)


# ------------------------------------------------------------------------------------------------ #


class SpacyConfig(Config):

    __CONFIG_NAME = "CONFIG_SPACY"

    def __init__(self) -> None:
        self._config_name = SpacyConfig.__CONFIG_NAME
