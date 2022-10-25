#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /experiment.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 10:57:43 am                                                #
# Modified   : Monday October 24th 2022 01:33:17 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Experiment Configuration Definition Module

This module has been adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""

from dataclasses import dataclass

from csf.config.base import Config
from csf.config.task import TaskConfig
from csf.config.model import TrainerConfig
from csf.config.runtime import RuntimeConfig

# ------------------------------------------------------------------------------------------------ #


@dataclass
class ExperimentConfig(Config):
    """Top-level configuration."""

    task: TaskConfig = TaskConfig()
    trainer: TrainerConfig = TrainerConfig()
    runtime: RuntimeConfig = RuntimeConfig()
