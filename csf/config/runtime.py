#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /runtime.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 10:57:43 am                                                #
# Modified   : Saturday November 5th 2022 12:15:56 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Runtime Configuration Module

This module has been adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""
import os
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional

from csf.base.config import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class RuntimeConfig(Config):
    """High-level configurations for Runtime.

    Discovers current tensorflow runtime environment. Also determines whether the runtime
    environment is kaggle or localhost. The base data directory for the runtime environment
    is stored in the .env file.

    Args:
        name: Defaults to 'runtime'.
        environment: e.g. 'local', 'kaggle'
        base_data_dir: e.g. 'data', 'kaggle'
        strategy_type: e.g. 'mirrored', 'tpu', etc.
        strategy: The distribution strategy object.
        device: One of ['cpu','gpu','tpu']
        gpus: List of available GPUs.
        num_gpus: The number of GPUs to use, if any.
        tpu: The address of the TPU to use, if any.
        num_workers: Integer number of multiprocessing cores available.
    """

    name: str = "runtime"
    environment: str = "Localhost"
    base_data_dir: str = "data"
    strategy_type: str = "mirrored"
    strategy: tf.distribute = None
    device: str = ""
    gpus: list = None
    num_gpus: int = 0
    tpu: Optional[str] = None
    num_workers: int = 12

    # Global model parallelism configurations.
    num_cores_per_replica: int = 1
    default_shard_dim: int = -1

    def __post_init__(self) -> None:
        try:
            self.tpu = (
                tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            )  # connect to tpu cluster
            self.strategy = tf.distribute.TPUStrategy(self.tpu)  # get strategy for tpu
            self.strategy_type = "tpu"
            self.device = "tpu"

        except:  # noqa(E722)
            self.gpus = tf.config.list_logical_devices("GPU")  # get logical gpus
            self.num_gpus = len(self.gpus)
            if self.num_gpus > 0:
                self.strategy = tf.distribute.MirroredStrategy(self.gpus)  # single-GPU or multi-GPU
                self.strategy_type = "mirrored"
                self.device = "gpu"
            else:
                self.strategy = tf.distribute.get_strategy()  # connect to single gpu or cpu
                self.strategy_type = "default"
                self.device = "cpu"

        self.environment = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Localhost")
        if self.environment == "Localhost":
            self.base_data_dir = "data/raw"
            os.environ["BASE_DATA_DIR"] = "data/raw"
        else:
            self.base_data_dir = "kaggle"
            os.environ[
                "BASE_DATA_DIR"
            ] = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"

    def model_parallelism(self):
        return dict(
            num_cores_per_replica=self.num_cores_per_replica,
            default_shard_dim=self.default_shard_dim,
        )
