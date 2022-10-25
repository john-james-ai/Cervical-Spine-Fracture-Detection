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
# Modified   : Monday October 24th 2022 02:10:22 pm                                                #
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
import tensorflow as tf
from dataclasses import dataclass
from typing import Union, Optional

from csf.config.base import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class RuntimeConfig(Config):
    """High-level configurations for Runtime.

    Args:
        distribution_strategy_type: e.g. 'mirrored', 'tpu', etc.
        distribution_strategy: The distribution strategy object.
        enable_xla: Whether or not to enable XLA.
        gpus: List of available GPUs.
        num_gpus: The number of GPUs to use, if any.
        gpu_thread_mode: Whether and how the GPU device uses its own threadpool.
        dataset_num_private_threads: Number of threads for a private threadpool
            created for all datasets computation.
        per_gpu_thread_count: thread count per GPU.
        tpu: The address of the TPU to use, if any.
        num_workers: Integer number of multiprocessing cores available.
        worker_hosts: comma-separated list of worker ip:port pairs for running
            multi-worker models with DistributionStrategy.
        task_index: If multi-worker training, the task index of this worker.
        all_reduce_alg: Defines the algorithm for performing all-reduce.
        num_packs: Sets `num_packs` in the cross device ops used in
            MirroredStrategy.  For details, see tf.distribute.NcclAllReduce.
        mixed_precision_dtype: dtype of mixed precision policy. It can be 'float32',
            'float16', or 'bfloat16'.
        loss_scale: The type of loss scale, or 'float' value. This is used when
            setting the mixed precision policy.
        run_eagerly: Whether or not to run the experiment eagerly.
        batchnorm_spatial_persistent: Whether or not to enable the spatial
            persistent mode for CuDNN batch norm kernel for improved GPU performance.
    """

    distribution_strategy_type: str = "mirrored"
    distribution_strategy: tf.distribution.Strategy = None
    device: str = ""
    enable_xla: bool = False
    gpus: list = None
    num_gpus: int = 0
    gpu_thread_mode: Optional[str] = None
    dataset_num_private_threads: Optional[int] = None
    per_gpu_thread_count: int = 0
    tpu: Optional[str] = None
    num_workers: int = 12
    worker_hosts: Optional[str] = None
    task_index: int = -1
    all_reduce_alg: Optional[str] = None
    num_packs: int = 1
    mixed_precision_dtype: Optional[str] = None
    loss_scale: Optional[Union[str, float]] = None
    run_eagerly: bool = False
    batchnorm_spatial_persistent: bool = False

    # XLA runtime params.
    # XLA params are only applied to the train_step.
    # These augments can improve training speed. They can also improve eval, but
    # may reduce usability and users would need to make changes to code.

    # Whether to enable XLA dynamic padder
    # infrastructure to handle dynamic shapes inputs inside XLA. True by
    # default. Disabling this may cause correctness issues with dynamic shapes
    # inputs, as XLA will just assume the inputs are with padded shapes. However
    # users can optionally set it to False to improve device time if masking is
    # already handled in the user side.
    # If None, will respect XLA default.
    tpu_enable_xla_dynamic_padder: Optional[bool] = None

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
            self.device = "TPU"

        except tf.python.framework.errors.NotFoundError:  # otherwise detect GPUs
            self.gpus = tf.config.list_logical_devices("GPU")  # get logical gpus
            self.num_gpus = len(self.gpus)
            if self.n_gpus > 0:
                self.strategy = tf.distribute.MirroredStrategy(self.gpus)  # single-GPU or multi-GPU
                self.strategy_type = "mirrored"
                self.device = "GPU"
            else:
                self.strategy = tf.distribute.get_strategy()  # connect to single gpu or cpu
                self.strategy_type = "default"
                self.device = "CPU"

    def model_parallelism(self):
        return dict(
            num_cores_per_replica=self.num_cores_per_replica,
            default_shard_dim=self.default_shard_dim,
        )
