#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /model.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 10:57:43 am                                                #
# Modified   : Monday October 24th 2022 01:36:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Runtime Configuration Definition Module

This module has been adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""

from dataclasses import dataclass

from csf.config.base import Config
from csf.config.optimization import OptimizationConfig

# ------------------------------------------------------------------------------------------------ #


@dataclass
class TrainerConfig(Config):
    """Configuration for trainer.

    Args:
        optimizer_config: optimizer config, it includes optimizer, learning rate,
            and warmup schedule configs.
        train_tf_while_loop: whether or not to use tf while loop.
        train_tf_function: whether or not to use tf_function for training loop.
        eval_tf_function: whether or not to use tf_function for eval.
        allow_tpu_summary: Whether to allow summary happen inside the XLA program
            runs on TPU through automatic outside compilation.
        steps_per_loop: number of steps per loop to report training metrics. This
            can also be used to reduce host worker communication in a TPU setup.
        summary_interval: number of steps between each summary.
        checkpoint_interval: number of steps between checkpoints.
        max_to_keep: max checkpoints to keep.
        continuous_eval_timeout: maximum number of seconds to wait between
            checkpoints, if set to None, continuous eval will wait indefinitely. This
            is only used continuous_train_and_eval and continuous_eval modes. Default
            value is 1 hrs.
        train_steps: number of train steps.
        validation_steps: number of eval steps. If -1, the entire eval dataset is
            used.
        validation_interval: number of training steps to run between evaluations.
        best_checkpoint_export_subdir: if set, the trainer will keep track of the
            best evaluation metric, and export the corresponding best checkpoint under
            `model_dir/best_checkpoint_export_subdir`. Note that this only works if
            mode contains eval (such as `train_and_eval`, `continuous_eval`, and
            `continuous_train_and_eval`).
        best_checkpoint_eval_metric: for exporting the best checkpoint, which
            evaluation metric the trainer should monitor. This can be any evaluation
            metric appears on tensorboard.
        best_checkpoint_metric_comp: for exporting the best checkpoint, how the
            trainer should compare the evaluation metrics. This can be either `higher`
            (higher the better) or `lower` (lower the better).
        validation_summary_subdir: A 'str', sub directory for saving eval summary.
    """

    optimizer_config: OptimizationConfig = OptimizationConfig()
    # Orbit settings.
    train_tf_while_loop: bool = True
    train_tf_function: bool = True
    eval_tf_function: bool = True
    eval_tf_while_loop: bool = False
    allow_tpu_summary: bool = False
    # Trainer intervals.
    steps_per_loop: int = 1000
    summary_interval: int = 1000
    checkpoint_interval: int = 1000
    # Checkpoint manager.
    max_to_keep: int = 5
    continuous_eval_timeout: int = 60 * 60
    # Train/Eval routines.
    train_steps: int = 0
    # Sets validation steps to be -1 to evaluate the entire dataset.
    validation_steps: int = -1
    validation_interval: int = 1000
    # Best checkpoint export.
    best_checkpoint_export_subdir: str = ""
    best_checkpoint_eval_metric: str = ""
    best_checkpoint_metric_comp: str = "higher"
    # Blowup recovery.
    loss_upper_bound: float = 1e6
    recovery_begin_steps: int = 0  # Enforcing the loss bound after these steps.
    # When max trials < 0, no recovery module; max trials = 0, we will check
    # the condition and fail the job if the condition happens; max trials > 0,
    # we will retore the model states.
    recovery_max_trials: int = 0
    validation_summary_subdir: str = "validation"
