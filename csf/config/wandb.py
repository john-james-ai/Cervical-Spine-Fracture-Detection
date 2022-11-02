#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /wandb.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 09:28:16 pm                                                #
# Modified   : Wednesday November 2nd 2022 05:41:54 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Project Configuration Definition Module"""
import os
from dataclasses import dataclass
import wandb
from typing import Optional
import logging
import logging.config

from csf.config.base import Config
from csv.config.data import FileConfig, DatasetConfig



# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ProjectConfig(Config):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = self.name + "-" + wandb.util.generate_id()


# ------------------------------------------------------------------------------------------------ #


@dataclass
class RunGroupConfig(Config):
    """RunGroupConfig: Groups JobConfig's into Experiments or related Data or Analysis Jobs"""

    project_config: ProjectConfig = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            self.id = self.project_config.name + "-" + self.name + "-" + wandb.util.generate_id()
        except AttributeError:



# ------------------------------------------------------------------------------------------------ #
@dataclass
class JobConfig(Config):
    """Lowest level unit of execution in WandB"""

    run_group_config: RunGroupConfig
    group: str = None
    project: str = None
    project_config: ProjectConfig = None
    save_code: Optional[bool] = True
    job_type: str = None
    tags: Optional[list] = None
    dir: Optional[str] = None
    notes: Optional[str] = None
    resume: Optional[bool, str] = None
    reinit: Optional[bool] = False
    magic: Optional[bool, dict, str] = False
    anonymous: str = "never"
    mode: str = "online"
    allow_val_change: bool = False
    force: bool = True
    sync_tensorboard: Optional[bool] = False
    monitor_gym: Optional[bool] = False
    config: dict = ""

    def __post_init__(self) -> None:
        self.id = (
            self.run_group_config.project_config.name
            + "-"
            + self.run_group_config.name
            + "-"
            + wandb.util.generate_id()
        )
        self.project_config = self.run_group_config.project_config
        self.project = self.project_config.name
        self.group = self.run_group_config.name
        self.dir = os.getenv("WANDB_DIR")
        self.mode = os.getenv("WANDB_MODE")


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataJobConfig(JobConfig):
    job_type: str = "data"
    data_task_config: DataTaskConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = (
            self.run_group_config.project_config.name
            + "-"
            + self.run_group_config.name
            + "-"
            + self.job_type
            + "-"
            + wandb.util.generate_id()
        )
        self.config = {self.data_task.config.as_dict()}


# ------------------------------------------------------------------------------------------------ #
@dataclass
class AnalysisJobConfig(JobConfig):
    job_type: str = "analysis"
    analysis_config: AnalysisConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = (
            self.run_group_config.project_config.name
            + "-"
            + self.run_group_config.name
            + "-"
            + self.job_type
            + "-"
            + wandb.util.generate_id()
        )
        self.config = self.


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ClassificationJobConfig(JobConfig):
    job_type: str = "classification"
    input_config: DatasetConfig

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = (
            self.run_group_config.project_config.name
            + "-"
            + self.run_group_config.name
            + "-"
            + self.job_type
            + "-"
            + wandb.util.generate_id()
        )
        self.config = {
            "input_config": {
                "id": self.input_config.id,
                "name": self.input_config.name,
                "type": self.input_config.__class__.__name__,
            }
        }
