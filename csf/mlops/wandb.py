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
# Created    : Tuesday October 25th 2022 06:00:11 pm                                               #
# Modified   : Tuesday October 25th 2022 07:57:24 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""WANDB Module."""
import wandb

# ------------------------------------------------------------------------------------------------ #


class WandB:
    def __init__(self) -> None:
        self._run = None
        self._artifact = None

    def start_run(self, project: str = None) -> None:
        self._run = wandb.init(project=project)

    def log_single_artifact(self, filepath: str, name: str, type: str) -> None:
        wandb.log_artifact(filepath, name=name, type=type)

    def log_artifact_dir(self, name: str, type: str, dir: str) -> None:
        self._artifact = wandb.Artifact(name, type)
        self._artifact.add_dir(dir)
        self._run.log_artifact(self._artifact)

    def download_artifact(self, artifact_name: str, type: str) -> None:
        """Downloads an existing artifact from wandb.

        Args:
            artifact_name (str): The format of the artifact must be
                'user_name/project_name/artifact_name:v1'

        Returns: The directory containing the downloaded artifact.
        """
        self._artifact = self._run.use_artifact(artifact_name, type=type)
        return self._artifact.download()

    def
