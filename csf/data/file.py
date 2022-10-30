#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /fileset.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 25th 2022 02:05:17 am                                               #
# Modified   : Tuesday October 25th 2022 06:41:21 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Data Artifact Definition Module

This module has been loosely adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""
import os
from dataclasses import dataclass
from typing import Optional
import wandb

from csf.base.entity import Entity

# ------------------------------------------------------------------------------------------------ #


@dataclass
class FileSet(Entity):
    """FileSet

    Args:
        name (str): A human readible name for a fileset that doesnt contain the file extension
        type (str): The type of the artifact, which is used to organize and differentiate artifacts.
            Common types include dataset or model, but you can use any string containing letters, numbers,
            underscores, hyphens, and dots.
        description (str):  Optional free text that offers a description of the artifact. The description is
            markdown rendered in the UI, so this is a good place to place tables, links, etc.
        metadata (dict): Optional structured data associated with the artifact, for example class
            distribution of a dataset. This will eventually be queryable and plottable in the UI.
            There is a hard limit of 100 total keys.
        aliases (list): A list of the aliases associated with this artifact. The list is mutable and
            calling save() will persist all alias changes.
        commit_hash (str):  The artifact's commit hash which is used in http URLs
        digest (str): The artifact's logical digest, a checksum of its contents. If an artifact
            has the same digest as the current latest version, then log_artifact is a no-op.
        entity (str): he name of the entity this artifact belongs to.
        id (str): The Artifact's id.
        manifest (ArtifactManifest): The artifact's manifest, listing all of its contents. You cannot add more
            files to an artifact once you've retrieved its manifest.
        project (str):  The name of the project this artifact belongs to.
        size (int): The size in bytes of the artifact. Includes any references tracked by this artifact.
        state (str):  The state of the artifact, which can be one of "PENDING", "COMMITTED", or "DELETED".
        version (str): The version of this artifact. For example, if this is the first version of an artifact,
            its version will be 'v0'.
        obj (wandb.WBValue): The object to add. Currently support one of Bokeh, JoinedTable, PartitionedTable,
            Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D
        localpath (str): The path to the directory being added.
        is_tmp (bool): Optional If true, then the file is renamed deterministically to avoid collisions.
            (default: False)
        uri (str): The URI path of the reference to add. Can be an object returned from
            Artifact.get_path to store a reference to another artifact's entry.
        checksum (bool): Optional Whether or not to checksum the resource(s) located at the reference URI.
            Checksumming is strongly recommended as it enables automatic integrity validation, however it
            can be disabled to speed up artifact creation. (default: True)
        max_objects (int): Optional The maximum number of objects to consider when adding a reference that
            points to directory or bucket store prefix. For S3 and GCS, this limit is 10,000 by default
            but is uncapped for other URI schemes. (default: None)
        root (str): The directory to replace with this artifact's files.
        mode (str): Optional. The mode in which to open the new file.
        encoding (str): The encoding in which to open the new file.
        settings (wandb.Settings, optional) A settings object to use when initializing an automatic run.
            Most commonly used in testing harness.
        on_kaggle (bool): True if the Fileset is stored on the Kaggle website.

    """

    name: str
    project: str
    id: str = ""
    type: str = "fileset"
    description: Optional[str] = ""
    metadata: dict = {}
    aliases: list = []

    commit_hash: str = ""
    digest: str = ""
    entity: str = ""  # Stored in environment variables
    manifest: wandb.ArtifactManifest = ""
    size: int = 0  # Provided by the repository
    state: str = "PENDING"
    version: str = "v0"
    obj: wandb.WBValue = ""
    localpath: str = ""  # Filepath from root or base directory
    uri: str = ""  # Filepath to directory containing fileset
    is_tmp: bool = False
    checksum: bool = True
    max_objects: int = 10000
    root: str = ""  # Root directory of the artifact
    mode: str = ""  # Stored in the environment variables
    encoding: str = "utf8"
    on_kaggle: str = False

    def __post_init__(self) -> None:
        self.mode = os.get_env("WANDB_MODE")
        self.entity = os.get_env("WANDB_ENTITY")
        self.id = self.name + "-" + wandb.util.generate_id()
        self.type = "fileset"
