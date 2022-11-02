#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /registry.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday September 5th 2022 04:51:01 pm                                               #
# Modified   : Tuesday October 25th 2022 11:53:37 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-Clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
# Constants used throughout the package
from datetime import datetime
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union
import wandb

# ------------------------------------------------------------------------------------------------ #

def FileRegistry


@dataclass
class Fileset:
    """General Artifact class

    Args:
        name (str): Human readible name for the Artifact
        type (str): Type of Artifact, e.g. 'dataset', 'model'
        project (str): Name of project for the Artifact. A project can be a collection of
            experiments or related jobs.
        description (str): Describes the purpose and intended use of the Artifact
        version (str): Defaults to 'v0'
        id (str): Defaults to 'entity-project-type-name-generated_id' format.
        entity (str): The userid for the wandb account.
        aliases (list): A list of aliases for the Artifact.
        digest (str): The artifact's logical digest, a checksum of its contents. If an artifact has the same digest as the current latest version, then log_artifact is a no-op.
        state (str): The state of the artifact, which can be one of "PENDING", "COMMITTED", or "DELETED".
        obj (wandb.WBValue) The object to add. Currently support one of Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D
        manifest (ArtifactManifest): The artifact's manifest, listing all of its contents. You cannot add more files to an artifact once you've retrieved its manifest.
        checksum (bool): Whether or not to checksum the resource(s) located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation, however it can be disabled to speed up artifact creation. (default: True)
        metadata (dict): Structured data associated with the artifact, for example class distribution of a dataset. This will eventually be queryable and plottable in the UI. There is a hard limit of 100 total keys.
        size (int): The size in bytes of the artifact. Includes any references tracked by this artifact.
        max_objects (int): The maximum number of objects to consider when adding a reference that points to directory or bucket store prefix. For S3 and GCS, this limit is 10,000 by default but is uncapped for other URI schemes. (default: None)
        is_tmp (bool): If true, then the file is renamed deterministically to avoid collisions. (default: False)
        commit_hash (str):  The artifact's commit hash which is used in http URLs
        root (str): The directory to replace with this artifact's files.
        local_path (str): The path to the file being added.
        target_path (str): he path to the portfolio. It must take the form {portfolio}, {project}/{portfolio} or {entity}/{project}/{portfolio}.
        uri (str): The URI path of the reference to add. Can be an object returned from Artifact.get_path to store a reference to another artifact's entry.
        table (wandb.Table) A wandb Table
        created (datetime): Datetime the Artifact was created
    """

    name: str
    type: str
    project: str
    description: str = None
    version: int = "v0"
    id: str = None
    entity: str = "aistudio"
    aliases: Optional[list] = []
    digest: Optional[str] = ""
    state: str = "PENDING"
    obj: wandb.WBValue = None
    manifest: Optional[wandb.ArtifactManifestEntry] = ""
    checksum: Optional[bool] = True
    metadata: Optional[dict] = {}
    size: Optional[int] = 0
    max_objects: Optional[int] = None
    is_tmp: Optional[bool] = False
    commit_hash: Optional[str] = None
    root: Optional[str] = None
    local_path: Optional[str] = None
    target_path: Optional[str] = None
    uri: Union[wandb.ArtifactEntry, str] = ""
    table: Optional[wandb.Table] = None
    created: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.id = (
            self.entity
            + "_"
            + self.project
            + "-"
            + self.type
            + "-"
            + self.name
            + "-"
            + wandb.util.generate_id()
        )

    def generate_artifact(self) -> dict:
        """Places all attributes in a dictionary, with the exception of name and type, two attributes used to initialize the Artifact."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if v not in [None, ""] and k not in ["name", "type"]
        }


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Table:
    """Encapsulates tabular data posted to WandB for Data and Analysis Tasks

    Args:
        columns (list): Columns in the table. Default is None.
        data Union[int,float,str,np.array,wandb.data_types.Video] Supports all WandB datatypes.
        dataframe (pd.DataFrame): Pandas DataFrame
        dtype Union[wandb.data_types]
        allow_mixed_types Union[True,False]
    """

    columns: list = None
    data: Optional[wandb.data_type] = None
    dataframe: pd.DataFrame = None
    dtype: Union[wandb.data_type] = None
    optional: Optional[True] = None
    allow_mixed_types: Union[True, False] = None

    def generate_artifact(self) -> dict:
        """Returns a dictionary of the non-null parameteres."""
        return {k: v for k, v in self.__dict__.items() if v not in [None, ""]}


# ------------------------------------------------------------------------------------------------ #
#                                         ENVIRONMENT                                              #
# ------------------------------------------------------------------------------------------------ #
IS_KAGGLE = False

# ------------------------------------------------------------------------------------------------ #
#                                 BASE ENVIRONMENT DIRECTORIES                                     #
# ------------------------------------------------------------------------------------------------ #
KAGGLE = {
    "INPUT": "../input/rsna-2022-cervical-spine-fracture-detection",
    "METADATA": "../working/metadata/rsna-2022-cervical-spine-fracture-detection",
    "PROCESSED": "../working/processed/rsna-2022-cervical-spine-fracture-detection",
    "MODELS": "../working/models/rsna-2022-cervical-spine-fracture-detection",
}
LOCAL = {
    "INPUT": "data/input",
    "METADATA": "data/metadata",
    "PROCESSED": "data/processed",
    "MODELS": "models",
}

if IS_KAGGLE:
    BASE_DIRS = KAGGLE
else:
    BASE_DIRS = LOCAL

# ------------------------------------------------------------------------------------------------ #
#                                    INPUT IMAGE DATA                                              #
# ------------------------------------------------------------------------------------------------ #
# Key Image Subdirectories
TRAIN_IMAGES_DIR = f"{BASE_DIRS['INPUT']}/train_images"
TEST_IMAGES_DIR = f"{BASE_DIRS['INPUT']}/test_images"
SEGMENTATION_DIR = f"{BASE_DIRS['INPUT']}/segmentations"

# Filepaths
TRAIN_IMAGES_FILEPATHS = f"{TRAIN_IMAGES_DIR}/**/*.dcm"
TEST_IMAGES_FILEPATHS = f"{TEST_IMAGES_DIR}/**/*.dcm"
SEGMENTATION_FILEPATHS = f"{SEGMENTATION_DIR}/*.nii"


# ------------------------------------------------------------------------------------------------ #
#                                   INPUT TABULAR DATA                                             #
# ------------------------------------------------------------------------------------------------ #
# Tabular Filenames
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"
TRAIN_BBOXES_FILENAME = "train_bounding_boxes.csv"


# Tabular Filepaths
TRAIN_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_FILENAME}"
TEST_FILEPATH = f"{BASE_DIRS['INPUT']}/{TEST_FILENAME}"
SAMPLE_SUBMISSION_FILEPATH = f"{BASE_DIRS['INPUT']}/{SAMPLE_SUBMISSION_FILENAME}"
TRAIN_BBOX_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_BBOXES_FILENAME}"


# ------------------------------------------------------------------------------------------------ #
#                                        TRAIN METADATA                                            #
# ------------------------------------------------------------------------------------------------ #
# Filenames
TRAIN_SLICE_METADATA_FILENAME = "train_slice_metadata.csv"
TRAIN_SCAN_METADATA_FILENAME = "train_scan_metadata.csv"


# Filepaths
TRAIN_SLICE_METADATA_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_SLICE_METADATA_FILENAME}"
TRAIN_SCAN_METADATA_FILEPATH = f"{BASE_DIRS['INPUT']}/{TRAIN_SCAN_METADATA_FILENAME}"


# ------------------------------------------------------------------------------------------------ #
#                                        SEGMENTATION METADATA                                     #
# ------------------------------------------------------------------------------------------------ #
# Labels
SEGMENTATION_METADATA_FILENAME = "segmentation_metadata.csv"
SEGMENTATION_METADATA_FILEPATH = f"{BASE_DIRS['INPUT']}/{SEGMENTATION_METADATA_FILENAME}"


# ------------------------------------------------------------------------------------------------ #
#                                           MULTIPROCESSING                                        #
# ------------------------------------------------------------------------------------------------ #
N_JOBS = 12

# ------------------------------------------------------------------------------------------------ #
#                                           VISUALIZATION                                          #
# ------------------------------------------------------------------------------------------------ #
FIG_SIZE = (12, 4)

# ------------------------------------------------------------------------------------------------ #
#                                             MODELS                                               #
# ------------------------------------------------------------------------------------------------ #
LABELING_MODEL = {"name": "EfficientNetB3", "code": "B3"}
KFOLDS = 5
BATCH_SIZE = 32
NO_CLASSES = 7
NO_EPOCHS = 25


# ------------------------------------------------------------------------------------------------ #
#                                           IMAGE PROCESSING                                       #
# ------------------------------------------------------------------------------------------------ #
WINDOW_DEFAULT = (1200, 250)
DICOM_IMAGE_SHAPE = [512, 512]
EFFICIENTNET_INPUT_SHAPES = {
    "B0": [224, 224, 3],
    "B1": [240, 240, 3],
    "B2": [260, 260, 3],
    "B3": [300, 300, 3],
    "B4": [380, 380, 3],
    "B5": [456, 456, 3],
    "B6": [528, 528, 3],
    "B7": [600, 600, 3],
}
IMAGE_CROP_SIZE = 128
RANDOM_ROTATION_FACTOR = 0.15
EFFICIENTNET_INPUT_SHAPE = EFFICIENTNET_INPUT_SHAPES[LABELING_MODEL["code"]]
EFFICIENTNET_IMAGE_SIZE = EFFICIENTNET_INPUT_SHAPE[0]
