#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /metadata.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 15th 2022 10:55:08 am                                              #
# Modified   : Saturday October 22nd 2022 07:08:58 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Vertebrae Segmentation and Labeling"""
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import pydicom
from pydicom import dcmread
from joblib import Parallel, delayed

from csf.data.base import BaseMETA
from csf import (
    TRAIN_IMAGES_DIR,
    TRAIN_IMAGES_FILEPATHS,
    TRAIN_SLICE_METADATA_FILEPATH,
    TRAIN_SCAN_METADATA_FILEPATH,
    N_JOBS,
)

# ------------------------------------------------------------------------------------------------ #


class SliceMETA(BaseMETA):
    """Encapsulates all DICOM SLICE metadata for exploratory purposes.

    filepaths (str): Path glob for all DICOM files.
    """

    def __init__(
        self,
        input_file_glob_pattern: str = TRAIN_IMAGES_FILEPATHS,
        output_filepath: str = TRAIN_SLICE_METADATA_FILEPATH,
        force: bool = False,
    ) -> None:
        super().__init__(
            input_file_glob_pattern=input_file_glob_pattern,
            output_filepath=output_filepath,
            force=force,
        )

    def load(self) -> None:
        """Loads the metadata.

        Load existing metadata from file, if it exists. Load from DICOM Datasets otherwise, or
        if Force is True.

        Args:
            force (bool): If True, compute from Slice metadata, even if the Scan metadata exists.
        """
        if self._force or not self._exists(self._output_filepath):

            slice_filepaths = glob(self._input_file_glob_pattern, recursive=True)
            data = Parallel(n_jobs=N_JOBS)(
                delayed(self._load)(slice_filepath) for slice_filepath in tqdm(slice_filepaths)
            )

            self._df = pd.DataFrame(data).set_index("SOPInstanceUID")
        else:
            self._df = pd.read_csv(self._output_filepath, header=0, index_col=0)

    def _load(self, filepath) -> dict:
        dataset = dcmread(filepath)
        return self._get_data(dataset)

    def query(self, expression: str = None) -> pd.DataFrame:
        return self._df.query(expression)

    def _get_data(self, dataset: pydicom.dataset.FileDataset) -> None:

        slice_metadata = {
            "SOPInstanceUID": dataset.SOPInstanceUID,
            "PatientID": dataset.PatientID,
            "StudyInstanceUID": dataset.StudyInstanceUID,
            "Series": dataset.SeriesInstanceUID.split(".")[-1],
            "SliceNumber": int(dataset.InstanceNumber),
            "SliceThickness": float(dataset.SliceThickness),
            "ImagePositionPatient_X": float(dataset.ImagePositionPatient[0]),
            "ImagePositionPatient_Y": float(dataset.ImagePositionPatient[1]),
            "ImagePositionPatient_Z": float(dataset.ImagePositionPatient[2]),
            "Rows": int(dataset.Rows),
            "Columns": int(dataset.Columns),
            "PixelSpacing_X": float(dataset.PixelSpacing[0]),
            "PixelSpacing_Y": float(dataset.PixelSpacing[1]),
            "RescaleIntercept": float(dataset.RescaleIntercept),
            "RescaleSlope": float(dataset.RescaleSlope),
            "TransferSyntaxUID": dataset.file_meta.TransferSyntaxUID,
            "SeriesInstanceUID": dataset.SeriesInstanceUID,
            "Filepath": os.path.join(
                TRAIN_IMAGES_DIR, dataset.SeriesInstanceUID, f"{dataset.InstanceNumber}.dcm"
            ),
        }
        if "WindowWidth" in dataset:
            width = dataset.WindowWidth
            if isinstance(width, pydicom.multival.MultiValue):
                width = float(width[0])
            else:
                width = float(str(width).replace(",", ""))
            slice_metadata["WindowWidth"] = width

        if "WindowCenter" in dataset:
            center = dataset.WindowCenter
            if isinstance(center, pydicom.multival.MultiValue):
                center = float(center[0])
            else:
                center = float(str(center).replace(",", ""))
            slice_metadata["WindowCenter"] = center

        return slice_metadata


# ------------------------------------------------------------------------------------------------ #


class ScanMETA(BaseMETA):
    """Aggregates and encapsulates the Scan level metadata from the slice level metadata.

    Args:
        filepath (str): Path to the Slice level metadata
    """

    def __init__(
        self,
        input_filepath: str = TRAIN_SLICE_METADATA_FILEPATH,
        output_filepath: str = TRAIN_SCAN_METADATA_FILEPATH,
        force: bool = False,
    ) -> None:
        super().__init__(
            input_filepath=input_filepath, output_filepath=output_filepath, force=force
        )

    @property
    def info(self) -> None:
        pass

    @property
    def shape(self) -> None:
        pass

    def load(self, force: bool = False) -> None:
        """Loads Scan level metadata.

        Load from existing file if it exists and Force is False, otherwise, create from Slice Metadata.

        Args:
            force (bool): If True, compute from Slice metadata, even if the Scan metadata exists.
        """
        if self._force or not self._exists(self._output_filepath):

            slice_metadata = pd.read_csv(
                self._input_filepath,
                usecols=[
                    "StudyInstanceUID",
                    "PatientID",
                    "SliceThickness",
                    "PixelSpacing_X",
                    "PixelSpacing_Y",
                    "RescaleIntercept",
                    "RescaleSlope",
                    "WindowWidth",
                    "WindowCenter",
                ],
                index_col=False,
            )

            slice_metadata["Slices"] = slice_metadata.StudyInstanceUID.map(
                slice_metadata.StudyInstanceUID.value_counts()
            )
            self._df = slice_metadata.groupby("StudyInstanceUID").first().reset_index()

        else:
            self._df = pd.read_csv(self._output_filepath)
