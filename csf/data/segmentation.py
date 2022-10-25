#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /segmentation.py                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 16th 2022 09:54:27 pm                                                #
# Modified   : Sunday October 23rd 2022 09:21:07 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Vertebrae Segmentation and Labeling Data"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
import nibabel as nib
from csf.data.base import BaseMETA
from csf import (
    SEGMENTATION_FILEPATHS,
    SEGMENTATION_METADATA_FILEPATH,
    TRAIN_SLICE_METADATA_FILEPATH,
    N_JOBS,
)

# ------------------------------------------------------------------------------------------------ #


class VertebraeLabelExtractor(BaseMETA):
    """Extracts ground-truth vertebrae labels from segmentation dataset

    Args:
        segmentation_filepath_glob_pattern (str): Glob pattern for segmentation files.
        output_filepath (str): Path to label / slice output file

    """

    def __init__(
        self,
        train_metadata_filepath: str = TRAIN_SLICE_METADATA_FILEPATH,
        segmentation_filepath_glob_pattern: str = SEGMENTATION_FILEPATHS,
        output_filepath: str = SEGMENTATION_METADATA_FILEPATH,
        force: str = False,
    ) -> None:

        self._train_metadata_filepath = train_metadata_filepath
        self._segmentation_filepath_glob_pattern = segmentation_filepath_glob_pattern
        self._output_filepath = output_filepath
        self._force = force
        self._df = pd.DataFrame()
        self._train_metadata = None

    def load(self) -> None:
        """Loads the training metadata."""

        if not self._force and os.path.exists(self._output_filepath):
            print(
                f"Labels have been extracted to {self._output_filepath}. Set force to True to override."
            )
        else:
            self._train_metadata = pd.read_csv(self._train_metadata_filepath, header=0, index_col=0)

            # Obtain all segmentation studies and their filenames
            study_uids, filepaths = self._get_segmented_studies()

            # Extract label information for each study
            study_labels = Parallel(n_jobs=N_JOBS, require="sharedmem", verbose=10)(
                delayed(self.extract)(filepath, uid)
                for filepath, uid in tqdm(zip(filepaths, study_uids))
            )

            # Concatenate study_labels to the labels dataframe
            self._df = pd.concat(study_labels)

            self._df.set_index("StudyInstanceUID", inplace=True)

    def extract(self, filepath: str, uid: str) -> None:
        """Extracts the volume, iterates, over slices, and labels, then updates the DataFrame

        Args:
            filepath (str): The path to the segmentation file
            uid (str): The StudyInstanceUID being processed.

        """

        df = self._read_study_metadata(uid)

        seg = nib.load(filepath)
        volume = seg.get_fdata()

        # Volume is in Saggittal Orientation and must be transposed to align with the training data
        volume = volume[:, ::-1, ::-1].transpose(2, 1, 0)
        n_slices = volume.shape[0]

        # Iterate over slices and extract C1 through C7 vertebrae labels.
        for slice_idx in range(n_slices):
            labels = list(np.unique(volume[slice_idx, :, :]))

            # Iterate over all labels except the first one, which is background, and ignore the thoratic spine vertebrae.
            for label in labels[1:]:
                if label < 8:
                    df.loc[df["SliceNumber"] == slice_idx, f"C{int(label)}"] = 1

        return df  # Study slices and associated vertebrae labels.

    def save(self) -> None:
        self._df.to_csv(self._output_filepath)

    def _get_segmented_studies(self) -> dict:
        """Return all segmented studies and their filenames."""

        # Segmentation and label data
        seg_filepaths = [filepath for filepath in glob(self._segmentation_filepath_glob_pattern)]
        seg_study_uids = [
            os.path.splitext(os.path.basename(filepath))[0] for filepath in seg_filepaths
        ]
        return seg_study_uids, seg_filepaths

    def _read_study_metadata(self, uid: str) -> pd.DataFrame:
        """Obtain training metadata for the study."""
        df = self._train_metadata.loc[(self._train_metadata["StudyInstanceUID"] == uid)]
        targets = [f"C{i}" for i in range(1, 8)]
        df[targets] = 0
        return df
