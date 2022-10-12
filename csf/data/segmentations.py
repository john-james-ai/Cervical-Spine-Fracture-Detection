#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /segmentations.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 4th 2022 03:05:45 am                                                #
# Modified   : Tuesday October 4th 2022 11:58:18 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
from glob import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------ #
SEGMENTATIONS_DIR = "data/raw/segmentations"
# ------------------------------------------------------------------------------------------------ #


class CSFSegmentations:
    def __init__(self, directory: str = SEGMENTATIONS_DIR) -> None:
        self._directory = directory
        self._segmentation_paths = []
        self._patients = None
        self._load()

    @property
    def n_patients(self) -> int:
        return len(self._patients)

    def get_patients(self, n: int = None, random_state: int = None) -> list:
        """Gets all or n randomly selected patients

        Args:
            n (int): Number of patients to return. None (default) returns all
        """
        if n:
            rng = np.random.default_rng(random_state)
            return rng.choice(self._patients, n, replace=False)

        else:
            return self._patients

    def _load(self) -> None:
        self._segmentation_paths = glob(os.path.join(self._directory, "*.nii"))
        self._patients = [
            os.path.basename(os.path.splitext(p)[0]) for p in self._segmentation_paths
        ]


class CSFPatientSegmentations:
    """Representation of segmentations for a single patient.

    Args:
        patient_id (str): The StudyInstanceUID for the patient

    Source:
        Adapted from https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340612
    """

    def __init__(self, patient_id) -> None:
        self._patient_id = patient_id
        self._filepath = f"{SEGMENTATIONS_DIR}/{patient_id}.nii"
        self._segmentations = None
        self._load()

    @property
    def patient_id(self) -> str:
        return self._patient_id

    @property
    def n_slices(self) -> int:
        return self._segmentations.shape[0]

    def get_label(self, slice_number: int) -> np.array:
        """Returns the ground truth labels for the designated slice

        Returns an array where the first value 0 is background and the remaining values
        correspond to the bone in the slice.

        Args:
            slice  (int): Zero indexed slice number, i.e. slice number 9 relates to 10.dcm""
        """
        return np.unique(self._segmentations[slice_number])

    def plot_seg(
        self,
        start_slice: int = 0,
        n_slices: int = 24,
        nrows: int = 4,
        ncols: int = 6,
        figsize: tuple = (24, 12),
    ) -> None:
        assert nrows * ncols == n_slices, "n_slices must <= nrows * ncols"

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        title = f"Segmentation Plot\nPatient Id: {self._patient_id}"
        fig.suptitle(title, weight="bold", size=20)

        for slice_number in range(start_slice, start_slice + n_slices):
            mask = self._segmentations[slice_number]

            x = (slice_number - start_slice) // ncols
            y = (slice_number - start_slice) % ncols

            axes[x, y].imshow(mask)
            axes[x, y].set_title(f"Slice: {slice_number}", fontsize=14, weight="bold")
            axes[x, y].axis("off")

    def _load(self) -> None:
        """Loads the segmentations and transforms them to shape (num_images, height, width)"""
        self._segmentations = nib.load(self._filepath).get_fdata()[:, ::-1, ::-1].transpose(2, 1, 0)
