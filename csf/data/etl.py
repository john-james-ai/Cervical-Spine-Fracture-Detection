#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /etl.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 15th 2022 10:55:08 am                                              #
# Modified   : Tuesday November 1st 2022 03:51:06 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Vertebrae Segmentation and Labeling

Two classes make up the data preprocessing as it pertains to segmentation and labeling. The
first is RSNASegmentationVertebraeExtractor, which extracts study, slice, and vertebrae
information from the NIfTI segmentation files.  The second class, RSNAVertebraeDataset is a
tensorflow dataset containing the DICOM image and the vertebrae targets.

"""
import os

# import tensorflow as tf
import numpy as np
import pandas as pd
import shlex
import subprocess
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Any, Union
from csf.base.operator import Operator

# ------------------------------------------------------------------------------------------------ #
#                                     KAGGLE DOWNLOADER                                            #
# ------------------------------------------------------------------------------------------------ #


class KaggleDownloader(Operator):
    """Downloads Dataset from Kaggle using the Kaggle API
    Args:
        name (str): name for the step in the pipeline
        competition (str): The name of the competition. Refer
            to the api syntax on the competition data page
        filename (str): The name of the zip file. This may be obtained by clicking on the
            'Download All' button on the data page.
        destination (str): The folder to which the data will be downloaded.
        force (bool): Indicates whether to force execution if current data exists.
    """

    def __init__(
        self, name: str, competition: str, filename: str, destination: str, force: bool = False
    ) -> None:
        super(KaggleDownloader, self).__init__(
            name=name,
            competition=competition,
            filename=filename,
            destination=destination,
            force=force,
        )

        self.filepath = os.path.join(self.destination, self.filename)

        self._command = (
            "kaggle competitions download" + " -p " + self._filepath + " -c " + self.competition
        )

    def execute(self, *args, **kwargs) -> Any:
        """Downloads compressed data via an API using bash
        Args:
            data: Not used
            context: not used.
        """
        if self.force or not os.path.exists(self.filepath):

            os.makedirs(self.destination, exist_ok=True)
            subprocess.run(shlex.split(self._command), check=True, text=True, shell=False)

    def return_code(self) -> str:
        return len(os.path.listdir(self.destination)) > 0


# ------------------------------------------------------------------------------------------------ #


class SegmentationVertebraeExtractor(Operator):
    """Task which extracts vertebrae information for each image in the RNSA Segmentation file.

    NIfTI files contain the image and vertebrae information in the sagittal plane. The data
    are transformed into axial plane slices to match the orientation in the RSNA DICOM
    image files. Once an image is transformed, the vertebrae present in that image are
    extracted into a binary vector, one element per vertebrae.  A value of one in the vector
    indicates that the vertebrae for that index was present in the image. Once the vertebrae vector
    is created, it is paired with the StudyInstanceUID, filename, and slice number into a
    pandas DataFrame.  Finally, the DataFrame is stored in the designed output file.

    Args:
        name (str): The name of the operation.
        force (bool): False will suppress execution if the output file is present. True
            will force execution, in either event.
        source (str): Path pattern which will return all NIfTI segmentation files.
        target (str): Path to the output file containing study and vertebrae data.
        n_jobs (int): The number of parallel jobs to execute. Defaults to 12. Be advised!
        verbose (int): Level of output to be produced by the Parallel joblib function.
            Defaults to 0.

    """

    N_VERTEBRAE = 7

    def __init__(
        self,
        name: str,
        source: str,
        target: str,
        n_jobs: int = 12,
        verbose: int = 10,
        force: bool = False,
    ) -> None:
        super().__init__(
            name=name, source=source, target=target, n_jobs=n_jobs, verbose=verbose, force=force
        )

    def execute(self, *args, **kwargs) -> Any:
        """Runs the program if forced or output file doesn't already exist."""
        if self.force or not os.path.exists(self.target):
            self._run()
        else:
            self._skipped = True

    def _run(self) -> None:
        filepaths = glob(self.source)
        data = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._process_study)(filepath) for filepath in tqdm(filepaths)
        )
        self._save(data)

    def _process_study(self, filepath: str) -> list:
        """Reads NIfTI data, extracts vertebrae data, and returns a list of dictionaries."""
        # Initialize a list that will contain dictionaries for each slice in the study
        study_data = []

        # Extract the study id from the filename.
        study_id = os.path.splitext(os.path.basename(filepath))[0]

        # Read data in sagittal orientation into a 3 dimensional array.
        seg_data = self._nibabel_io.read(filepath)

        # Iterate through each (transposed to axial) slice and return a list of dictionaries - one entry per slice.
        for slice_number, (slice) in enumerate(seg_data[:, ::-1, ::-1].transpose(2, 0, 1)):

            # The unique non-zero values indicate the vertebrae present in the slice.
            if len(np.nonzero(np.unique(slice))) > 0:
                vertabrae = np.unique(slice)[
                    1:
                ]  # If np.unique stops returning sorted values, we're screwed.
                slice_data = self._process_slice(study_id, slice_number, vertabrae)
                study_data.append(slice_data)

        return study_data

    def _process_slice(self, study_id: str, slice_number: int, vertebrae: np.array) -> dict:
        slice_data = {}
        slice_data["StudyInstanceUID"] = study_id
        slice_data["SliceNumber"] = slice_number
        for i in range(1, SegmentationVertebraeExtractor.N_VERTEBRAE + 1):
            if i in vertebrae:
                slice_data[f"C{i}"] = 1
            else:
                slice_data[f"C{i}"] = 0

        return slice_data

    def _save(self, data) -> None:
        """Save list of lists of dictionaries."""
        vertebrae_data = []
        # Parallel returns a list of results from multiprocessing.
        for study in data:
            vertebrae_data.extend(study)
        df = pd.DataFrame(vertebrae_data)
        os.makedirs(os.path.dirname(self.target), exist_ok=True)
        df.to_csv(self.target, index=False)

    def return_code(self) -> Union[bool, str]:
        if self._skipped:
            return "skipped"
        else:
            return os.path.exists(self.target)
