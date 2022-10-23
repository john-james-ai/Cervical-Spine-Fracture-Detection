#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /preprocess.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 17th 2022 04:01:36 am                                                #
# Modified   : Sunday October 23rd 2022 10:45:30 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import numpy as np
import pydicom
from typing import Union
import tensorflow as tf
from tensorflow import keras
from csf.data.transforms import Hounsfield, Windower, Crop, Resize

from csf import (
    DICOM_IMAGE_SHAPE,
    EFFICIENTNET_INPUT_SHAPE,
    WINDOW_DEFAULT,
)

# ------------------------------------------------------------------------------------------------ #


class RSNALabelDatasetGenerator(tf.keras.utils.Sequence):
    """Generates the RSNA Dataset for labeling

    Args:
        train_segmentation_metadata_filepath (str): Filepath to the segmentation labels

    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        n_classes: int = 7,
        input_shape: tuple = DICOM_IMAGE_SHAPE,
        output_shape: tuple = EFFICIENTNET_INPUT_SHAPE,
        shuffle: bool = True,
        seed: int = None,
    ) -> None:
        self._df = df.copy()
        self._batch_size = batch_size
        self._n_classes = n_classes
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._shuffle = shuffle
        self._targets = [f"C{i}" for i in range(1, 8)]

        self._n = len(self._df)

    def __getitem__(self, index: int) -> Union[np.array, list]:
        batch = self._df[index * self._batch_size : (index + 1) * self._batch_size]  # noqa E203
        X, y = self._get_data(batch)
        return X, y

    def _get_data(self, batch: pd.DataFrame) -> Union[np.array, list]:
        X_batch = np.asarray([self._get_features(filepath) for filepath in batch["Filepath"]])
        y_batch = np.asarray([self._get_labels(batch)])

        return X_batch, keras.utils.to_categorical(y_batch, num_classes=self._n_classes)

    def _get_features(self, filepath: str) -> np.array:
        dicom = pydicom.dcmread(filepath)

        # Convert data to hounsfield units
        hounsfield = Hounsfield()
        image = hounsfield(dicom)

        # Window the input
        windower = Windower(window=WINDOW_DEFAULT)
        image = windower(image)

        # Crop the image to the ROI
        crop = Crop()
        image = crop(image)

        # Resize the height/width dimensions (first two dimensions) of the image to match that expected
        # by the Efficientnet model input.
        resize = Resize(output_shape=self._output_shape[:2])
        image = resize(image)

        # Add the rgb channel expected by Efficientnet model
        image = tf.constant(image)
        image = tf.image.grayscale_to_rgb(image)

        assert image.shape == self._output_shape, "Image shape does not match output shape."

        return image

    def _get_labels(self, batch: pd.DataFrame) -> pd.DataFrame:
        return batch[self._targets]

    def __len__(self) -> int:
        return self._n // self._batch_size

    def on_epoch_end(self):
        """Shuffles the training dataframe after each epoch"""
        if self._shuffle:
            self._df = self._df.sample(frac=1)
