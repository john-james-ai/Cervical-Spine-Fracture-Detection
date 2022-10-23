#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /transforms.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 18th 2022 03:32:14 am                                               #
# Modified   : Sunday October 23rd 2022 11:11:09 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pydicom
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from csf import WINDOW_DEFAULT


# ------------------------------------------------------------------------------------------------ #
#                                    HOUNSFIELD TRANSFORMER                                        #
# ------------------------------------------------------------------------------------------------ #
def to_hounsfield(dicom: pydicom.FileDataset):
    """Takes a DICOM FileDataset and returns an image converted to Hoounsfield units.

    Args:
        dicom (pydicom.FileDataset): DICOM FileDataset containing the image to transform
    """
    # Extract scaling parameters
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope
    # Obtain the image
    image = dicom.pixel_array
    image = slope.astype(np.float64) * image.astype(np.float64) + intercept.astype(np.int16)
    return image


# ------------------------------------------------------------------------------------------------ #
class Hounsfield(layers.Layer):
    """Linear transformation to Hounsfield Units"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, dicom: pydicom.FileDataset) -> np.array:
        """Takes a DICOM FileDataset and returns a transformed image

        Args:
            dicom (pydicom.FileDataset): DICOM FileDataset containing the image to transform
        """
        return to_hounsfield(dicom)


# ------------------------------------------------------------------------------------------------ #
#                                        WINDOWER                                                  #
# ------------------------------------------------------------------------------------------------ #


def windower(image: np.array, window: tuple = WINDOW_DEFAULT) -> np.array:
    window_width = window[0]
    window_center = window[1]

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    image[image < img_min] = img_min
    image[image > img_max] = img_max

    return image


# ------------------------------------------------------------------------------------------------ #
class Windower(layers.Layer):
    def __init__(self, window: tuple = WINDOW_DEFAULT, **kwargs) -> None:
        super().__init__(**kwargs)
        self._window = window

    def call(self, image: np.array) -> np.array:
        return windower(image=image, window=self._window)


# ------------------------------------------------------------------------------------------------ #
#                                           CROP                                                   #
# ------------------------------------------------------------------------------------------------ #
def crop(image: np.array) -> np.array:
    mask = image == 0

    # Find the cervical spine area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    image = image[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]  # noqa E203

    return image


class Crop(layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(self, image: np.array) -> np.array:
        return crop(image=image, keep_size=self._keep_size)


# ------------------------------------------------------------------------------------------------ #
#                                          RESIZE                                                  #
# ------------------------------------------------------------------------------------------------ #


def resize(image: np.array, output_shape: tuple) -> np.array:
    """Resizes an image and adds an rgb channel if rgb_dim is not None

    Args:
        image (np.array): Image pixel data
        output_shape (list): The output shape of the image in 2D
    """

    new_image = tf.constant(image)
    new_image = tf.image.resize(new_image, size=output_shape)

    return new_image


# ------------------------------------------------------------------------------------------------ #


class Resize(layers.Layer):
    """Resizes an image.

    Args:
        output_shape (tuple): The output size of the image in 2D
    """

    def __init__(self, output_shape: tuple, **kwargs) -> None:
        super().__init__(kwargs)
        self._output_shape = output_shape

    def call(self, image: np.array) -> np.array:
        return resize(image=image, output_shape=self._output_shape)
