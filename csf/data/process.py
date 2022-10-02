#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /process.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 1st 2022 10:32:32 am                                               #
# Modified   : Saturday October 1st 2022 03:39:54 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import logging
import logging.config
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology

# ------------------------------------------------------------------------------------------------ #
from atelier.workflow.operators import Operator

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IMG_WIDTH = 20
IMG_HEIGHT = 10
# ------------------------------------------------------------------------------------------------ #


class HounsfieldTransformer(Operator):
    """Transforms a medical image to hounsfield units

    Args:
        name (str): The name of the operator or pipeline step.
        params (dict): Compile time parameters:
            display (bool): True to display before and after images.
    """

    def __init__(self, name: str, params: dict = {}) -> None:
        super(HounsfieldTransformer, self).__init__(name, params)
        self._display = self._params.get("display", False)

    def execute(self, data: np.array) -> np.array:
        """Transforms a DICOM image into hounsfield units

        Args:
            data (pydicom.dataset.Dataset): A DICOM Dataset
        """
        image = data.pixel_array
        hu_image = image * data.RescaleSlope + data.RescaleIntercept

        if self._display:
            plt.figure(figsize=(IMG_WIDTH, IMG_HEIGHT))
            plt.style.use("grayscale")

            plt.subplot(151)
            plt.imshow(image)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(152)
            plt.imshow(hu_image)
            plt.title("Hounsfield Transformed")
            plt.axis("off")

        return hu_image


# ------------------------------------------------------------------------------------------------ #


class Windower(Operator):
    """Adjusts brightness of image to highlight particular structures in the CT Scan

    Args:
        name (str): The name of the operator or pipeline step.
        params (dict): Dictionary containing the following two elements:
            center (int): Midpoint of range of the CT numbers in hounsfield units to be displayed
            width (int): The range of the CT numbers in hounsfield units an image contains.
            display (bool): True to show the image before and after windowing
    """

    def __init__(self, name: str, params: dict = {}) -> None:
        super(Windower, self).__init__(name, params)
        self._center = self._params.get("center", None)
        self._width = self._params.get("width", None)
        self._display = self._params.get("display", False)

    def execute(self, data: np.array) -> np.array:
        """Conducts the windowing

        Args:
            data (np.array): Numpy array of pixels in hounsfield units.
        """
        img_min = self._center - self._width // 2
        img_max = self._center + self._width // 2
        window_image = data.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image


# ------------------------------------------------------------------------------------------------ #


class Denoiser(Operator):
    """Creates a mask to remove noise from the image

    Args:
        name (str): The name of the operator or pipeline step.
        params (dict): The following compile time parameter(s):
            radius (int): The radius of the circle in hu units used by morphology.dilation
            display (bool): True to show the image before and after denoising.

    """

    def __init__(self, name: str, params: dict = {}) -> None:
        super(Denoiser, self).__init__(name, params)
        self._radius = self._params.get("radius", None)
        self._display = self._params.get("display", False)

    def execute(self, data: np.array) -> np.array:
        """Performs the noise reduction of the image.

        Args:
            data (np.array): Numpy array of pixels in hounsfield units.
        """
        # morphology.dilation creates a segmentation of the image.
        # If one pixel is between the origin and the edge of a square of size
        # 5x5, the pixel belongs to the same class

        # We can instead use a circular using: morphology.disk(2)
        # In this case the pixel belongs to the same class if it's between the origin
        # and the radius

        segmentation = morphology.dilation(data, np.ones((self._radius, self._radius)))
        labels, label_nb = ndimage.label(segmentation)

        label_count = np.bincount(labels.ravel().astype(np.int))
        # The size of label_count is the number of classes/segmentations found

        # We don't use the first class since it's the background
        label_count[0] = 0

        # We create a mask with the class with more pixels
        # In this case should be the spine
        mask = labels == label_count.argmax()

        # Improve the spine mask
        mask = morphology.dilation(mask, np.ones((5, 5)))
        mask = ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))

        # Since the the pixels in the mask are zero's and one's
        # We can multiple the original image to only keep the spine region
        masked_image = mask * data

        if self._display:
            plt.figure(figsize=(16, 4))
            plt.subplot(141)
            plt.imshow(data)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(142)
            plt.imshow(mask)
            plt.title("Mask")
            plt.axis("off")

            plt.subplot(143)
            plt.imshow(masked_image)
            plt.title("Final Image")
            plt.axis("off")

        return masked_image
