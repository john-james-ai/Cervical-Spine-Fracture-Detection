#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /slices.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday September 17th 2022 01:36:54 am                                            #
# Modified   : Saturday October 1st 2022 08:09:58 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import numpy as np
import matplotlib.pyplot as plt
from pydicom.dataset import FileDataset

# ------------------------------------------------------------------------------------------------ #


class CSFSlice:
    """Represents a single slice from a Dicom image

    dataset (FileDataset): Dicom Image from File
    """

    def __init__(self, dataset: FileDataset) -> None:
        self._dataset = dataset
        self._patient_id = dataset["PatientID"]
        self._content_date = dataset["ContentDate"]

    @property
    def patient_id(self) -> str:
        return self._patient_id

    @property
    def content_date(self) -> str:
        return self._content_date

    def plot_slice(self, save: bool = False) -> None:
        image = self._dataset.pixel_array

        print(image.shape)

        hu_image = self._transform_to_hu(self._dataset, image)
        brain_image = self._window_image(hu_image, 40, 80)
        bone_image = self._window_image(hu_image, 400, 1000)

        plt.figure(figsize=(20, 10))
        plt.style.use("grayscale")

        plt.subplot(151)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(152)
        plt.imshow(hu_image)
        plt.title("Hu image")
        plt.axis("off")

        plt.subplot(153)
        plt.imshow(brain_image)
        plt.title("brain image")
        plt.axis("off")

        plt.subplot(154)
        plt.imshow(bone_image)
        plt.title("bone image")
        plt.axis("off")

    def print_slice(self) -> None:
        self._print_slice(self._dataset)

    def _transform_to_hu(self, dataset, image) -> np.array:
        intercept = dataset.RescaleIntercept
        slope = dataset.RescaleSlope
        hu_image = image * slope + intercept
        return hu_image

    def _window_image(self, image, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image

    # ------------------------------------------------------------------------------------------------ #
    # Source: Pydicom Examples
    # Link: https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_printing_dataset.html#sphx-glr-auto-examples-input-output-plot-printing-dataset-py
    def _print_slice(self, dataset, indent=0):
        """Go through all items in the dataset and print them with custom format

        Modelled after Dataset._pretty_str()
        """
        dont_print = ["Pixel Data", "File Meta Information Version"]

        indent_string = "   " * indent
        next_indent_string = "   " * (indent + 1)

        for data_element in dataset:
            if data_element.VR == "SQ":  # a sequence
                print(indent_string, data_element.name)
                for sequence_item in data_element.value:
                    self._print_slice(sequence_item, indent + 1)
                    print(next_indent_string + "---------")
            else:
                if data_element.name in dont_print:
                    print("""<item not printed -- in the "don't print" list>""")
                else:
                    repr_value = repr(data_element.value)
                    if len(repr_value) > 50:
                        repr_value = repr_value[:50] + "..."
                    print(
                        "{0:s} {1:s} = {2:s}".format(indent_string, data_element.name, repr_value)
                    )
