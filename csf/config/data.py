#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /data.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 24th 2022 07:35:27 am                                                #
# Modified   : Monday October 24th 2022 11:38:34 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Data Configuration Definition Module

This module has been adapted from the TensorFlow / Models library and is subject to the terms
and conditions of the Apache License Version 2.0, included in the LICENSE document of this
package by reference.

Reference: :ref:`TensorFlow <https://github.com/tensorflow/models>`

"""
import os
from dataclasses import dataclass
import tensorflow as tf
from glob import glob
from typing import Union, Optional

from csf.config.base import Config

# ------------------------------------------------------------------------------------------------ #


@dataclass
class FileConfig(Config):
    """File Configuration

    Args:
        root_dir (str): The root directory for the file. This is set based upon the value of 'on_kaggle'
        kaggle_base_input_folder (str): The read-only Kaggle folder containing all input data
        kaggle_base_working_folder (str): The Kaggle folder in which working files are stored.
        local_base_input_folder (str): The local folder containing all input data
        local_base_working_folder (str): The local folder containing working files.
        path (str): Path or path specification for a file or file set, not including root_dir
        full_path (str): Full path or path specification to the file or file set.
        on_kaggle (bool): True if running in a Kaggle notebook, False otherwise. Default = False
        input_data (bool): True if the File is part of the provided input data, False otherwise. Default = True
        path_is_glob (bool): True if the path should be treated as a glob, False otherwise. Default = False
    """

    KAGGLE_BASE_INPUT_FOLDER: str = "kaggle/input/rsna-2022-cervical-spine-fracture-detection"
    KAGGLE_BASE_WORKING_FOLDER: str = "kaggle/working/"
    LOCAL_BASE_INPUT_FOLDER = "data/input/"
    KAGGLE_BASE_WORKING_FOLDER = "data/working/"

    root_dir: str = ""
    path: str = ""
    full_path: str = ""
    on_kaggle: bool = False
    input_data: bool = True
    path_is_glob: bool = False

    def __post_init__(self) -> None:
        """Sets the root_dir into which the file resides"""
        if self.on_kaggle and self.input_data:
            self.root_dir = self.kaggle_base_input_folder
        elif self.on_kaggle and not self.input_data:
            self.root_dir = self.kaggle_base_working_folder
        elif not self.on_kaggle and self.input_data:
            self.root_dir = self.local_base_input_folder
        else:
            self.root_dir = self.local_base_working_folder

        if self.path_is_glob:
            self.full_path = glob(pathname=self.path, root_dir=self.root_dir)
        else:
            self.full_path = os.path.join(self.root_dir, self.path)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class AugmentationConfig(Config):
    """Augmentation Configuration for a Dataset

    Args:
        frac (float): The fraction of the Dataset to augment for each augmentation
            operation. The data are shuffled prior to each augmentation and a
            different frac of observations are randomly selected for the augmentation
            operation. Default = 0.33
        flip_horizontal (bool): Whether to flip the image horizontally. Default = True.
        flip_vertical (bool): Whether to flip the image vertically. Default = True.
        crop (bool): Whether to crop the image. Default = True
        crop_shape (tuple): The target shape of the cropped image. Default = (128,128)
        rotate (bool): Whether to randomly rotate images. Default = True
        rotation_factor (np.float16): The rotation factor to apply. Default = 0.15. Which means
            that the images will be rotated a random number of degrees in [-0.15,0.15].
    """

    frac: float = 0.33
    flip_horizontal: bool = True
    flip_vertical: bool = True
    crop: bool = True
    rotate: bool = True
    crop_shape: tuple = (128, 128)
    rotation_factor: float = 0.15


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetConfig(Config):
    """TensorFlow Dataset Configuration

    Args:
        input_file (FileConfig): Configuration object for the input file.
        input_image_shape (tuple): Shape of input images. Default = (512,512)
        output_image_shape (tuple): Shape of output images. Default (224,224,3), the shape
            expected by the model.
        n_classes (int): The number of output classes. Defaults to 7; for the labeling task.
            Alternatively, n_classes is 8, including 7 vertebrae and patient_overall,
            for the fracture detection task.
        classes (list): The list of output classes for the classification task this Dataset supports.
        augmentation_config (AugmentationConfig): Configuration describing the data augmentation operation.
        batch_size (int): Size of batch for training. Default = 32
        batch_drop_remainder (bool): Whether partial batches at end of training should be dropped.
        training (bool): True if the Dataset is a training dataset.
        shuffle_buffer_size (Union[str,int]): Size of buffer to use in shuffling the data. Defaults
            to 'infer' which signals the Dataset Builder to set this value to the number of
            elements in the Dataset.
        prefetch_buffer_size (int): Buffer size to use when prefetching data during the training
            process. Defaults to 'AUTOTUNE', which dynamically calculates this value.
        seed (int): Pseudorandom seed for reproducibility.
    """

    input_file_config: FileConfig = ""
    input_image_shape: tuple = (512, 512)
    output_image_shape: tuple = (224, 224, 3)
    n_classes: int = 7  # Or 8 if including patient_overall
    classes: list = []
    augmentation_config: Optional[AugmentationConfig] = None
    batch_size: int = 32
    batch_drop_remainder: bool = True
    training: bool = True
    type: str = "train"
    shuffle_buffer_size: Union[str, int] = "infer"  # Should equal to number of elements in Dataset
    prefetch_buffer_size: Optional[int] = tf.data.AUTOTUNE  # Dynamically calculated at runtime
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.labels = [f"C{i}" for i in range(1, 8)]
        if self.n_classes == 8:
            self.labels.append("patient_overall")
