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
# Modified   : Thursday October 27th 2022 01:53:37 pm                                              #
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

    input_image_shape: tuple = (512, 512)
    output_image_shape: tuple = (224, 224, 3)
    n_classes: int = 7  # Or 8 if including patient_overall
    classes: list = []
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
