#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /dataset.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 17th 2022 04:01:36 am                                                #
# Modified   : Monday October 24th 2022 07:24:50 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import pydicom
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Union
from csf.data.transforms import Hounsfield, Windower, Crop, Resize

from csf import (
    EFFICIENTNET_INPUT_SHAPE,
    WINDOW_DEFAULT,
    BATCH_SIZE,
    EFFICIENTNET_IMAGE_SIZE,
    IMAGE_CROP_SIZE,
    RANDOM_ROTATION_FACTOR,
)

# ------------------------------------------------------------------------------------------------ #


class DatasetBuilder(ABC):
    """Base class for tensorflow dataset builder subclasses."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
        buffer_size: Union[int, str] = "infer",
        output_shape: tuple = EFFICIENTNET_INPUT_SHAPE,
        training: bool = True,
        eager: bool = False,
        seed: int = None,
    ) -> None:
        self._metadata = metadata
        self._batch_size = tf.constant(batch_size)
        self._buffer_size = tf.constant(buffer_size)
        self._output_shape = tf.constant(output_shape)
        self._training = training
        self._eager = eager
        self._seed = seed
        self._buffer_size = len(metadata) if buffer_size == "infer" else buffer_size
        self._ds = None

    @property
    def dataset(self) -> None:
        return self._ds

    @property
    def length(self) -> int:
        return sum(1 for _ in self._ds.unbatch())

    @property
    def spec(self) -> tf.TensorSpec:
        return self._ds.element_spec

    @property
    @abstractmethod
    def targets(self) -> list:
        pass

    def build(self) -> None:
        """Builds a TensorFlow dataset containing images and labels"""
        # Set eager execution to support custom functionality such as process_image
        tf.config.run_functions_eagerly(self._eager)

        images_ds = self._build_images()
        labels_ds = self._build_labels()
        self._ds = tf.data.Dataset.zip((images_ds, labels_ds))

        if self._training:
            if self._eager:
                self._ds = self._ds.map(self._augment).unbatch()

        self._configure_for_performance()

    def _build_images(self) -> tf.data.Dataset:
        filepaths = tf.constant(self._metadata["Filepath"].values)
        filepaths_ds = tf.data.Dataset.from_tensor_slices(filepaths)

        if self._eager:

            images = filepaths_ds.map(
                lambda filepath: tf.py_function(
                    func=self._process_image, inp=[filepath], Tout=(tf.float32, tf.float32)
                )
            )
        else:
            images = filepaths_ds.map(
                self._process_image,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        return images

    def _build_labels(self) -> tf.data.Dataset:
        labels = list(self._metadata[self.targets].values)
        labels = tf.constant(labels)
        return tf.data.Dataset.from_tensor_slices(labels)

    def _process_image(self, filepath: tf.Tensor) -> tf.image:
        """Reads the DICOM file, and performs basic preprocessing and returns a tensorflow image."""

        filepath = bytes.decode(filepath.numpy())

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

    def _augment(self, image, labels) -> tf.Tensor:

        images = []
        images.append(image)
        images.append(tf.image.stateless_random_flip_left_right(image=image, seed=self._seed))
        images.append(tf.image.stateless_random_flip_up_down(image=image, seed=self._seed))
        images.append(
            tf.image.resize_with_crop_or_pad(
                image=tf.image.stateless_random_crop(
                    value=image, size=(IMAGE_CROP_SIZE, IMAGE_CROP_SIZE), seed=self._seed
                ),
                target_height=EFFICIENTNET_IMAGE_SIZE,
                target_width=EFFICIENTNET_IMAGE_SIZE,
            )
        )
        images.append(
            tf.keras.layers.RandomRotation(
                factor=RANDOM_ROTATION_FACTOR, fill_mode="constant", seed=self._seed, fill_value=0.0
            )
        )
        labels = tf.repeat(labels, repeats=len(images))

        return images, labels

    def _configure_for_performance(self):
        self._ds = self._ds.shuffle(buffer_size=len(self._ds))
        self._ds = self._ds.batch(self._batch_size, drop_remainder=True)
        self._ds = self._ds.repeat()
        self._ds = self._ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# ------------------------------------------------------------------------------------------------ #


class RSNALabelDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
        buffer_size: Union[int, str] = "infer",
        output_shape: tuple = EFFICIENTNET_INPUT_SHAPE,
        training: bool = True,
        eager: bool = False,
        seed: int = None,
    ) -> None:
        super().__init__(
            metadata=metadata,
            batch_size=batch_size,
            buffer_size=buffer_size,
            output_shape=output_shape,
            training=training,
            eager=eager,
            seed=seed,
        )

    @property
    def targets(self) -> list:
        return [f"C{i}" for i in range(1, 8)]
