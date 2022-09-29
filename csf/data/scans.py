#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /scans.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday September 15th 2022 03:50:46 am                                            #
# Modified   : Thursday September 29th 2022 01:12:46 pm                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import logging
import logging.config
from pydicom import dcmread

from . import train_images, Study
from .slices import CSFSlice

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class CSFScan:
    """Class for managing and viewing Dicom files containing CT scans.

    Args:
        study_id (str): The study id associated with the scan.
        basedir (str): Train or test images directory
    """

    def __init__(self, study: Study, basedir: str = train_images) -> None:
        self._study = study
        self._basedir = basedir
        self._directory = os.path.join(basedir, study.id)
        self._files = []
        self._slices = []
        self._img3d = None
        self._img_shape = None
        self._load_slices()

    def __str__(self):
        return "\nScan for Study Id: {} with {} fractures.\n\tC1: {}\n\tC2: {}\n\tC3: {}\n\tC4: {}\n\tC5: {}\n\tC6: {}\n\tC7: {}\n".format(
            self._study.id,
            self.self_study.total,
            self._study.C1,
            self._study.C2,
            self._study.C3,
            self._study.C4,
            self._study.C5,
            self._study.C6,
            self._study.C7,
        )

    def __repr__(self):
        return "{self.__class__.__name__}(study={self._study},basedir={self._basedir})".format(
            self=self
        )

    @property
    def study(self) -> str:
        return self._study

    @property
    def basedir(self) -> str:
        return self._basedir

    @property
    def img_shape(self) -> list:
        return self._img_shape

    @property
    def patient_id(self) -> str:
        return self._patient_id

    @property
    def content_date(self) -> str:
        return self._content_date

    def plot_3d_slices(self) -> np.array:
        # TODO: Won't work with slice objects.
        # pixel aspects, assuming all slices are the same
        ps = self._slices[0].PixelSpacing
        ss = self._slices[0].SliceThickness
        ax_aspect = ps[1] / ps[0]
        sag_aspect = ps[1] / ss
        cor_aspect = ss / ps[0]

        # create 3D array
        img_shape = list(self._slices[0].pixel_array.shape)
        img_shape.append(len(self._slices))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(self._slices):
            img2d = s.pixel_array
            img3d[:, :, i] = img2d

        # plot 3 orthogonal slices
        a1 = plt.subplot(1, 3, 1)
        plt.imshow(img3d[:, :, img_shape[2] // 2])
        a1.set_aspect(ax_aspect)

        a2 = plt.subplot(1, 3, 2)
        plt.imshow(img3d[:, img_shape[1] // 2, :])
        a2.set_aspect(sag_aspect)

        a3 = plt.subplot(1, 3, 3)
        plt.imshow(img3d[img_shape[0] // 2, :, :].T)
        a3.set_aspect(cor_aspect)

        plt.show()

    def get_slice(self, slice_number: int = None) -> CSFSlice:
        """Returns a designated or random slice object

        Args:
            slice_number (int): Zero indexed slice number
        """
        slice_number = slice_number or self._get_random_slice_number()
        return self._slices[slice_number]

    def view_slice(self) -> None:
        fig, ax = plt.subplots()
        ax.volume = self._img3d
        ax.index = ax.volume.shape[0] // 2
        ax.imshow(ax.volume[ax.index])
        fig.canvas.mpl_connect("key_press_event", self._process_key)

    def _process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == "p":
            self._previous_slice(ax)
        elif event.key == "n":
            self._next_slice(ax)
        fig.canvas.draw()

    def _previous_slice(self, ax):
        """Go to the previous slice."""
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def _next_slice(self, ax):
        """Go to the next slice."""
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

    def _load_slices(self) -> None:
        """Sorts and returns the slices having a SliceLocation"""
        logger.info("Loading slices for {}".format(self._study.id))
        self._slices = []
        for fname in os.listdir(self._directory):
            fpath = os.path.join(self._directory, fname)
            ds = dcmread(fpath)
            slice = CSFSlice(dataset=ds)
            self._slices.append(slice)

        self._set_scan_metadata()

        logger.info("{} Slices Loaded".format(len(self._slices)))

    def _set_scan_metadata(self) -> None:
        """Extracts metadata from the first slice for the scan properties."""
        self._patient_id = self._slices[0].patient_id
        self._content_date = self._slices[0].content_date

    def _get_random_slice_number(self) -> int:
        rng = default_rng()
        return rng.integers(low=0, high=len(self._slices), size=1)[0]
