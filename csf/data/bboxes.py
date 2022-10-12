#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /bboxes.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 4th 2022 03:15:09 am                                                #
# Modified   : Tuesday October 4th 2022 06:27:28 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------------ #


class CSFBoundingBoxes:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._bounding_boxes = None
        self._patients = None
        self._summary = None
        self._errors = None
        self._multiple_boxes = None

    @property
    def shape(self) -> tuple:
        return self._bounding_boxes.shape

    @property
    def n_patients(self) -> int:
        return len(self._patients)

    @property
    def n_errors(self) -> pd.DataFrame:
        return len(self._errors)

    @property
    def n_multiple_boxes(self) -> int:
        return len(self._multiple_boxes)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._bounding_boxes.head(n)

    def get_bounding_boxes(self) -> pd.DataFrame:
        return self._bounding_boxes

    def get_patients(self, n: int = None, random_state: int = None) -> list:
        """Gets all or n randomly selected patients with bounding boxes

        Args:
            n (int): Number of patients to return. None (default) returns all
        """
        if n:
            rng = np.random.default_rng(random_state)
            return rng.choice(self._patients, n, replace=False)

        else:
            return self._patients

    def summarize(self) -> pd.DataFrame:
        return self._summary

    def get_errors(self) -> pd.DataFrame:
        return self._errors

    def get_multiple_boxes_summary(self) -> pd.DataFrame:
        return self._multiple_boxes.sort_values("count")

    def get_patients_multiple_boxes(self) -> np.array:
        return self._multiple_boxes.index.values

    def get_multiple_boxes(self) -> pd.DataFrame:
        return self._bounding_boxes[
            self._bounding_boxes["StudyInstanceUID"].isin(self._multiple_boxes.index.values)
        ].sort_values(["StudyInstanceUID", "slice_number"])

    def load(self) -> None:
        self._bounding_boxes = pd.read_csv(self._filepath)
        self._patients = set(self._bounding_boxes["StudyInstanceUID"])
        self._summary = self._bounding_boxes.groupby("StudyInstanceUID").agg(
            {"slice_number": ["min", "max", "count"]}
        )
        self._summary.columns = self._summary.columns.droplevel()
        self._summary["range"] = self._summary["max"] - self._summary["min"] + 1
        self._errors = self._summary[(self._summary["range"] < self._summary["count"])]
        self._multiple_boxes = self._summary[(self._summary["range"] > self._summary["count"])]
