#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /base.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 15th 2022 10:55:08 am                                              #
# Modified   : Friday October 21st 2022 08:13:24 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Vertebrae Segmentation and Labeling"""
import os
from abc import ABC, abstractmethod
import pandas as pd

# ------------------------------------------------------------------------------------------------ #


class BaseMETA(ABC):
    def __init__(
        self,
        input_file_glob_pattern: str = None,
        input_filepath: str = None,
        output_filepath: str = None,
        force: bool = False,
    ) -> None:
        self._input_file_glob_pattern = input_file_glob_pattern
        self._input_filepath = input_filepath
        self._output_filepath = output_filepath
        self._force = force
        self._df = None

    @property
    def info(self) -> None:
        self._df.info()

    @property
    def shape(self) -> tuple:
        return self._df.shape

    @property
    def n_studies(self) -> int:
        return self._df.index.nunique()

    @property
    def n_slices(self) -> int:
        return self._df.shape[0]

    @property
    def slices_per_study(self) -> pd.DataFrame:
        return self._df["SliceNumber"].describe().T

    @property
    def missing(self) -> int:
        return self._df.isna().sum().sum()

    @property
    def columns(self) -> list:
        return self._df.columns.values

    @property
    def nunique(self) -> pd.DataFrame:
        return self._df.nunique(axis=0)

    @property
    def duplicate(self) -> int:
        return self._df.duplicated().sum()

    @property
    def input_file_glob_pattern(self) -> str:
        return self._input_file_glob_pattern

    @property
    def input_filepath(self) -> str:
        return self._input_filepath

    @property
    def output_filepath(self) -> str:
        return self._output_filepath

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)

    def describe(self, column: str = None) -> pd.DataFrame:
        if column:
            try:
                print(self._df[column].describe().T)
            except KeyError:
                print(f"{column} is invalid")

        else:
            print(self._describe().T)

    @abstractmethod
    def load(self) -> None:
        pass

    def save(self) -> None:
        if self._force or not os.path.exists(self._output_filepath):
            os.makedirs(os.path.dirname(self._output_filepath), exist_ok=True)
            self._df.to_csv(self._output_filepath)
        else:
            print(f"{self._output_filepath} already exists and was not overwritten.")

    def _exists(self, filepath: str) -> bool:
        return os.path.exists(filepath)
