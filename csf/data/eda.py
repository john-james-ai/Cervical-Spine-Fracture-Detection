#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /eda.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday September 13th 2022 06:23:18 pm                                             #
# Modified   : Wednesday September 14th 2022 11:46:56 am                                           #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pandas as pd
from numpy.random import default_rng

# import matplotlib.pyplot as plt
# import seaborn as sns

# ------------------------------------------------------------------------------------------------ #


class CervicalSpineFractures:
    """Exploratory Data Analysis of Cervical Spine Fracture Training Dataset

    Args:
        filepath (str): The path to the dataset.
    """

    __original_columns = [
        "StudyInstanceUID",
        "patient_overall",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    ]
    __craniovertebral_region = ["C1", "C2"]
    __subaxial_region = ["C3", "C4", "C5", "C6", "C7"]
    __vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._df = self._get_data(filepath)
        self._summary = self._summarize()

    @property
    def info(self) -> pd.DataFrame:
        d = {}
        d["Columns"] = CervicalSpineFractures.__original_columns
        d["Dtype"] = self._df[CervicalSpineFractures.__original_columns].dtypes.values
        d["Non-Null Count"] = self._df[CervicalSpineFractures.__original_columns].count().values
        d["Null Count"] = (
            self._df[CervicalSpineFractures.__original_columns].isnull().sum(axis=0).values
        )
        d["Minimum"] = (
            self._df[CervicalSpineFractures.__original_columns]
            .min(axis=0, numeric_only=None)
            .values
        )
        d["Maximum"] = (
            self._df[CervicalSpineFractures.__original_columns]
            .max(axis=0, numeric_only=None)
            .values
        )
        d["Num Unique"] = self._df[CervicalSpineFractures.__original_columns].nunique(axis=0).values
        d["Memory Usage"] = (
            self._df[CervicalSpineFractures.__original_columns]
            .memory_usage(index=False, deep=True)
            .values
        )
        df = pd.DataFrame(data=d)
        return df

    @property
    def n_patients(self) -> int:
        return int(self._summary["n_patients"])

    @property
    def n_fractures(self) -> int:
        return int(self._summary["n_fractures"])

    @property
    def n_craniovertebral(self) -> int:
        return int(self._summary["n_craniovertebral"])

    @property
    def n_subaxial(self) -> int:
        return int(self._summary["n_subaxial"])

    @property
    def n_patients_with_fracture(self) -> int:
        return int(self._summary["n_patients_fracture"])

    @property
    def p_patients_with_fracture(self) -> int:
        return float(self._summary["p_patients_fracture"])

    @property
    def n_patients_craniovertebral(self) -> int:
        return int(self._summary["n_patients_craniovertebral"])

    @property
    def p_patients_craniovertebral(self) -> float:
        return float(self._summary["p_patients_craniovertebral"])

    @property
    def n_patients_subaxial(self) -> int:
        return int(self._summary["n_patients_subaxial"])

    @property
    def p_patients_subaxial(self) -> float:
        return float(self._summary["p_patients_subaxial"])

    @property
    def fractures_by_vertebrae(self) -> pd.DataFrame:
        return self._summary["fractures_by_vertebrae"]

    @property
    def fractures_by_region(self) -> pd.DataFrame:
        return self._summary["fractures_by_region"]

    def sample(self, n: int = 5) -> pd.DataFrame:
        rng = default_rng()
        indices = rng.integers(low=0, high=self._df.shape[0], size=n)
        return self._df[CervicalSpineFractures.__original_columns].loc[indices]

    def _get_data(self, filepath: str) -> pd.DataFrame:
        """Reads data from filepath.

        Args:
            filepath: (str): Location of file containing data
        Returns:
            DataFrame of original data.
        """
        df = pd.read_csv(filepath, index_col=False)
        df["craniovertebral"] = df[CervicalSpineFractures.__craniovertebral_region].any(axis=1)
        df["n_craniovertebral"] = df[CervicalSpineFractures.__craniovertebral_region].sum(axis=1)
        df["subaxial"] = df[CervicalSpineFractures.__subaxial_region].any(axis=1)
        df["n_subaxial"] = df[CervicalSpineFractures.__subaxial_region].sum(axis=1)
        df["total"] = df[CervicalSpineFractures.__vertebrae].sum(axis=1)
        return df

    def _summarize(self) -> dict:
        """Extracts summary and descriptive statistics from the data."""
        d = {}
        # Study level
        d["n_patients"] = self._df.shape[0]
        d["n_fractures"] = self._df["total"].sum()
        d["n_craniovertebral"] = self._df["n_craniovertebral"].sum()
        d["n_subaxial"] = self._df["n_subaxial"].sum()

        # Patient Level
        d["n_patients_fracture"] = self._df["patient_overall"].sum()
        d["p_patients_fracture"] = round(d["n_patients_fracture"] / d["n_patients"] * 100, 2)
        d["n_patients_craniovertebral"] = self._df["craniovertebral"].sum()
        d["p_patients_craniovertebral"] = round(
            d["n_patients_craniovertebral"] / d["n_patients"] * 100, 2
        )
        d["n_patients_subaxial"] = self._df["subaxial"].sum()
        d["p_patients_subaxial"] = round(d["n_patients_subaxial"] / d["n_patients"] * 100, 2)

        # Vertebrae Level
        df = self._df[CervicalSpineFractures.__vertebrae].copy()
        df1 = df.sum(axis=0).to_frame().rename(columns={0: "Number of Fractures"})
        df1["Percent of Fractures"] = round(df1["Number of Fractures"] / d["n_fractures"] * 100, 2)
        d["fractures_by_vertebrae"] = df1

        df = self._df[["n_craniovertebral", "n_subaxial"]].copy()
        df1 = df.sum(axis=0).to_frame().rename(columns={0: "Number of Fractures"})
        df1["Percent of Fractures"] = round(df1["Number of Fractures"] / d["n_fractures"] * 100, 2)
        d["fractures_by_region"] = df1

        return d
