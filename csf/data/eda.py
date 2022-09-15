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
# Modified   : Wednesday September 14th 2022 09:48:28 pm                                           #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

FIGSIZE = (12, 6)
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

    def patient_diagnoses_plot(self, figsize: tuple = FIGSIZE) -> None:
        """Barchart of the number of patients with and without fractures.

        Args:
            figsize (tuple): Height and width of plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.countplot(data=self._df, x="patient_overall")
        self._annotate_bars(ax=ax, total=self._df.shape[0], nudge=0.05)

        ax.set_title("Patient Diagnosis Plot")
        ax.set_xticks([0, 1], ["No Fracture", "Fracture"])

        plt.show()

    def fractures_plot(self, figsize: tuple = FIGSIZE) -> None:
        """Barplot showing number of fractures (non_fractures) by vertebrae

        Args:
            figsize (tuple): Height and width of plot
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Fracture Distribution")
        gs = GridSpec(1, 2, width_ratios=[1, 2])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        sns.barplot(
            x=self._summary["fractures_by_region"].index,
            y=self._summary["fractures_by_region"]["Number of Fractures"],
            ax=ax0,
            palette="Blues",
        )
        sns.barplot(
            x=self._summary["fractures_by_vertebrae"].index,
            y=self._summary["fractures_by_vertebrae"]["Number of Fractures"],
            ax=ax1,
            palette="Blues",
        )

        # total_fractures = self._summary["n_fractures"]
        ax0.set_title("Fractures by Region")
        ax0.set_xlabel("Region")
        ax0.set_xticks([0, 1], ["Craniovertebral", "Subaxial"])
        self._annotate_bars(ax=ax0, total=self._df["total"].sum(), nudge=0.15)

        ax1.set_title("Fractures by Vertebrae")
        ax1.set_xlabel("Vertebrae")
        self._annotate_bars(ax=ax1, total=self._df["total"].sum(), nudge=0.15)

    def patient_fractures_plot(self, figsize: tuple = FIGSIZE) -> None:
        """Barplot showing number of Patients  (non_fractures) by vertebrae

        Args:
            figsize (tuple): Height and width of plot
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Patient Fracture Distribution")
        gs = GridSpec(1, 2, width_ratios=[1, 2])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        sns.barplot(
            x=self._summary["fractures_by_region"].index,
            y=self._summary["fractures_by_region"]["Number of Fractures"],
            ax=ax0,
            palette="Blues",
        )
        sns.barplot(
            x=self._summary["fractures_by_vertebrae"].index,
            y=self._summary["fractures_by_vertebrae"]["Number of Fractures"],
            ax=ax1,
            palette="Blues",
        )
        ax0.set_title("Fractures by Region")
        ax0.set_xlabel("Region")
        ax0.set_xticks([0, 1], ["Craniovertebral", "Subaxial"])
        self._annotate_bars(ax=ax0, total=self._df.shape[0], nudge=0.15)

        ax1.set_title("Fractures by Vertebrae")
        ax1.set_xlabel("Vertebrae")
        self._annotate_bars(ax=ax1, total=self._df.shape[0], nudge=0.15)

    def patient_fracture_count_plot(self, figsize: tuple = FIGSIZE) -> None:
        """Distribution of fracture counts among patients.

        Args:
            figsize (tuple): Height and width of plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.countplot(data=self._df, x="total", palette="Blues_r")
        self._annotate_bars(ax=ax, total=self._df.shape[0], nudge=0.07)

        ax.set_title("Patient Fracture Count Distribution Plot")

    def fracture_correlation_plot(self, figsize: tuple = FIGSIZE) -> None:
        """Plots the correlation among fractures at various sites.

        Args:
            figsize (tuple): Height and width of plot
        """
        # Compute the correlation matrix
        corr = self._df[CervicalSpineFractures.__vertebrae].corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(250, 30, l=65, center="light", as_cmap=True)
        cmap = sns.color_palette("Blues", as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

    def _annotate_bars(self, ax: plt.Axes, total: int, nudge: float) -> plt.Axes:
        """Adds count and percent annotations to bar charts

        Args:
            ax (plt.Axes): The axes object to annotate
            total (int): The divisor for the percentage calculation
            nudge (float): The amount by which x is adjusted to center text.
        """
        # Format the number in the center of the bar.
        for container in ax.containers:
            ax.bar_label(container, label_type="center")
        # Format percentages at the top of the bar using patches.
        for p in ax.patches:
            pct = round(p.get_height() / total * 100, 1)
            text = "{}%".format(str(pct))
            x = p.get_x() + p.get_width() / 2 - nudge
            y = p.get_y() + p.get_height()
            ax.annotate(text, (x, y))

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

        # Fractures by Vertebrae
        df = self._df[CervicalSpineFractures.__vertebrae].copy()
        df1 = df.sum(axis=0).to_frame().rename(columns={0: "Number of Fractures"})
        df1["Percent of Fractures"] = round(df1["Number of Fractures"] / d["n_fractures"] * 100, 2)
        d["fractures_by_vertebrae"] = df1

        # Fractures by Region
        df = self._df[["n_craniovertebral", "n_subaxial"]].copy()
        df1 = df.sum(axis=0).to_frame().rename(columns={0: "Number of Fractures"})
        df1["Percent of Fractures"] = round(df1["Number of Fractures"] / d["n_fractures"] * 100, 2)
        d["fractures_by_region"] = df1

        return d
