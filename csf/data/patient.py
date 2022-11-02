#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /patient.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday September 13th 2022 06:23:18 pm                                             #
# Modified   : Tuesday October 25th 2022 01:35:30 pm                                               #
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
from typing import Union
from csf import FIG_SIZE

# ------------------------------------------------------------------------------------------------ #


@pytest.mark.skiptest
class CTResults:
    """Collection of patient outcomes and the C1,C7 and Overall fracture targets.

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

    def __init__(self, df: str) -> None:
        self._df = df
        self._df["total"] = self._df[CTResults.__vertebrae].sum(axis=1)

    @property
    def info(self) -> pd.DataFrame:
        d = {}
        d["Columns"] = self._df.columns
        d["Dtype"] = self._df.dtypes.values
        d["Non-Null Count"] = self._df.count().values
        d["Null Count"] = self._df.isnull().sum(axis=0).values
        d["Minimum"] = self._df.min(axis=0, numeric_only=None).values
        d["Maximum"] = self._df.max(axis=0, numeric_only=None).values
        d["Num Unique"] = self._df.nunique(axis=0).values
        d["Memory Usage"] = self._df.memory_usage(index=False, deep=True).values
        df = pd.DataFrame(data=d)
        return df

    @property
    def n_patients(self) -> int:
        return self._df["StudyInstanceUID"].nunique()

    @property
    def n_scans(self) -> int:
        return self._df.shape[0]

    @property
    def n_fractures(self) -> int:
        return int(self._df["total"].sum())

    @property
    def n_craniovertebral(self) -> int:
        return int(self._df[CTResults.__craniovertebral_region].sum(axis=1).sum())

    @property
    def p_craniovertebral(self) -> int:
        return float(
            round(
                self._df[CTResults.__craniovertebral_region].sum().sum() / self._df.shape[0] * 100,
                2,
            )
        )

    @property
    def n_subaxial(self) -> int:
        return int(self._df[CTResults.__subaxial_region].sum().sum())

    @property
    def p_subaxial(self) -> float:
        return float(
            round(
                self._df[CTResults.__subaxial_region].sum(axis=1).sum() / self._df.shape[0] * 100, 2
            )
        )

    @property
    def n_patients_with_fracture(self) -> int:
        return self._df[self._df["total"] > 0].shape[0]

    @property
    def p_patients_with_fracture(self) -> float:
        return round(self._df[self._df["total"] > 0].shape[0] / self.n_patients * 100, 2)

    @property
    def n_fractures_by_vertebrae(self) -> pd.DataFrame:
        df = self._df[CTResults.__vertebrae].copy()
        return df.sum(axis=0).to_frame().rename(columns={0: "Number of Fractures"})

    @property
    def n_fractures_by_region(self) -> pd.DataFrame:
        d = {}
        d["n_craniovertebral"] = self._df[CTResults.__craniovertebral_region].sum(axis=1).sum()
        d["n_subaxial"] = self._df[CTResults.__subaxial_region].sum(axis=1).sum()
        return pd.DataFrame(data=d, index=[0]).T.rename(columns={0: "Number of Fractures"})

    @property
    def n_patients_by_fracture_count(self) -> pd.DataFrame:
        df = self._df[CTResults.__vertebrae].copy()
        return df.sum(axis=0).to_frame().rename(columns={0: "Number of CTResults"})

    def sample(self, n: int = 5) -> pd.DataFrame:
        rng = default_rng()
        indices = rng.integers(low=0, high=self._df.shape[0], size=n)
        return self._df.loc[indices]

    def get_scan_by_fracture_count(self, n: int = 1, random_state: Union[bool, int] = None) -> dict:
        """Returns a sample scan object with n fractures

        Args:
            n (int): Number fractures in [0,6]
        Returns:
            Scan object with Study id, and C1,C7 fracture data.
        """
        df = self._df[(self._df["total"] == n)]
        return df.sample(n=1, replace=False, random_state=random_state, axis=0)

    def patient_diagnoses_plot(self, figsize: tuple = FIG_SIZE) -> None:
        """Barchart of the number of CTResults with and without fractures.

        Args:
            figsize (tuple): Height and width of plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.countplot(data=self._df, x="patient_overall")
        self._annotate_bars(ax=ax, total=self._df.shape[0], nudge=0.05)

        ax.set_title("Patient Diagnosis Plot")
        ax.set_xticks([0, 1], ["No Fracture", "Fracture"])

        plt.show()

    def fractures_plot(
        self, figsize: tuple = FIG_SIZE, title: str = "Fracture Distribution"
    ) -> None:
        """Barplot showing number of fractures (non_fractures) by vertebrae

        Args:
            figsize (tuple): Height and width of plot
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        gs = GridSpec(1, 2, width_ratios=[1, 2])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        sns.barplot(
            x=self.n_fractures_by_region.index,
            y=self.n_fractures_by_region["Number of Fractures"],
            ax=ax0,
            palette="Blues",
        )
        sns.barplot(
            x=self.n_fractures_by_vertebrae.index,
            y=self.n_fractures_by_vertebrae["Number of Fractures"],
            ax=ax1,
            palette="Blues",
        )

        ax0.set_title("Fractures by Region")
        ax0.set_xlabel("Region")
        ax0.set_xticks([0, 1], ["Craniovertebral", "Subaxial"])
        self._annotate_bars(ax=ax0, total=self._df["total"].sum(), nudge=0.15)

        ax1.set_title("Fractures by Vertebrae")
        ax1.set_xlabel("Vertebrae")
        self._annotate_bars(ax=ax1, total=self._df["total"].sum(), nudge=0.15)

    def patient_fracture_count_plot(self, figsize: tuple = FIG_SIZE) -> None:
        """Distribution of fracture counts among CTResults.

        Args:
            figsize (tuple): Height and width of plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.countplot(data=self._df, x="total", palette="Blues_r")
        self._annotate_bars(ax=ax, total=self._df.shape[0], nudge=0.07)

        ax.set_title("Patient Fracture Count Distribution Plot")

    def fracture_correlation_plot(self, figsize: tuple = FIG_SIZE) -> None:
        """Plots the correlation among fractures at various sites.

        Args:
            figsize (tuple): Height and width of plot
        """
        # Compute the correlation matrix
        corr = self._df[CTResults.__vertebrae].corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)

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
            annot=True,
            fmt=".1%",
            ax=ax,
            cbar_kws={"shrink": 0.5},
        )

        ax.set_title("Fracture Site Correlation Plot")

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
