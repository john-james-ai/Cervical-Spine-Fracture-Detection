#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /__init__.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday September 5th 2022 04:51:01 pm                                               #
# Modified   : Tuesday September 13th 2022 10:53:15 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-Clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------------------------ #
sns.set_palette("Blues_r")
sns.set_style("whitegrid")
# ------------------------------------------------------------------------------------------------ #
FIGSIZE = (12, 8)
PALETTE = "Blues_r"
TITLE_FONTSIZE = 20


class Barplot:
    """Wraps Seaborn Barplot"""

    def plot(self, data: pd.DataFrame, x: str, y: str, hue: None, title: str = None) -> None:
        """Renders a seaborn barplot

        Args:
            data (pd.DataFrame): Dataset for plotting.
            x,y,hue (str): Names of variables for long-form plotting of data.
        """
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax = sns.barplot(data=data, x=x, y=y, hue=hue, palette=PALETTE, ax=ax)

        if title:
            ax.set_title(title, fontsize=TITLE_FONTSIZE)

        plt.tight_layout()
        plt.show()
