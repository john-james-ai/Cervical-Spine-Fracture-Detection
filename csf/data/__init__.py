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
# Modified   : Saturday September 17th 2022 12:37:44 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-Clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from .base import Study

# ------------------------------------------------------------------------------------------------ #
segmentations = "data/raw/segmentations"
train_images = "data/raw/train_images"
test_images = "data/raw/test_images"
