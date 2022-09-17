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
# Created    : Saturday September 17th 2022 12:36:04 am                                            #
# Modified   : Saturday September 17th 2022 12:50:29 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------ #


@dataclass(repr=True)
class Study:
    id: str
    patient_overall: int
    C1: int
    C2: int
    C3: int
    C4: int
    C5: int
    C6: int
    C7: int
    total: int

    def __str__(self) -> str:
        return "\nStudy Id: {}\n\tPatient Overall: {}\n\tNum Fractures {}\n\tC1: {}\n\tC2: {}\n\tC3: {}\n\tC4: {}\n\tC5: {}\n\tC6: {}\n\tC7: {}\n".format(
            self.id,
            self.patient_overall,
            self.total,
            self.C1,
            self.C2,
            self.C3,
            self.C4,
            self.C5,
            self.C6,
            self.C7,
        )
