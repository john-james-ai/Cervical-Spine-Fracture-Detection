#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.6                                                                              #
# Filename   : /io.py                                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 29th 2022 12:46:06 am                                              #
# Modified   : Tuesday November 1st 2022 11:35:31 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""IO Module"""
import os
from abc import abstractmethod
import pandas as pd
import pickle
import tensorflow as tf
import nibabel as nib
import yaml
from typing import Any, Union, List
from csf.base.service import Service

# ------------------------------------------------------------------------------------------------ #


class IO(Service):
    @classmethod
    def read(cls, filepath: str, **kwargs) -> Any:
        if os.path.exists(filepath):
            return cls._read(filepath, **kwargs)
        else:
            raise FileNotFoundError("File {} not found.".format(filepath))

    @classmethod
    @abstractmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        pass

    @classmethod
    def write(cls, filepath: str, data: Any, **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cls._write(filepath, data, **kwargs)

    @classmethod
    @abstractmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                        CSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #


class CSVIO(IO):
    @classmethod
    def _read(
        cls,
        filepath: str,
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        low_memory: bool = False,
    ) -> pd.DataFrame:
        return pd.read_csv(
            filepath, header=header, index_col=index_col, usecols=usecols, low_memory=low_memory
        )

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sep: str = ",",
        index: bool = False,
        index_label: bool = None,
        encoding: str = "utf-8",
    ) -> None:

        data.to_csv(filepath, sep=sep, index=index, index_label=index_label, encoding=encoding)


# ------------------------------------------------------------------------------------------------ #
#                                        YAML IO                                                   #
# ------------------------------------------------------------------------------------------------ #


class YamlIO(IO):
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> dict:

        with open(filepath, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise (e)

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        with open(filepath, "w") as f:
            yaml.dump(data, f)


# ------------------------------------------------------------------------------------------------ #
#                                         PICKLE                                                   #
# ------------------------------------------------------------------------------------------------ #


class PickleIO(IO):
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:

        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except pickle.PickleError() as e:
                raise (e)

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)


# ------------------------------------------------------------------------------------------------ #
#                                         NIBABEL                                                  #
# ------------------------------------------------------------------------------------------------ #


class NibabelIO(IO):
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        return nib.load(filepath).get_fdata()

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                         DICOM                                                    #
# ------------------------------------------------------------------------------------------------ #


class DicomIO(IO):
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> tf.uint16:
        return tf.io.read_file(filepath)

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                              H5                                                  #
# ------------------------------------------------------------------------------------------------ #


class H5IO(IO):
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> tf.keras.Model:
        return tf.keras.models.load_model(filepath)

    @classmethod
    def _write(cls, filepath: str, data: tf.keras.Model, **kwargs) -> None:
        data.save(filepath)


# ------------------------------------------------------------------------------------------------ #
#                                       IO FACTORY                                                 #
# ------------------------------------------------------------------------------------------------ #
class IOFactory:

    __io = {
        "csv": CSVIO,
        "yaml": YamlIO,
        "yml": YamlIO,
        "pkl": PickleIO,
        "pickle": PickleIO,
        "nii": NibabelIO,
        "nib": NibabelIO,
        "dcm": DicomIO,
        "h5": H5IO,
    }

    @classmethod
    def create(cls, ftype: str, **kwargs) -> IO:
        try:
            return IOFactory.__io[ftype]
        except KeyError as e:
            raise ValueError("No IO object exists for file type {}\n{}.".format(ftype, e))
