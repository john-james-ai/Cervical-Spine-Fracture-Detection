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
# Modified   : Sunday October 30th 2022 06:07:10 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""IO Module"""
import os
from abc import ABC, abstractmethod
import pandas as pd
import pickle
import tensorflow as tf
import nibabel as nib
import yaml
from typing import Any, Union
from csf.base.service import Service

# ------------------------------------------------------------------------------------------------ #


class IO(Service):
    def __init__(self, basedir: str) -> None:
        self._basedir = basedir

    def _is_write_safe(self, filepath, force: str = False) -> bool:
        return force or not os.path.exists(filepath)

    def read(self, filepath: str, **kwargs) -> Any:
        filepath = self._get_path(filepath)
        print(filepath)
        return self._read(filepath, **kwargs)

    def write(self, filepath: str, data: Any, force: bool = False, **kwargs) -> None:
        filepath = self._get_path(filepath)
        print(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._write(filepath=filepath, data=data, force=force, **kwargs)

    def _get_path(self, filepath: str) -> str:
        return os.path.join(self._basedir, filepath)

    @abstractmethod
    def _read(self, filepath: str, **kwargs) -> Any:
        pass

    @abstractmethod
    def _write(self, filepath: str, data: Any, force: bool = False, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                        CSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #


class CSVIO(IO):
    def __init__(self, basedir: str) -> None:
        super().__init__(basedir=basedir)

    def _read(
        self,
        filepath: str,
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: list = None,
        low_memory: bool = False,
    ) -> pd.DataFrame:

        try:
            return pd.read_csv(
                filepath, header=header, index_col=index_col, usecols=usecols, low_memory=low_memory
            )
        except FileNotFoundError() as e:
            print("File {} not found.\n{}".format(filepath, e))

    def _write(
        self,
        filepath: str,
        data: pd.DataFrame,
        sep: str = ",",
        index: bool = False,
        index_label: bool = None,
        encoding: str = "utf-8",
        force: bool = True,
    ) -> None:

        if self._is_write_safe(filepath=filepath, force=force):
            data.to_csv(filepath, sep=sep, index=index, index_label=index_label, encoding=encoding)


# ------------------------------------------------------------------------------------------------ #
#                                        YAML IO                                                   #
# ------------------------------------------------------------------------------------------------ #


class YamlIO(IO):
    def __init__(self, basedir: str) -> None:
        super().__init__(basedir=basedir)

    def _read(self, filepath: str, **kwargs) -> dict:

        with open(filepath, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise (e)

    def _write(self, filepath: str, data: Any, force: bool = True) -> None:

        if self._is_write_safe(filepath=filepath, force=force):
            with open(filepath, "w") as f:
                yaml.dump(data, f)


# ------------------------------------------------------------------------------------------------ #
#                                         PICKLE                                                   #
# ------------------------------------------------------------------------------------------------ #


class PickleIO(IO):
    def __init__(self, basedir: str) -> None:
        super().__init__(basedir=basedir)

    def _read(self, filepath: str, **kwargs) -> Any:

        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except pickle.PickleError() as e:
                raise (e)

    def _write(self, filepath: str, data: Any, force: bool = True) -> None:

        if self._is_write_safe(filepath=filepath, force=force):
            with open(filepath, "wb") as f:
                pickle.dump(data, f)


# ------------------------------------------------------------------------------------------------ #
#                                         NIBABEL                                                  #
# ------------------------------------------------------------------------------------------------ #


class NibabelIO(IO):
    def __init__(self, basedir: str) -> None:
        super().__init__(basedir=basedir)

    def _read(self, filepath: str, **kwargs) -> Any:

        return nib.load(filepath).get_fdata()

    def _write(self, filepath: str, data: Any, force: bool = True) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                         DICOM                                                    #
# ------------------------------------------------------------------------------------------------ #


class DicomIO(IO):
    def __init__(self, basedir: str) -> None:
        super().__init__(basedir=basedir)

    def _read(self, filepath: str, **kwargs) -> tf.uint16:
        return tf.io.read_file(filepath)

    def _write(self, filepath: str, data: Any, force: bool = True) -> None:
        pass


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
        "dcm": DicomIO,
    }

    def __init__(self, basedir: str) -> None:
        self._basedir = basedir

    def create(self, ftype: str, **kwargs) -> IO:
        try:
            return IOFactory.__io[ftype](self._basedir)
        except KeyError as e:
            raise ValueError("No IO object exists for file type {}\n{}.".format(ftype, e))
