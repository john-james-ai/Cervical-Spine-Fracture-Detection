#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /test_patients.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday September 13th 2022 08:32:23 pm                                             #
# Modified   : Saturday October 15th 2022 09:39:17 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pytest
import pandas as pd
import logging
import logging.config

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.CTResults
class TestCTResults:
    def test_info(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.info, "\n")
        assert isinstance(csf_test.info, pd.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_CTResults(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} CTResults.\n".format(csf_test.n_CTResults))
        assert isinstance(csf_test.n_CTResults, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_scans(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} scans.\n".format(csf_test.n_scans))
        assert isinstance(csf_test.n_scans, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_fractures(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} fractures.\n".format(csf_test.n_fractures))

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_craniovertebral(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} craniovertebral fractures.\n".format(csf_test.n_craniovertebral))
        assert isinstance(csf_test.n_craniovertebral, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_subaxial(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} subaxial fractures.\n".format(csf_test.n_subaxial))
        assert isinstance(csf_test.n_subaxial, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_CTResults_with_fracture(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} CTResults with fractures.\n".format(csf_test.n_CTResults_with_fracture)
        )
        assert isinstance(csf_test.n_CTResults_with_fracture, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_CTResults_with_fracture(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of CTResults with fractures.\n".format(
                csf_test.p_CTResults_with_fracture
            )
        )
        assert isinstance(csf_test.p_CTResults_with_fracture, float)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_CTResults_craniovertebral(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} CTResults with craniovertebral fractures.\n".format(
                csf_test.n_craniovertebral
            )
        )
        assert isinstance(csf_test.n_craniovertebral, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_CTResults_craniovertebral(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of CTResults with craniovertebral fractures.\n".format(
                csf_test.p_craniovertebral
            )
        )
        assert isinstance(csf_test.p_craniovertebral, float)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_CTResults_subaxial(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} CTResults with subaxial fractures.\n".format(csf_test.n_subaxial))
        assert isinstance(csf_test.n_subaxial, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_subaxial(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of CTResults with subaxial fractures.\n".format(
                csf_test.p_subaxial
            )
        )
        assert isinstance(csf_test.p_subaxial, float)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_fractures_by_vertebrae(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.n_fractures_by_vertebrae, "\n")
        assert isinstance(csf_test.n_fractures_by_vertebrae, pd.core.frame.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_fractures_by_region(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.n_fractures_by_region, "\n")
        assert isinstance(csf_test.n_fractures_by_region, pd.core.frame.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_samples(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.sample(), "\n")
        assert isinstance(csf_test.sample(), pd.core.frame.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_CTResults_by_fracture_count(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        CTResults = csf_test.n_CTResults_by_fracture_count
        assert isinstance(CTResults, pd.core.frame.DataFrame)
        print(CTResults)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_plots(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        csf_test.patient_diagnoses_plot()
        csf_test.fractures_plot()
        csf_test.patient_fracture_count_plot()
        csf_test.fracture_correlation_plot()
        assert True

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
