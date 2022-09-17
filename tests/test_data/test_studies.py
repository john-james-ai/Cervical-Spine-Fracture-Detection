#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /test_eda.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday September 13th 2022 08:32:23 pm                                             #
# Modified   : Saturday September 17th 2022 12:56:09 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pytest
import pandas as pd
import logging
import logging.config
from csf.data import Study

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


@pytest.mark.eda
class TestEDA:
    def test_info(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.info, "\n")
        assert isinstance(csf_test.info, pd.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_patients(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} patients.\n".format(csf_test.n_patients))
        assert isinstance(csf_test.n_patients, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_patient_scans(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} patient scans.\n".format(csf_test.n_patient_scans))
        assert isinstance(csf_test.n_patient_scans, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_scans(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        assert isinstance(csf_test.scans, list)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_patient_scans_found(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        assert isinstance(csf_test.patient_scans_found, list)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_patient_scans_found(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} patient scans found.\n".format(csf_test.n_patient_scans_found))
        assert isinstance(csf_test.n_patient_scans_found, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_patient_scans_found(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of the patient scans found.\n".format(
                csf_test.p_patient_scans_found
            )
        )
        assert isinstance(csf_test.p_patient_scans_found, (float, int))

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_scans(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} scans.\n".format(csf_test.n_scans))
        assert isinstance(csf_test.n_scans, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_fractures(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} fractures.\n".format(csf_test.n_fractures))
        assert isinstance(csf_test.n_fractures, int)

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

    def test_n_patients_with_fracture(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\nThere are {} patients with fractures.\n".format(csf_test.n_patients_with_fracture))
        assert isinstance(csf_test.n_patients_with_fracture, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_patients_with_fracture(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of patients with fractures.\n".format(
                csf_test.p_patients_with_fracture
            )
        )
        assert isinstance(csf_test.p_patients_with_fracture, float)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_patients_craniovertebral(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} patients with craniovertebral fractures.\n".format(
                csf_test.n_patients_craniovertebral
            )
        )
        assert isinstance(csf_test.n_patients_craniovertebral, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_patients_craniovertebral(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of patients with craniovertebral fractures.\n".format(
                csf_test.p_patients_craniovertebral
            )
        )
        assert isinstance(csf_test.p_patients_craniovertebral, float)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_n_patients_subaxial(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} patients with subaxial fractures.\n".format(
                csf_test.n_patients_subaxial
            )
        )
        assert isinstance(csf_test.n_patients_subaxial, int)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_p_patients_subaxial(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print(
            "\nThere are {} percent of patients with subaxial fractures.\n".format(
                csf_test.p_patients_subaxial
            )
        )
        assert isinstance(csf_test.p_patients_subaxial, float)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_fractures_by_vertebrae(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.fractures_by_vertebrae, "\n")
        assert isinstance(csf_test.fractures_by_vertebrae, pd.core.frame.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_fractures_by_region(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.fractures_by_region, "\n")
        assert isinstance(csf_test.fractures_by_region, pd.core.frame.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_samples(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        print("\n", csf_test.sample(), "\n")
        assert isinstance(csf_test.sample(), pd.core.frame.DataFrame)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_sample_study_by_fracture_count(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        study = csf_test.get_sample_study_by_fracture_count(n=2)
        print(study)
        assert isinstance(study, Study)

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_plots(self, caplog, csf_test):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        csf_test.patient_diagnoses_plot()
        csf_test.fractures_plot()
        csf_test.patient_fractures_plot()
        csf_test.patient_fracture_count_plot()
        csf_test.fracture_correlation_plot()
        assert True

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
