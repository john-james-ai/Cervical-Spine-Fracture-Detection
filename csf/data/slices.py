#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /slices.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday September 17th 2022 01:36:54 am                                            #
# Modified   : Saturday September 17th 2022 04:47:36 am                                            #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #


# ------------------------------------------------------------------------------------------------ #
# Source: Pydicom Examples
# Link: https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_printing_dataset.html#sphx-glr-auto-examples-input-output-plot-printing-dataset-py
def print_slice(dataset, indent=0):
    """Go through all items in the dataset and print them with custom format

    Modelled after Dataset._pretty_str()
    """
    dont_print = ["Pixel Data", "File Meta Information Version"]

    indent_string = "   " * indent
    next_indent_string = "   " * (indent + 1)

    for data_element in dataset:
        if data_element.VR == "SQ":  # a sequence
            print(indent_string, data_element.name)
            for sequence_item in data_element.value:
                print_slice(sequence_item, indent + 1)
                print(next_indent_string + "---------")
        else:
            if data_element.name in dont_print:
                print("""<item not printed -- in the "don't print" list>""")
            else:
                repr_value = repr(data_element.value)
                if len(repr_value) > 50:
                    repr_value = repr_value[:50] + "..."
                print("{0:s} {1:s} = {2:s}".format(indent_string, data_element.name, repr_value))
