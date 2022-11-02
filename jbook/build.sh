#!/usr/bin/sh
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Cervical Spine Fracture Detection                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /build.sh                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/Cervical-Spine-Fracture-Detection                  #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 12th 2022 08:00:10 am                                             #
# Modified   : Wednesday October 12th 2022 08:01:45 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #

# Delete prior build
echo "Removing prior jupyter-book build artifacts..."
rm -r jbook/_build/
#Prepare notebook display customizations
echo "Preparing notebook tags..."
python3 jbook/prep_notebooks.py
# Rebuilds the book
echo "Building book..."
jb build jbook/
# Commit book to gh-pages
echo "Committing changes to github pages..."
ghp-import -o -n -p -f jbook/_build/html