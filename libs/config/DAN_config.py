#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : DAN_config.py
import os

from easydict import EasyDict

OPTION = EasyDict()
OPTION.ROOT_PATH = os.path.join(os.path.expanduser('~'), 'my_farm/fsvos/DANet')
OPTION.SNAPSHOTS_DIR = os.path.join(OPTION.ROOT_PATH, 'workdir')
OPTION.DATASETS_DIR = os.path.join(OPTION.ROOT_PATH, 'data')

OPTION.TRAIN_SIZE = (241, 425)
OPTION.TEST_SIZE = (241, 425)
