""" General imports and global variables including logging and busy application tracker"""

import threading
from collections import OrderedDict
import numpy as np
import pandas as pd
import datetime
import time
import random
import re
import sys
import os
from os.path import join
import pickle
import json
import traceback
import copy
from pytz import timezone
import openpyxl
import uuid
import shutil

import logging

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DBG = 4  # put 0 for no debug message


# logger
LOGGER = logging.getLogger('retrain_log')
LOGGER.setLevel(logging.DEBUG)
if len(LOGGER.handlers) == 0:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.log(level=logging.INFO, msg="-" * 20 + "\n" + "LOGGER START" + "\n" + "-" * 20 + "\n")


def add_to_log(mess, level=None):
    print(mess)
    if (level is None) or (level == "info"):
        LOGGER.log(level=logging.INFO, msg=mess)
    elif level == "warn":
        LOGGER.log(level=logging.WARNING, msg=mess)
    elif level == "error":
        LOGGER.log(level=logging.ERROR, msg=mess)
    elif level == "crit":
        LOGGER.log(level=logging.CRITICAL, msg=mess)
    else:
        LOGGER.log(level=logging.INFO, msg=mess)


class Log:
    """Configurable log list"""

    def __init__(self, expected_cols):
        self.logs = []
        self.expected_cols = expected_cols

    def add(self, dico: dict):
        dico_real = {}
        for k, v in dico.items():
            if k in self.expected_cols:
                dico_real[k] = v
            else:
                add_to_log("LOG ERROR: unexpected column {} --> ignored".format(k))
        self.logs.append(dico_real)

    def reinit(self):
        self.logs = []

    def get_logs(self):
        return self.logs

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


LG = Log(["time", "importance", "file", "category", "description"])
