"""Configuration parameters for the training of the ML model: main_train.py"""
import os
from os.path import join
import time
import copy

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIMESTAMP = time.strftime("%Y%m%d-%H%M")

# running setup
CONFIG_TR = {
    # GENERAL
    # directories
    "scenario": "",
    "MAIN_DIR": MAIN_DIR,
    "TIMESTAMP": TIMESTAMP,

    # scenarios
    "dir_inter": join(MAIN_DIR, "inter"),
    "dir_model": join(MAIN_DIR, "inter", "ml"),
    "dir_output": join(MAIN_DIR, "output", "ml"),
    "PREF": "xgb_v1_",
    "model_name": "xgb_v1",

    "fp_dataset": join(MAIN_DIR, "inter", "ml", "dataset_v4.json"),
    "fp_labels": join(MAIN_DIR, "inter", "ml", "labels_v4_cor.csv"),

    "taxonomy_path":join(MAIN_DIR, "inter", "ml", "taxonomy.csv"),
    "unique_words":join(MAIN_DIR, "inter", "ml", "vocab_unique_words.csv"),

    "fp_preprocessed": join(MAIN_DIR, "inter", "train_preproc.csv"),

    # parameters
    "feature": "root",  # word or root
    # "test_size": 0.4,
    "n_split": 20,
    "threshold_classification":0.5,
    "lr": 0.3,
    "n_estimators": 50,
    "max_depth": 5,
    "min_child_weight": 1,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "reg_lambda": 0,

    "SVC_C":1.0,
    "SVC_kernel":"rbf",
    "SVC_degree": 3,

    # steps
    # "DO_PREPROC": True,
    "DO_ML": True,
    # "DO_ML": False,
    "DO_ML_train": True,
    # "DO_ML_train": False,
    "DO_ML_predict": True,
    # "DO_ML_predict": False,
    "DO_ML_full_train": True,
    # "DO_ML_full_train": False,

    "DO_PERFORMANCE": True,
    # "DO_PERFORMANCE": False,

    # DEBUG
    "SAMPLING": None,
    # "SAMPLING": 100,
}



class ConfigTr:
    def __init__(self):
        self.params = copy.deepcopy(CONFIG_TR)

    def set(self, param_name, value):
        self.params[param_name] = value

    def get(self, param_name):
        return self.params[param_name]


CFG_TR = ConfigTr()
