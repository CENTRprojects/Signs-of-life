"""Configuration parameters for the retraining"""
from os.path import join

from ml.retraining.imports import MAIN_DIR

SAMPLES_FOLDER = r"app_domains\ml\retraining\input\samples"
INIT_2K_SAMPLES = r"app_domains\ml\retraining\input\init_2K.json"
HISTORICAL_PERFORMANCE = r"app_domains\ml\retraining\perf_history.json"
SCREENSHOT_FOLDER = r"app_domains\ml\retraining\original_images"

# SAMPLING = 200  # limit number of samples for dev phase
SAMPLING = None
# SAVE_HIST = True
SAVE_HIST = SAMPLING is None

# data selection
VERSION_SELECTION = 1
N_TOP_DISAGREEMENT = 10000
N_MAX_LABELS = 2
MIN_PREVAILING_THRESH = 0.51

# training configuration
ML_DO_PERFS = True
# TRAIN_AT_LV2 = False  # if not, at LV3 (multi class classification)
TRAIN_AT_LV2 = True
# CROSS_VAL_N_SPLIT = 5
CROSS_VAL_N_SPLIT = 20
THRESHOLD_CLASSIFICATION = 0.5

CONFIG_FENG = {
    "MAIN_DIR": MAIN_DIR,
    "name_model": "xgb_v1",
    "taxonomy_path": join(MAIN_DIR, "input", "taxonomy.csv"),
    "unique_words": join(MAIN_DIR, "input", "vocab_unique_words.csv"),
    "feature": "root"
}
CONFIG_FENG_MULTI = {
    "MAIN_DIR": MAIN_DIR,
    "name_model": "xgb_multi_class_v1",
    "taxonomy_path": join(MAIN_DIR, "input", "taxonomy.csv"),
    "unique_words": join(MAIN_DIR, "input", "vocab_unique_words.csv"),
    "feature": "root"
}

DO_HYPERPARAM = False
# DO_HYPERPARAM = True
DO_FULL_TR = True
# DO_FULL_TR = False
VERSION_TOKENISATION = 1
VERSION_TRAINING = 1

# out
OUT_FOLDER_NAME = "retraining_{}"
OUT_EXCEL_REPORT_NAME = "report_{}.xlsx"
OUT_MODEL_NAME = "xgb_v1.pkl"
OUT_MULTICLASS_MODEL_NAME = "xgb_multiclass_v1.pkl"
OUT_HYPERPARAMERS_OPTI_NAME = "hyperparam_opti_{}.csv"

DICO_LABELS_TO_LV3 = {
    "high_content": 0,
    "parked_notice_registrar": 1,
    "parked_notice_individual_content": 1,
    "blocked": 2,
    "under_construction": 3,
    "starter": 3,
    "expired": 4,
    "index_of": 5,
    "blank_page": 5,
    "for_sale": 1,
    "reserved": 1
}

DICO_LABELS_TO_LV2 = {
    "high_content": 0,
    "parked_notice_registrar": 1,
    "parked_notice_individual_content": 1,
    "blocked": 1,
    "under_construction": 1,
    "starter": 1,
    "expired": 1,
    "index_of": 1,
    "blank_page": 1,
    "for_sale": 1,
    "reserved": 1
}

DICO_LABELS_TO_ENDCLASS = {
    "high_content": "",
    "parked_notice_registrar": "",
    "parked_notice_individual_content": "other",
    "blocked": 'blocked',
    "under_construction": 'construction',
    "starter": 'starter',
    "expired": 'expired',
    "index_of": 'index_of',
    "blank_page": 'other',
    "for_sale": 'sale',
    "reserved": 'reserved'
}

CONFIG_XGB = {
    "lr": 0.5,
    "n_estimators": 25,
    "max_depth": 9,
    "min_child_weight": 2,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "scale_pos_weight": 2.1,
    "reg_lambda": 0.1,
}

CONFIG_XGB_MCLASS = {
    "lr": 0.5,
    "n_estimators": 10,
    "max_depth": 5,
    "min_child_weight": 1,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    # "scale_pos_weight": 2.1,
    "reg_lambda": 0,
}




OPTI_XGB = {
    "lr": [0.03, 0.1, 0.3, 0.5],
    "n_estimators": [10, 25, 50, 75, 100, 150, 200],
    # "max_depth": 5,
    "max_depth": [5, 7, 9, 11],
    "min_child_weight": [1, 2],
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "scale_pos_weight": 2.1,
    "reg_lambda": [0, 0.1, 0.01],
}

OPTI_XGB_MCLASS = {
    "lr": [0.03, 0.1, 0.3, 0.5],
    "n_estimators": [10, 25, 50, 75, 100, 150, 200],
    # "max_depth": 5,
    "max_depth": [5, 7, 9, 11],
    "min_child_weight": [1, 2],
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "scale_pos_weight": 2.1,
    "reg_lambda": [0, 0.1, 0.01],
}

# assumptions
# errors
# 29
# 55
# regist
# 15 cct
# 14 gtld


ACCURACY_ERRORS = 0.9902
PREVALENCE_ERRORS_CCTLD = 0.29
PREVALENCE_ERRORS_GTLD = 0.55
ACCURACY_REGISTRARS = 0.98  # todo: update
PREVALENCE_REGISTRARS_CCTLD = 0.15
PREVALENCE_REGISTRARS_GTLD = 0.14

# October 2020
# Accuracy 90.2 %
# Precision 85.3 %
# Recall 88.7 %
# f1 score 87.0 %
# Confusion Matrix
# -----   1188 116
# -----   86 675



DEFAULT_COL = '00000000'

DICO_COLORS_TO_LV2 = {
    "high_content": '000000FF',
    "parked_notice_registrar": '00FF0000',
    "parked_notice_individual_content": '0000FF00',
    "blocked": '0000FFFF',
    "under_construction": '00FFFF00',
    "starter": '00FF00FF',
    "expired": '0088FF88',
    "index_of": '00808080',
    "blank_page": '00000000',
    "for_sale": '00FFFFFF',
    "reserved": '00F0F0F0',
    "unsure": '00101010',
}
