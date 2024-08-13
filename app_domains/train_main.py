"""Script to train a new machine learning model"""
# Imports
import time
import os
from os.path import join
from random import shuffle

import pandas as pd
import json

from ml.config_train import CONFIG_TR
from ml_classif import train_classif_models, text_to_tokens, VOCABULARY
from performance import assess_performance

a = 1

def wait_before_start():
    while True:
        if os.path.isfile(join(CONFIG_TR["MAIN_DIR"], "output",  "lol.txt")):
            break
        else:
            time.sleep(60)


def summarize_performance(df_pred, df_pred_tr, mdl, X_test, y_test_cv):
    """Compute and generat prediction performance report"""
    print("INFO:---DO_performance---")
    # PRED_COLS = ["id", "url", "actual", "pred", "filtered_text"]
    y_train = df_pred_tr["actual"]
    y_pred_tr = df_pred_tr["pred"]

    y_test = df_pred["actual"]
    y_pred_te = df_pred["pred"]

    assess_performance(y_train, y_test, y_pred_tr, y_pred_te, mdl, VOCABULARY + ["lg_text"], X_test, y_test_cv)

    pass


def train():
    """Main function to train LV1 and LV2 classifiers + data exploration steps"""
    with open(CONFIG_TR["fp_dataset"],  "r", encoding='utf-8') as f:
        df = json.loads(f.read().encode('raw_unicode_escape').decode())
    shuffle(df)
    df = pd.DataFrame(df)

    if CONFIG_TR["SAMPLING"] is not None:
        df = df.head(CONFIG_TR["SAMPLING"])
    print("INFO data size :{}".format(len(df)))
    print("INFO data columns :{}".format(",".join(list(df.columns))))

    # preproc
    df = text_to_tokens(df)

    if CONFIG_TR["DO_ML"]:
        df_pred, df_pred_tr, mdl, X_test, y_test_cv = train_classif_models(df)

    if CONFIG_TR["DO_PERFORMANCE"]:

        # evaluate performance
        summarize_performance(df_pred, df_pred_tr, mdl, X_test, y_test_cv)

    print("INFO ----- ALL DONE---------")


if __name__ == '__main__':
    # wait_before_start()

    tic_tic = time.time()
    train()
    print("Total time in seconds (main): {}".format(time.time() - tic_tic))
