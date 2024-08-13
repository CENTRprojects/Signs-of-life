"""OLD: script performing ML steps
-> Refer to retraining/main_retraining.py for all things ML"""
from os.path import join
import numpy as np
import pandas as pd
import re
import string
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from ml.config_train import CONFIG_TR

SET_PUNCTUATION = set(list(string.punctuation))
pat_punct = string.punctuation.replace("_", "").replace("-", "")
pat_spe = "\n\t\r "
NON_TOK_PAT = "[" + pat_punct + pat_spe + "]+"
RE_NON_TOK_PAT = re.compile(NON_TOK_PAT)

MIN_TEXT_LG = 3

PRED_COLS = ["id", "url", "actual", "pred", "filtered_text"]


def get_reference_words():
    """Get reference vocabulary for parking (in all languages) from taxonomy.csv"""
    def remove_to(x):
        if x.startswith("to "):
            return x[3::]
        else:
            return x

    fp_trans = join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "taxonomy.csv")
    df_trans = pd.read_csv(fp_trans, encoding="utf-8")

    for colo in ["root", "word", "trs_direct"]:
        df_trans[colo] = df_trans[colo].apply(str.lower)

    # df_trans_list = pd.melt(df_trans, id_vars=["root", "word"], var_name="lgg", value_name="local_word")
    # df_trans_list["local_word"] = df_trans_list["local_word"].apply(str.lower)
    # df_trans_list["word"] = df_trans_list["word"].apply(remove_to)

    # DICO_TRANSLATION = dict(zip(df_trans_list["local_word"], df_trans_list[CONFIG_TR["feature"]]))
    DICO_TRANSLATION = dict(zip(df_trans["trs_direct"], df_trans[CONFIG_TR["feature"]]))
    VOCABULARY = sorted(list(df_trans[CONFIG_TR["feature"]].drop_duplicates()))

    # spe words
    spe_words = pd.read_csv(join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "vocab_unique_words.csv"))
    for i, row in spe_words.iterrows():
        wd = row[CONFIG_TR["feature"]]
        VOCABULARY.append(wd)
        DICO_TRANSLATION[wd] = wd

    print("\tVocabulary length:{}".format(len(VOCABULARY)))
    return DICO_TRANSLATION, VOCABULARY

DICO_TRANSLATION, VOCABULARY = get_reference_words()


def translate(w):
    """Tanslate a word only if relevant for parking classification"""
    if w in DICO_TRANSLATION:
        return DICO_TRANSLATION[w]
    else:
        return w

def tokenize_normalize_translate(txt):
    """Split by words using alphanumeric/dash/underscore, normalize and translate only parking words"""
    txt = RE_NON_TOK_PAT.split(txt)

    # clean-up tokenized
    txt = [e for e in txt if (len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION)]

    # lower
    txt = [e.lower() for e in txt]

    # partial translation
    txt = [translate(e) for e in txt]
    return txt


def get_relevant_words(x):
    """gather parking words"""
    relev = []
    for wd in x:
        if wd in DICO_TRANSLATION:
            relev.append(DICO_TRANSLATION[wd])
    return relev


def add_url_token(txt, url):
    txt = re.sub(("www." + url).replace(".", "\."), "x_url", txt, flags=re.IGNORECASE)
    txt = re.sub(url.replace(".", "\."), "x_url", txt, flags=re.IGNORECASE)
    return txt


def text_to_tokens(df):
    """tokenize text"""
    # Remove null
    ind_null = df["text"].isnull()
    print("\t--- N null text before token preproc :{} --> removed".format(np.sum(ind_null)))
    df = df[~ind_null]
    df = df.reset_index(drop=True)

    # Add URL token
    df["text"] = df[["text", "url"]].apply(lambda x: add_url_token(x[0], x[1]), axis=1)

    # Tokenizations
    df["text"] = df["text"].apply(tokenize_normalize_translate)
    df["lg_text"] = df["text"].apply(len)

    # filtered text
    df["filtered_text"] = df["text"].apply(get_relevant_words)

    # Remove empty
    ind_empty = df["text"].apply(lambda x: len(x) < MIN_TEXT_LG)
    print("\t--- N short text after token preproc :{} --> removed".format(np.sum(ind_empty)))
    df = df[~ind_empty]
    df = df.reset_index(drop=True)

    return df


def split_dataset(df, target_col):
    """split dataset into train and test sets"""
    df = shuffle(df, random_state=123)

    test_size = CONFIG_TR["test_size"]
    lg_train = int((1 - test_size) * len(df))
    X_train = df[0:lg_train]
    y_train = df[target_col][0:lg_train]
    X_test = df[lg_train::]
    y_test = df[target_col][lg_train::]
    print("INFO---Dataset size---\nINFO---Train: \t{}\nINFO---Test: \t{}".format(len(X_train), len(X_test)))
    return X_test, X_train, y_test, y_train


def train_classif_models(df):
    """Main function to train TFIDF + ML Classfier"""
    df_pred_full = df_pred_tr_full = None
    df = df.reset_index(drop=True)

    # load vocabulary
    vocab = []

    # split dataset
    target_col = "target"
    feat_col = "text"

    # features conversion to numeric
    vectorizer = CountVectorizer(analyzer='word',
                                 vocabulary=VOCABULARY
                                 )
    # X_test, X_train, y_test, y_train = split_dataset(df, target_col)

    skf = StratifiedKFold(n_splits=CONFIG_TR["n_split"])
    # mdl = XGBClassifier(**PARAM_INIT_XGBC)
    mdl = LogisticRegression(penalty='l1', solver='liblinear')

    i_set = 0
    for train_index, test_index in skf.split(df, df[target_col]):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = df.loc[train_index], df.loc[test_index]
        y_train, y_test = df.loc[train_index, target_col], df.loc[test_index, target_col]

        if CONFIG_TR["DO_ML_train"]:
            print("INFO:---DO_ML_train---")

            x_tr_num = np.concatenate([vectorizer.fit_transform(X_train[feat_col].astype(np.str)).toarray(),
                                       np.expand_dims(X_train["lg_text"].values, axis=1)], axis=1)
            xx = csr_matrix(x_tr_num)
            # xx = csr_matrix(vectorizer.fit_transform(X_train[feat_col].astype(np.str)).toarray())
            mdl.fit(xx, list(y_train))

            # # save vectorizer
            # with open(join(CONFIG_TR["dir_model"], CONFIG_TR["PREF"] + "tfidf_vecto.pkl"),'wb+') as f:
            #     pickle.dump(vectorizer, f)
            # # save model
            # with open(join(CONFIG_TR["dir_model"], CONFIG_TR["PREF"] + "model.pkl"),'wb+') as f:
            #     pickle.dump(mdl, f)

        if CONFIG_TR["DO_ML_predict"]:
            print("INFO:---DO_ML_predict---")

            # # load models
            # mdl_fpath = join(CONFIG_TR["dir_model"], CONFIG_TR["PREF"] + "model.pkl")
            # with open(mdl_fpath, 'rb+') as f:
            #     mdl = pickle.load(f)

            # # vecto
            # vectorizer_fpath = join(CONFIG_TR["dir_model"], CONFIG_TR["PREF"] + "tfidf_vecto.pkl")
            # with open(vectorizer_fpath, 'rb+') as f:
            #     vectorizer = pickle.load(f)

            # pred tr
            # xx = csr_matrix(vectorizer.transform(X_train[feat_col].astype(np.str)).toarray())
            preds = list(mdl.predict_proba(xx)[:, 1])
            ids = list(y_train.index)
            urls = list(X_train["url"])
            # texts = list(X_train[feat_col])
            texts = list(X_train["filtered_text"])
            df_pred_tr = pd.DataFrame(list(zip(ids, urls, list(y_train), preds, texts)), columns=PRED_COLS)

            # predict te
            X_test_values = np.concatenate([vectorizer.fit_transform(X_test[feat_col].astype(np.str)).toarray(),
                                            np.expand_dims(X_test["lg_text"].values, axis=1)], axis=1)
            # X_test_values = vectorizer.transform(X_test[feat_col].astype(np.str)).toarray()
            xx = csr_matrix(X_test_values)
            preds = list(mdl.predict_proba(xx)[:, 1])
            X_test_values = pd.DataFrame(X_test_values, columns=VOCABULARY + ["lg_text"])

            ids = list(y_test.index)
            urls = list(X_test["url"])
            # texts = list(X_test[feat_col])
            texts = list(X_test["filtered_text"])
            df_pred = pd.DataFrame(list(zip(ids, urls, list(y_test), preds, texts)), columns=PRED_COLS)

            if df_pred_full is None:
                df_pred_full = df_pred.copy(deep=True)
            else:
                df_pred_full = pd.concat([df_pred_full, df_pred], axis=0, sort=False)

            if df_pred_tr_full is None:
                df_pred_tr_full = df_pred_tr.copy(deep=True)
            else:
                df_pred_tr_full = pd.concat([df_pred_tr_full, df_pred_tr], axis=0, sort=False)

    df_pred_full = df_pred_full.reset_index(drop=True)
    df_pred_full["pred_categ"] = df_pred_full["pred"].apply(lambda x: int(x > CONFIG_TR["threshold_classification"]))
    df_pred_full.to_csv(join(CONFIG_TR["dir_output"], CONFIG_TR["PREF"] + "predictions.csv"), index=False)

    return df_pred_full, df_pred_tr_full, mdl, X_test_values, y_test


if __name__ == '__main__':
    # wait_before_start()
    txt = "metalldetektoren-bayern.de.Diese Website wurde eingestellt!...Rieger, Vertrieb von Metalldetektoren, Bayern, Tesoro, XP, Fisher"
    txt_mid = tokenize_normalize_translate(txt)
    print(txt_mid)
    print(get_relevant_words(txt_mid))
