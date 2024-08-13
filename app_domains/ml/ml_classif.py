""""""
from os.path import join
import numpy as np
import pandas as pd
import pickle
import re
import string
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from xgboost import XGBClassifier

# GLOBAL VARS----------------------
from ml.config_train import CONFIG_TR
from ref_sent import SAMPLE_POSITIVE, SAMPLE_NEGATIVE

# tokenizers
from inltk.inltk import setup as ind_setup

print("loading indian language models")
ind_setup("hi")
print("--> hi done")
ind_setup("pa")
ind_setup("gu")
ind_setup("kn")
ind_setup("ml")
ind_setup("mr")
ind_setup("bn")
ind_setup("ta")
ind_setup("ur")
ind_setup("ne")
print("--> all done")

# from tokenizers import SENT_TOK_ID_TO_TOKENIZER, WORD_TOK_ID_TO_TOKENIZER
from tokenizers import *

# tokenizers to use
df_lgg_to_tok = pd.read_csv(join(CONFIG_TR["MAIN_DIR"], "input", "lgg_tokenizers.csv"))
DICO_LGG_TO_WORD_TOK_ID = dict(zip(list(df_lgg_to_tok["lgg"]), list(df_lgg_to_tok["word_tokenizer"])))
DICO_LGG_TO_SENT_TOK_ID = dict(zip(list(df_lgg_to_tok["lgg"]), list(df_lgg_to_tok["sentence_tokenizer"])))

SET_PUNCTUATION = set(list(string.punctuation) + ["▁"])
pat_punct = string.punctuation.replace("_", "").replace("-", "")
pat_spe = "\n\t\r "
NON_TOK_PAT = "[" + pat_punct + pat_spe + "]+"
RE_NON_TOK_PAT = re.compile(NON_TOK_PAT)
TOK_SENT_SEP = "[.?:!;·˳̥۰ᴉ•․﮳。：；？！」』؟:]+"
RE_TOK_SENT_SEP = re.compile(TOK_SENT_SEP)

MIN_TEXT_LG = 3

PRED_COLS = ["id", "url", "actual", "pred", "filtered_text", "language"]

DICO_GGLE_TO_LDETECT = {"af": "af", "am": "am", "ar": "ar", "bg": "bg", "bn": "bn", "ca": "ca", "chr": "chr",
                        "cs": "cs", "cy": "cy", "da": "da", "de": "de", "el": "el", "en": "en", "en-gb": "en",
                        "es": "es", "et": "et", "eu": "eu", "fi": "fi", "fil": "fil", "fr": "fr", "gu": "gu",
                        "hi": "hi", "hr": "hr", "hu": "hu", "id": "id", "is": "is", "it": "it", "iw": "iw", "ja": "ja",
                        "kn": "kn", "ko": "ko", "lt": "lt", "lv": "lv", "ml": "ml", "mr": "mr", "ms": "ms", "nl": "nl",
                        "no": "no", "pl": "pl", "pt-br": "pt", "pt-pt": "pt", "ro": "ro", "ru": "ru", "sk": "sk",
                        "sl": "sl", "sr": "sr", "sv": "sv", "sw": "sw", "ta": "ta", "te": "te", "th": "th", "tr": "tr",
                        "uk": "uk", "ur": "ur", "vi": "vi", "zh": "zh-cn", "zh-cn": "zh-cn", "zh-tw": "zh-tw"}


def replace_dico(x, dico):
    if x in dico:
        return dico[x]
    else:
        return x

def get_reference_words():
    """"Parse taxonomy file to get the parking vocabulary
    DICO_TRANSLATION_BY_LGG = dict translations of by Language,
    VOCABULARY = the base vocabulary (before translation)
     DICO_WORD_TYPES = dictionary of words into their core/attribute category
     """
    fp_trans = CONFIG_TR["taxonomy_path"]
    df_trans = pd.read_csv(fp_trans, encoding="utf-8")
    # df_trans = pd.read_csv(fp_trans)

    for colo in ["root", "word", "trs_direct", "lgg"]:
        df_trans[colo] = df_trans[colo].apply(str.lower)

    df_trans["lgg"] = df_trans["lgg"].apply(lambda x: replace_dico(x, DICO_GGLE_TO_LDETECT))

    # df_trans[["lgg"]].to_csv(join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "google_code_to_langdetect.csv"))

    # df_trans["root"] = df_trans["root"].apply(lambda x:x.replace(" ", "_"))

    # DICO_TRANSLATION = dict(zip(df_trans_list["local_word"], df_trans_list[CONFIG_TR["feature"]]))

    VOCABULARY = sorted(list(df_trans[CONFIG_TR["feature"]].drop_duplicates()))
    DICO_WORD_TYPES = dict(zip(df_trans["trs_direct"], df_trans["word_type"]))
    DICO_WORD_TYPES.update(dict(zip(df_trans["root"], df_trans["word_type"])))

    #
    DICO_TRANSLATION_BY_LGG = {"other": {}}
    all_lggs = list(df_trans["lgg"].drop_duplicates())
    for lgg in all_lggs:
        ind_lgg = df_trans["lgg"] == lgg
        dico_lgg = dict(zip(df_trans.loc[ind_lgg, "trs_direct"], df_trans.loc[ind_lgg, CONFIG_TR["feature"]]))
        DICO_TRANSLATION_BY_LGG[lgg] = dico_lgg
        DICO_TRANSLATION_BY_LGG["other"].update(dico_lgg)

    # DICO_TRANSLATION_BY_LGG = dict(zip(df_trans["trs_direct"], df_trans[CONFIG_TR["feature"]]))

    # spe words
    spe_words = pd.read_csv(CONFIG_TR["unique_words"])
    for i, row in spe_words.iterrows():
        wd = row[CONFIG_TR["feature"]]
        VOCABULARY.append(wd)
        DICO_WORD_TYPES[wd] = row["word_type"]

        for k in DICO_TRANSLATION_BY_LGG.keys():
            DICO_TRANSLATION_BY_LGG[k].update({wd: wd})

    print("\tVocabulary length:{}".format(len(VOCABULARY)))
    return DICO_TRANSLATION_BY_LGG, VOCABULARY, DICO_WORD_TYPES


DICO_TRANSLATION_BY_LGG, VOCABULARY, DICO_WORD_TYPES = get_reference_words()

OTHER_FEATURES = ["n_sentences", "n_words", "n_pair_same", "n_pair_successive", "min_sent_size_same",
                  "min_sent_size_successive"]
FEATURES = VOCABULARY + OTHER_FEATURES

SPE_TOKENS = [
    ["index of", "index_of"],
    ["hello world", "hello_world"],
    ["lorem ipsum", "lorem_ipsum"],
    ["no website", "no_website"]
]


# -----------------------

def add_multi_word_tokens(txt):
    """Replace multiple words parking vocabulary into a one word token"""
    for pair in SPE_TOKENS:
        left = pair[0].split(" ")[0]
        right = pair[0].split(" ")[1]
        txt = re.sub(left + "[ .]+" + right, pair[1], txt, flags=re.IGNORECASE)
    return txt


def translate(w, lgg):
    """Dictionary based translation"""
    lgg_to_use = lgg
    if lgg not in DICO_TRANSLATION_BY_LGG:
        lgg_to_use = "other"

    if w in DICO_TRANSLATION_BY_LGG[lgg_to_use]:
        return DICO_TRANSLATION_BY_LGG[lgg_to_use][w]
    else:
        return w


def spit_sentences(txt):
    txt = RE_TOK_SENT_SEP.split(txt)
    return txt


def tokenize_and_correct(txt, lgg):
    """Split by words using alphanumeric/dash/underscore and check if in mispelled words"""

    # Sentence tokenization
    if lgg in DICO_LGG_TO_SENT_TOK_ID:
        sent_tokenizer = eval(SENT_TOK_ID_TO_TOKENIZER[DICO_LGG_TO_SENT_TOK_ID[lgg]])
    else:
        sent_tokenizer = eval(SENT_TOK_ID_TO_TOKENIZER["std"])

    # Word tokenization
    if lgg in DICO_LGG_TO_WORD_TOK_ID:
        word_tokenizer = eval(WORD_TOK_ID_TO_TOKENIZER[DICO_LGG_TO_WORD_TOK_ID[lgg]])
    else:
        word_tokenizer = eval(WORD_TOK_ID_TO_TOKENIZER["std"])

    txt = sent_tokenizer(txt)

    # clean-up sentences:
    txt = [e for e in txt if ((len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION))]

    txt = [word_tokenizer(sent) for sent in txt]
    # txt = RE_NON_TOK_PAT.split(txt)

    # clean-up tokenized
    txt = [[e.lower() for e in sent if (len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION)] for sent in txt]

    # partial translation
    txt = [[translate(e, lgg) for e in sent] for sent in txt]
    return txt


def get_relevant_words(x, lgg):
    """Collect identified parking words"""
    relev = []
    lgg_to_use = lgg
    if lgg not in DICO_TRANSLATION_BY_LGG:
        lgg_to_use = "other"

    for sent in x:
        for wd in sent:
            if wd in DICO_TRANSLATION_BY_LGG[lgg_to_use]:
                relev.append(DICO_TRANSLATION_BY_LGG[lgg_to_use][wd])
    return relev


def add_url_token(txt, url):
    """Replace the domain name into the token 'x_url'"""
    txt = re.sub(("www." + url).replace(".", "\."), "x_url", txt, flags=re.IGNORECASE)
    txt = re.sub(url.replace(".", "\."), "x_url", txt, flags=re.IGNORECASE)
    return txt


def get_sequence_pari_attr(txt, lgg):
    """Get count of Core/Attribute word in the same sentence and in successive sentence"""
    n_pair_same = n_pair_successive = 0
    min_sent_size_same = min_sent_size_successive = 1000
    lgg_to_use = lgg
    if lgg not in DICO_TRANSLATION_BY_LGG:
        print("missing {}".format(lgg))
        lgg_to_use = "other"


    n_core_prec = 0
    n_attr_prec = 0
    for sent in txt:
        n_core_curr = 0
        n_attr_curr = 0

        for word in sent:

            if word in DICO_TRANSLATION_BY_LGG[lgg_to_use]:
                type_word = DICO_WORD_TYPES[word]

                if type_word == "core":
                    n_core_curr += 1
                elif type_word == "attr":
                    n_attr_curr += 1

        if (n_attr_curr > 0) and (n_core_curr > 0):
            n_pair_same += 1
            if len(sent) < min_sent_size_same:
                min_sent_size_same = len(sent)

        elif (n_attr_curr > 0 and n_core_prec > 0) or (n_attr_prec > 0 and n_core_curr > 0):
            n_pair_successive += 1
            if len(sent) < min_sent_size_successive:
                min_sent_size_successive = len(sent)

        n_core_prec = n_core_curr
        n_attr_prec = n_attr_curr

    return n_pair_same, n_pair_successive, min_sent_size_same, min_sent_size_successive


def get_former_test_data():
    """Add fake v1's dataset"""
    # add test trains
    df_pos = pd.DataFrame(SAMPLE_POSITIVE)
    df_pos = df_pos.rename(columns={"txt": "text", "lg": "language"})
    df_pos["target"] = 1
    df_pos["is_former"] = True
    df_neg = pd.DataFrame(SAMPLE_NEGATIVE)
    df_neg = df_neg.rename(columns={"txt": "text", "lg": "language"})
    df_neg["target"] = 0
    df_pos["is_former"] = True
    return pd.concat([df_pos, df_neg], axis=0, sort=False).reset_index(drop=True)


def text_to_tokens(df):
    """Convert text into ready to use ML features"""
    df_lab = pd.read_csv(CONFIG_TR["fp_labels"])
    df = pd.merge(df, df_lab, how="left", on="url")

    df = df.rename(columns={"clean_text": "text", "ACT_SUB_CATEGORY": "target"})

    # add older dataset
    # df_former = get_former_test_data()
    # print("Former reference data added: {}".format(len(df_former)))
    # df = pd.concat([df, df_former], axis=0, sort=False).reset_index(drop=True)


    # Remove null
    ind_null = df["text"].isnull() | (df["text"] == "None")
    print("\t--- N null text before token preproc :{} --> removed".format(np.sum(ind_null)))
    df = df[~ind_null]
    df = df.reset_index(drop=True)

    # Add spe tokens
    # url
    df["text"] = df[["text", "url"]].apply(lambda x: add_url_token(x[0], x[1]), axis=1)

    # multi_words
    df["text"] = df["text"].apply(add_multi_word_tokens)

    # Tokenizations
    df["text"] = df[["text", "language"]].apply(lambda x: tokenize_and_correct(x[0], x[1]), axis=1)
    df["n_sentences"] = df["text"].apply(len)
    df["n_words"] = df["text"].apply(lambda x: sum([len(e) for e in x]))

    # count core/attribute
    df["n_core_attr_pairs"] = df[["text", "language"]].apply(lambda x: get_sequence_pari_attr(x[0], x[1]), axis=1)
    df["n_pair_same"] = df["n_core_attr_pairs"].apply(lambda x: x[0])
    df["n_pair_successive"] = df["n_core_attr_pairs"].apply(lambda x: x[1])
    df["min_sent_size_same"] = df["n_core_attr_pairs"].apply(lambda x: x[2])
    df["min_sent_size_successive"] = df["n_core_attr_pairs"].apply(lambda x: x[3])
    df = df.drop(["n_core_attr_pairs"], axis=1)

    # filtered text
    df["filtered_text"] = df[["text", "language"]].apply(lambda x: get_relevant_words(x[0], x[1]), axis=1)

    # Remove empty
    ind_empty = df["n_words"].apply(lambda x: x < MIN_TEXT_LG)
    print("\t--- N short text after token preproc :{} --> removed".format(np.sum(ind_empty)))
    df = df[~ind_empty]
    df = df.reset_index(drop=True)

    # conv
    df["target"] = df["target"].astype(int)

    return df


def split_dataset(df, target_col):
    """Split dataset into train/test sets"""
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
    """Main function to train CountVectorizer + Classfier"""
    df_pred_full = df_pred_tr_full = None
    df = df.reset_index(drop=True)

    # load vocabulary
    vocab = []

    # split dataset
    target_col = "target"
    feat_col = "text"
    other_feats = OTHER_FEATURES
    feats = VOCABULARY + other_feats

    # features conversion to numeric
    vectorizer = CountVectorizer(analyzer='word',
                                 vocabulary=VOCABULARY)
    # X_test, X_train, y_test, y_train = split_dataset(df, target_col)

    skf = StratifiedKFold(n_splits=CONFIG_TR["n_split"])

    PARAM_INIT_XGBC = {
        "max_depth": CONFIG_TR["max_depth"],
        "learning_rate": CONFIG_TR["lr"],
        "n_estimators": CONFIG_TR["n_estimators"],
        "objective": "binary:logistic",
        "booster": 'gbtree',
        # "n_jobs": 6,
        "n_jobs": -1,
        "nthread": None,
        "gamma": 0,
        "min_child_weight": CONFIG_TR["min_child_weight"],
        "max_delta_step": 0,
        "subsample": 1,
        "colsample_bytree": CONFIG_TR["colsample_bytree"],
        "colsample_bylevel": CONFIG_TR["colsample_bylevel"],
        "reg_alpha": 0,
        "reg_lambda": CONFIG_TR["reg_lambda"],
        "scale_pos_weight": 1,
        "base_score": 0.5,
        "random_state": 0,
        "seed": None,
        "missing": None}

    PARAM_INIT_SVC = {
        "C": CONFIG_TR["SVC_C"],
        "kernel": CONFIG_TR["SVC_kernel"],
        "degree": CONFIG_TR["SVC_degree"],
        "gamma": 'scale',
        "coef0": 0.0,
        "shrinking": True,
        "probability": True,
        "tol": 1e-3,
        "cache_size": 200,
        "class_weight": None,
        "verbose": False,
        "max_iter": -1,
        "decision_function_shape": 'ovr',
        "break_ties": False,
        "random_state": None
    }

    mdl = XGBClassifier(**PARAM_INIT_XGBC)
    # mdl = SVC(**PARAM_INIT_SVC)

    # balancing
    n_pos = df[target_col].sum()
    scale_pos_weight = (len(df) - n_pos) / n_pos
    mdl.set_params(**{"scale_pos_weight": scale_pos_weight})

    all_feats_test = []
    i_set = 0
    for train_index, test_index in skf.split(df, df[target_col]):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = df.loc[train_index], df.loc[test_index]
        y_train, y_test = df.loc[train_index, target_col], df.loc[test_index, target_col]

        if CONFIG_TR["DO_ML_train"]:
            print("INFO:---DO_ML_train---")

            x_tr_num = np.concatenate([vectorizer.fit_transform(X_train[feat_col].astype(np.str)).toarray(),
                                       X_train[other_feats].values], axis=1)
            xx = csr_matrix(x_tr_num)
            # xx = csr_matrix(vectorizer.fit_transform(X_train[feat_col].astype(np.str)).toarray())
            # mdl.fit(xx, list(y_train))
            mdl.fit(xx, list(y_train))

            # # save vectorizer
            # with open(join(CONFIG_TR["dir_model"], CONFIG_TR["PREF"] + "tfidf_vecto.pkl"),'wb+') as f:
            #     pickle.dump(vectorizer, f)
            # # save model
            # with open(join(CONFIG_TR["dir_model"], CONFIG_TR["PREF"] + "model.pkl"),'wb+') as f:
            #     pickle.dump(mdl, f)

        if CONFIG_TR["DO_ML_predict"]:
            print("INFO:---DO_ML_predict---")

            # # # load models
            # mdl_fpath = join(CONFIG_TR["MAIN_DIR"], "input","xgb_v5.pkl")
            # with open(mdl_fpath, 'rb+') as f:
            #     mdl = pickle.load(f)
            # # # vecto
            # vectorizer_fpath = join(CONFIG_TR["MAIN_DIR"], "input", "vecto_xgb_v5.pkl")
            # with open(vectorizer_fpath, 'rb+') as f:
            #     vectorizer = pickle.load(f)

            # pred tr
            # xx = csr_matrix(vectorizer.transform(X_train[feat_col].astype(np.str)).toarray())
            preds = list(mdl.predict_proba(xx)[:, 1])
            ids = list(y_train.index)
            urls = list(X_train["url"])
            # texts = list(X_train[feat_col])
            texts = list(X_train["filtered_text"])
            lggs = list(X_test["language"])
            df_pred_tr = pd.DataFrame(list(zip(ids, urls, list(y_train), preds, texts, lggs)), columns=PRED_COLS)

            # predict te
            X_test_values = np.concatenate([vectorizer.fit_transform(X_test[feat_col].astype(np.str)).toarray(),
                                            X_test[other_feats].values], axis=1)
            # X_test_values = vectorizer.transform(X_test[feat_col].astype(np.str)).toarray()
            all_feats_test.append((list(df.loc[test_index, "url"]), X_test_values.copy()))

            xx = csr_matrix(X_test_values)
            preds = list(mdl.predict_proba(xx)[:, 1])
            X_test_values = pd.DataFrame(X_test_values, columns=VOCABULARY + other_feats)

            ids = list(y_test.index)
            urls = list(X_test["url"])
            # texts = list(X_test[feat_col])
            texts = list(X_test["filtered_text"])
            lggs = list(X_test["language"])
            df_pred = pd.DataFrame(list(zip(ids, urls, list(y_test), preds, texts, lggs)), columns=PRED_COLS)

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

    # full training
    if CONFIG_TR["DO_ML_full_train"]:
        print("INFO:---DO_ML_train---")

        mdl = XGBClassifier(**PARAM_INIT_XGBC)
        mdl.set_params(**{"scale_pos_weight": scale_pos_weight})

        vectorizer = CountVectorizer(analyzer='word',
                                     vocabulary=VOCABULARY)

        x_tr_num = np.concatenate([vectorizer.fit_transform(df[feat_col].astype(np.str)).toarray(),
                                   df[other_feats].values], axis=1)
        xx = csr_matrix(x_tr_num)
        # xx = csr_matrix(vectorizer.fit_transform(X_train[feat_col].astype(np.str)).toarray())
        mdl.fit(xx, list(df[target_col]))

        # save
        with open(join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "models", CONFIG_TR["model_name"] + ".pkl"), "wb+") as f:
            pickle.dump(mdl, f)
        with open(join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "models", "vecto_" + CONFIG_TR["model_name"] + ".pkl"),
                  "wb+") as f:
            pickle.dump(vectorizer, f)

        print("full model saved")

    return df_pred_full, df_pred_tr_full, mdl, X_test_values, y_test, feats, all_feats_test

