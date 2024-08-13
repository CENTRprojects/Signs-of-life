import copy

from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from ml.feature_eng import Featurer
from ml.retraining.imports import *
from ml.retraining.config_retrain import *
from ml.retraining.utils import gtm, replace_dico
from performance import performance_binary_classification, performance_multiclass_classification

FP_TAXONOMY_PATH = join(MAIN_DIR, "inter", "ml", "taxonomy.csv")
FP_UNIQUE_WORDS = join(MAIN_DIR, "inter", "ml", "vocab_unique_words.csv")
FP_PREPROCESSED: join(MAIN_DIR, "inter", "train_preproc.csv")
FEATURE_COL = "root"  # word or root

DICO_GGLE_TO_LDETECT = {"af": "af", "am": "am", "ar": "ar", "bg": "bg", "bn": "bn", "ca": "ca", "chr": "chr",
                        "cs": "cs", "cy": "cy", "da": "da", "de": "de", "el": "el", "en": "en", "en-gb": "en",
                        "es": "es", "et": "et", "eu": "eu", "fi": "fi", "fil": "fil", "fr": "fr", "gu": "gu",
                        "hi": "hi", "hr": "hr", "hu": "hu", "id": "id", "is": "is", "it": "it", "iw": "iw", "ja": "ja",
                        "kn": "kn", "ko": "ko", "lt": "lt", "lv": "lv", "ml": "ml", "mr": "mr", "ms": "ms", "nl": "nl",
                        "no": "no", "pl": "pl", "pt-br": "pt", "pt-pt": "pt", "ro": "ro", "ru": "ru", "sk": "sk",
                        "sl": "sl", "sr": "sr", "sv": "sv", "sw": "sw", "ta": "ta", "te": "te", "th": "th", "tr": "tr",
                        "uk": "uk", "ur": "ur", "vi": "vi", "zh": "zh-cn", "zh-cn": "zh-cn", "zh-tw": "zh-tw"}


def get_reference_words():
    """"Parse taxonomy file to get the parking vocabulary
    DICO_TRANSLATION_BY_LGG = dict translations of by Language,
    VOCABULARY = the base vocabulary (before translation)
     DICO_WORD_TYPES = dictionary of words into their core/attribute category
     DICO_WORD_TO_CLASS = dictionary of words into their end class (for sale, registered,...)
     """
    fp_trans = FP_TAXONOMY_PATH
    df_trans = pd.read_csv(fp_trans, encoding="utf-8")

    for colo in ["root", "word", "trs_direct", "lgg"]:
        df_trans[colo] = df_trans[colo].apply(str.lower)

    df_trans["lgg"] = df_trans["lgg"].apply(lambda x: replace_dico(x, DICO_GGLE_TO_LDETECT))

    VOCABULARY = sorted(list(df_trans[FEATURE_COL].drop_duplicates()))
    DICO_WORD_TYPES = dict(zip(df_trans["trs_direct"], df_trans["word_type"]))
    DICO_WORD_TYPES.update(dict(zip(df_trans["root"], df_trans["word_type"])))
    DICO_WORD_TO_CLASS = dict(zip(df_trans["trs_direct"], df_trans["end_class"]))
    DICO_WORD_TO_CLASS.update(dict(zip(df_trans["root"], df_trans["end_class"])))

    #
    DICO_TRANSLATION_BY_LGG = {"other": {}}
    all_lggs = list(df_trans["lgg"].drop_duplicates())
    for lgg in all_lggs:
        ind_lgg = df_trans["lgg"] == lgg
        dico_lgg = dict(zip(df_trans.loc[ind_lgg, "trs_direct"], df_trans.loc[ind_lgg, FEATURE_COL]))
        DICO_TRANSLATION_BY_LGG[lgg] = dico_lgg
        DICO_TRANSLATION_BY_LGG["other"].update(dico_lgg)

    # DICO_TRANSLATION_BY_LGG = dict(zip(df_trans["trs_direct"], df_trans[CONFIG_TR["feature"]]))

    # spe words
    spe_words = pd.read_csv(FP_UNIQUE_WORDS)
    for i, row in spe_words.iterrows():
        wd = row[FEATURE_COL]
        VOCABULARY.append(wd)
        DICO_WORD_TYPES[wd] = row["word_type"]
        DICO_WORD_TO_CLASS[wd] = row["end_class"]

        for k in DICO_TRANSLATION_BY_LGG.keys():
            DICO_TRANSLATION_BY_LGG[k].update({wd: wd})

    print("\tVocabulary length:{}".format(len(VOCABULARY)))
    return DICO_TRANSLATION_BY_LGG, VOCABULARY, DICO_WORD_TYPES, DICO_WORD_TO_CLASS


DICO_TRANSLATION_BY_LGG, VOCABULARY, DICO_WORD_TYPES, DICO_WORD_TO_CLASS = get_reference_words()
OTHER_FEATURES = ["n_sentences", "n_words", "n_pair_same", "n_pair_successive", "min_sent_size_same",
                  "min_sent_size_successive"]


def complete_init2k_labels(x):
    """Replace legacy 0,1 labels with labtool sampling standard"""
    if isinstance(x, float) and (x == 1):
        return ["parked_notice_individual_content"]
    elif isinstance(x, float) and (x == 0):
        return ["high_content"]
    else:
        raise ValueError(f"Unexepected target value : {x}")


def get_most_common_label(x):
    """Return 1st highest count of list x"""
    labels = [(e, x.count(e)) for e in set(x)]
    labels = sorted(labels, key=lambda x: -x[1])
    return labels[0][0]


def get_list_prevailing_labels(x):
    """Get labels by descending order of votes"""
    labels = [(e, x.count(e)) for e in set(x)]
    labels = sorted(labels, key=lambda x: -x[1])

    max_cnt = labels[0][1]
    labels = [e[0] for e in labels if (e[1] == max_cnt)]
    return labels


def extract_target(x):
    """Extract most common label"""

    label = get_most_common_label(x)

    if TRAIN_AT_LV2:
        return DICO_LABELS_TO_LV2[label]
    else:
        return DICO_LABELS_TO_LV3[label]


def is_err_tok_words(x):
    """type of error: tokenization"""
    if not isinstance(x, float) and not isinstance(x, int):
        return False
    else:
        return x <= 1


def get_scenarios(params):
    """list out hyperparameters scenario to attempt"""
    list_scenario = []
    changing_params = []

    # get dimensions
    dict_n = dict()
    order = list(params.keys())
    for elem in order:
        val = params[elem]
        if not isinstance(val, list):
            params[elem] = [val]
            dict_n[elem] = [0]
        else:
            if len(val) > 1:
                changing_params.append(elem)
            dict_n[elem] = list(range(0, len(val)))

    import itertools
    list_scen = list(itertools.product(*[dict_n[val] for val in order]))

    for scenario in list_scen:
        list_scenario.append(dict(zip(order, [params[elem][scenario[order.index(elem)]] for elem in order])))

    return list_scenario


def get_xgbc_params(config):
    """Get hyperparameters of the ML model"""
    if TRAIN_AT_LV2:
        if config is None:
            config = CONFIG_XGB
        PARAM_INIT_XGBC = {
            "max_depth": config["max_depth"],
            "learning_rate": config["lr"],
            "n_estimators": config["n_estimators"],
            "objective": "binary:logistic",
            "booster": 'gbtree',
            # "n_jobs": 6,
            "n_jobs": -1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": config["min_child_weight"],
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": config["colsample_bytree"],
            "colsample_bylevel": config["colsample_bylevel"],
            "reg_alpha": 0,
            "reg_lambda": config["reg_lambda"],
            "scale_pos_weight": config["scale_pos_weight"],
            "base_score": 0.5,
            "random_state": 0,
            "seed": None,
            "missing": None}

    else:
        if config is None:
            config = CONFIG_XGB_MCLASS

        PARAM_INIT_XGBC = {
            "max_depth": config["max_depth"],
            "learning_rate": config["lr"],
            "n_estimators": config["n_estimators"],
            "objective": "multi:softmax",
            "num_class": 6,
            "booster": 'gbtree',
            # "n_jobs": 6,
            "n_jobs": -1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": config["min_child_weight"],
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": config["colsample_bytree"],
            "colsample_bylevel": config["colsample_bylevel"],
            "reg_alpha": 0,
            "reg_lambda": config["reg_lambda"],
            # "scale_pos_weight": config["scale_pos_weight"],
            "base_score": 2.5,
            "random_state": 0,
            "seed": None,
            "missing": None}

    return PARAM_INIT_XGBC


def add_suggested_error(df_pred_error):
    """Error analysis with suggestion of directions to explore"""
    df_pred_error = df_pred_error.copy(deep=True)

    # suggested error
    df_pred_error["suggested_error"] = ""
    df_pred_error["err_is_tok_word"] = ""
    df_pred_error["err_is_tok_or_trans"] = ""
    df_pred_error["err_is_ml"] = ""
    df_pred_error["err_is_lbl_fn"] = ""
    df_pred_error["err_is_lbl_fp"] = ""
    df_pred_error["n_words_attrib"] = ""
    dico_end_class_to_vocabulary = {}
    # attributes vocabulary
    vocab_attr = VOCABULARY.copy()
    for wd in VOCABULARY:
        if DICO_WORD_TYPES[wd] == "core":
            vocab_attr.remove(wd)
    # end class by attributes vocabulary
    for wd in vocab_attr:
        ec = DICO_WORD_TO_CLASS[wd]
        if ec in dico_end_class_to_vocabulary:
            dico_end_class_to_vocabulary[ec].append(wd)
        else:
            dico_end_class_to_vocabulary[ec] = [wd]
    for i, row in df_pred_error.iterrows():

        # features
        usr_label = row["target_label"]
        pred_ml = row["pred_ml_park"] == "true"
        tgt = row["target"]
        n_words = row["n_words"]
        err_is_tok_word = row["err_is_tok_word"]

        end_class = ""
        if usr_label in DICO_LABELS_TO_ENDCLASS:
            end_class = DICO_LABELS_TO_ENDCLASS[usr_label]
        if end_class == "":
            cnt_word_class = 0
        else:
            lst_words_of_end_class = dico_end_class_to_vocabulary[end_class]
            cnt_word_class = row[lst_words_of_end_class].sum()
        n_words_attrib = row[vocab_attr].sum()

        # indicators
        err_is_tok_word = is_err_tok_words(n_words)
        err_is_tok_or_trans = (not err_is_tok_word) and (tgt == 1) and (n_words_attrib == 0)
        err_is_ml = (tgt == 1) and (n_words_attrib > 0) and (pred_ml == False)
        err_is_lbl_fn = (tgt == 0) and (n_words_attrib > 0) and (pred_ml == True)
        err_is_lbl_fp = (tgt == 1) and (n_words_attrib == 0) and (pred_ml == False) and (n_words > 100)

        suggested_error = []
        if err_is_tok_word:
            suggested_error.append("TOK")
        if err_is_tok_or_trans:
            suggested_error.append("TOK_or_TRANS_or_REF")
        if err_is_ml:
            suggested_error.append("ML")
        if err_is_lbl_fn:
            suggested_error.append("LblFN")
        if err_is_lbl_fp:
            suggested_error.append("LblFP")

        # fill
        values = [["suggested_error", suggested_error],
                  ["n_words_attrib", n_words_attrib],
                  ["err_is_tok_or_trans", err_is_tok_or_trans],
                  ["err_is_ml", err_is_ml],
                  ["err_is_lbl_fn", err_is_lbl_fn],
                  ["err_is_lbl_fp", err_is_lbl_fp]
                  ]
        for colo, value in values:
            if colo == "suggested_error":
                df_pred_error.loc[i, colo] = " - ".join(value)
            else:
                df_pred_error.loc[i, colo] = value

    return df_pred_error


class Trainer():
    """Class handling all the steps of a ML lifecycle"""

    def __init__(self, raw):
        """Initialize trainer with a dataset"""
        if isinstance(raw, pd.DataFrame):
            self.df_raw = raw
        elif isinstance(raw, list):
            self.df_raw = pd.DataFrame(raw)
        else:
            raise ValueError("Unexpected raw data format: list or Dataframe expected")

        self.df_train = None  # must minimally include "text" and "target"
        self.algo = None
        self.predictions = None
        self.output_data = {}

    def preprocess(self):
        """Feature engineerning"""
        df = copy.deepcopy(self.df_raw)
        df = df.rename(columns={"html_text": "text"})

        if "target" in df.columns:
            if "user_choices" not in df.columns:  # original samples only
                df["user_choices"] = np.nan
            ind_no_user_choices = df["user_choices"].isnull()
            df.loc[ind_no_user_choices, "user_choices"] = df.loc[ind_no_user_choices, "target"].apply(
                complete_init2k_labels)
        df["target"] = df["user_choices"].apply(extract_target)

        if TRAIN_AT_LV2:
            FT = Featurer(CONFIG_FENG)
        else:
            FT = Featurer(CONFIG_FENG_MULTI)

        df_tr = []
        y_train = []
        for i, row in df.iterrows():
            all_ml_features, _ = FT.transform(row["text"], row["url"], row["language"])
            y_train.append(row["target"])

            df_tr.append(all_ml_features)

        df_tr = pd.DataFrame(np.concatenate(df_tr), columns=FT.FEATURES)
        y_train = pd.Series(y_train, name="target")

        self.x_train = df_tr
        self.y_train = y_train

    def split(self):
        """Split dataset for CV evaluation"""
        skf = StratifiedKFold(n_splits=CROSS_VAL_N_SPLIT)

        for id_tr, id_te in skf.split(self.x_train, self.y_train):
            print("TRAIN:", len(id_tr), "TEST:", len(id_te))
            # X_train, X_test = df.loc[train_index], df.loc[test_index]
            yield id_tr, id_te

    def init_model(self, config=None):
        """Get the initial ML model"""
        PARAM_INIT_XGBC = get_xgbc_params(config)

        # ML model
        mdl = XGBClassifier(**PARAM_INIT_XGBC)

        # balancing
        if TRAIN_AT_LV2:
            n_pos = self.y_train.sum()
            scale_pos_weight = (len(self.x_train) - n_pos) / n_pos
            mdl.set_params(**{"scale_pos_weight": scale_pos_weight})

        self.algo = mdl

    def train(self, id_df=None):
        """Train one time on the indexed rows id_df, full dataset if None"""
        if id_df is None:
            x = self.x_train
            y = self.y_train
        else:
            x = self.x_train.loc[id_df]
            y = self.y_train.loc[id_df]

        # x_tr_num = np.concatenate([self.vectorizer.fit_transform(df["text"].astype(np.str)).toarray(),
        #                            df[OTHER_FEATURES].values], axis=1)
        x_csr = csr_matrix(x)
        # y = list(df["target"])
        self.algo.fit(x_csr, y)

    def predict(self, id_df=None):
        """Predict one time on the indexed rows id_df, full dataset if None"""
        # info data
        if id_df is None:
            df_pred = self.df_raw.copy(deep=True)
        else:
            df_pred = self.df_raw.loc[id_df].copy(deep=True)

        # features
        if id_df is None:
            x = self.x_train
        else:
            x = self.x_train.loc[id_df]

        # predict
        x_csr = csr_matrix(x)
        if TRAIN_AT_LV2:
            preds = list(self.algo.predict_proba(x_csr)[:, 1])
            df_pred["predicted_proba"] = preds
            df_pred["predicted"] = [int(e >= THRESHOLD_CLASSIFICATION) for e in preds]
        else:
            preds = list(self.algo.predict(x_csr))
            df_pred["predicted_proba"] = preds
            df_pred["predicted"] = preds

        # add features
        df_feats = pd.DataFrame(x, columns=VOCABULARY + OTHER_FEATURES)
        for colo in df_feats.columns:
            df_pred[colo] = df_feats[colo]

        return df_pred

    def assess_perfs(self, df_pred, prefix="", print_only=False):
        """Evaluate prediction performance of the ML model"""
        if TRAIN_AT_LV2:
            assert "target" in df_pred.columns
            assert "predicted" in df_pred.columns
            assert "predicted_proba" in df_pred.columns

            y_test = df_pred["target"]
            y_pred_te = df_pred["predicted_proba"]

            feats = VOCABULARY + OTHER_FEATURES
            # metrics
            thresh = THRESHOLD_CLASSIFICATION
            metrics = performance_binary_classification(y_test, y_pred_te, thresh, name=prefix[0:-1])

        else:
            assert "target" in df_pred.columns
            assert "predicted" in df_pred.columns
            y_test = df_pred["target"]
            y_pred_te = df_pred["predicted"]
            metrics = performance_multiclass_classification(y_test, y_pred_te, name=prefix[0:-1])

        # print("\n".join([f"{k}:{v}" for k, v in metrics.items()]))

        if not print_only:
            print("\n".join([f"{k}:{v}" for k, v in metrics.items()]))

            self.output_data.update({f"{prefix}performance": metrics})

        return metrics

    def assess_tld_lgg_perfs(self, df_pred, prefix="", print_only=False):
        """Assess performance by TLD and by Language"""
        assert "target" in df_pred.columns
        assert "predicted_proba" in df_pred.columns

        lggs = df_pred["language"].value_counts(ascending=False, dropna=True).reset_index(drop=False)

        # languages
        df_lgg = []
        for i, row in lggs.iterrows():
            cnt = row["language"]
            if cnt > 1:
                lgg = row["index"]

                df_pred_cur = df_pred[df_pred["language"] == lgg]

                y_test = df_pred_cur["target"]
                y_pred_te = df_pred_cur["predicted_proba"]

                # metrics
                metrics = performance_binary_classification(y_test, y_pred_te, THRESHOLD_CLASSIFICATION,
                                                            name=prefix[0:-1], with_confusion=True)

                dico_resu = OrderedDict()
                dico_resu["language"] = lgg
                dico_resu["count"] = cnt
                dico_resu.update(metrics)

                df_lgg.append(dico_resu)

        self.output_data["performance_by_lgg"] = pd.DataFrame(df_lgg)

        # tlds
        tlds = df_pred["tld"].value_counts(ascending=False, dropna=True).reset_index(drop=False)
        df_tld = []
        for i, row in tlds.iterrows():
            cnt = row["tld"]
            if cnt > 1:
                tld = row["index"]

                df_pred_cur = df_pred[df_pred["tld"] == tld]

                y_test = df_pred_cur["target"]
                y_pred_te = df_pred_cur["predicted_proba"]

                # metrics
                metrics = performance_binary_classification(y_test, y_pred_te, THRESHOLD_CLASSIFICATION,
                                                            name=prefix[0:-1], with_confusion=False)

                dico_resu = OrderedDict()
                dico_resu["tld"] = tld
                dico_resu["count"] = cnt
                dico_resu.update(metrics)

                df_tld.append(dico_resu)

        self.output_data["performance_by_tld"] = pd.DataFrame(df_tld)

    def error_analysis(self, df_pred):
        """Perform error analysis on the CV predictions"""
        assert "target" in df_pred.columns
        assert "predicted" in df_pred.columns

        ind_error = df_pred["target"] != df_pred["predicted"]

        df_pred_error = df_pred[ind_error].copy(deep=True)

        cols = list(df_pred_error)
        start_cols = ["url", "target", "predicted", "screenshot", "target_label"]
        other_cols = [e for e in cols if e not in start_cols]

        try:
            df_pred_error = df_pred_error[start_cols + other_cols]
        except Exception as e:
            print(f"Error with columns availability in prediction table -> ignored formatting: {type(e)} - {str(e)}")

        # error suggestion
        if TRAIN_AT_LV2:
            df_pred_error = add_suggested_error(df_pred_error)

        self.output_data["prediction_errors"] = df_pred_error
        self.output_data["all_predictions"] = df_pred.copy(deep=True)

    def save_config(self, out_folder):
        """Save trained ML model"""
        out_model = OUT_MODEL_NAME

        # save
        with open(join(out_folder, out_model), "wb+") as f:
            pickle.dump(self.algo, f)

        LG.add({"time": gtm(), "importance": "INFO", "category": "ml model", "description": "trained ml model saved"})
        print("full model saved")

    def optimize_hyperparams(self, out_folder):
        """Optimize hyperparameters: Grid Search on provided configuration parameters"""
        print("STARTING HYPERPARAMETERS OPTIMIZATION")
        # hyperparams
        if TRAIN_AT_LV2:
            list_scenarios = get_scenarios(OPTI_XGB)
        else:
            list_scenarios = get_scenarios(OPTI_XGB_MCLASS)
        # list_scenarios = list_scenarios[0:2]
        tot_scen = len(list_scenarios)

        # loop scenario
        all_perfs = []
        keep_cols = ["target", "predicted", "predicted_proba"]
        for idx_scen, scen in enumerate(list_scenarios):
            print(f"progress: {idx_scen}/{tot_scen} : {scen}")
            all_pred_cv = []
            for id_tr, id_te in self.split():
                self.init_model(scen)

                self.train(id_tr)

                df_pred_cv = self.predict(id_te)
                df_pred_cv["target"] = self.y_train.loc[id_te]

                all_pred_cv.append(df_pred_cv[keep_cols].copy(deep=True))

            df_pred_cv = pd.concat(all_pred_cv, axis=0, sort=False)
            metrics = self.assess_perfs(df_pred_cv, prefix="hyperparam_", print_only=True)
            metrics.update(scen)

            all_perfs.append(metrics)
            a = 1

        pd.DataFrame(all_perfs).to_csv(join(out_folder, OUT_HYPERPARAMERS_OPTI_NAME.format(gtm())), index=False)

        print("END OF HYPERPARAMETERS OPTIMIZATION")

    def full_cv_train(self):
        """Train the model on all folds of the dataset"""
        all_preds_cv = []
        df_pred_tr = None
        for id_tr, id_te in self.split():

            self.init_model()

            self.train(id_tr)

            df_pred_cv = self.predict(id_te)
            df_pred_cv["target"] = self.y_train.loc[id_te]

            if df_pred_tr is None:
                df_pred_tr = self.predict(id_tr)
                df_pred_tr["target"] = self.y_train.loc[id_tr]

            self.assess_perfs(df_pred_cv, print_only=True)

            all_preds_cv.append(df_pred_cv.copy(deep=True))

        df_pred_cv = pd.concat(all_preds_cv, axis=0)

        return df_pred_cv, df_pred_tr
