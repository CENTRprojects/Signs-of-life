"""Script that analyzes prediction errors following the ML model retraining"""
import json
from os.path import join
import numpy as np
import pandas as pd

from ml.config_train import CONFIG_TR
from ml.ml_classif import VOCABULARY


def distance_feats(feat1, feat2):
    # count words
    d_count = np.sum(np.abs(feat1[0:66] - feat2[0:66]))
    d_words = np.abs(feat1[67] - feat2[67]) / 100
    d_sentences = np.abs(feat1[66] - feat2[66]) / 5
    d_pairs = np.sum(np.abs(feat1[68:70] - feat2[68:70]))
    return d_count + d_pairs + d_words + d_sentences


def feat_to_dico(feats, suf):
    n_words = feats[67]
    n_sent = feats[66]
    n_pair_same = feats[68]
    n_pair_succ = feats[69]

    spe_words = ""
    words_feats = feats[0:66]
    non_null_indexes = np.where(words_feats > 0)
    for idx in list(non_null_indexes[0]):
        word = VOCABULARY[idx]
        word_cnt = words_feats[idx]
        spe_words += word + ":" + str(word_cnt) + "_"

    dico = {"n_words" + suf: n_words,
            "n_sent" + suf: n_sent,
            "n_pair_same" + suf: n_pair_same,
            "n_pair_succ" + suf: n_pair_succ,
            "spe_words" + suf: spe_words}
    return dico


def error_analysis(preds, feats):
    fp_out = join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "errors", "error_analysis.csv")
    p_out_text = join(CONFIG_TR["MAIN_DIR"], "inter", "ml", "errors", "text")

    # save texts
    with open(CONFIG_TR["fp_dataset"], "r", encoding='utf-8') as f:
        txts = json.loads(f.read().encode('raw_unicode_escape').decode())

    for txt in txts:
        with open(join(p_out_text, txt["url"] + ".txt"), "w") as f:
            f.write(str(txt["clean_text"]))

    # filter errors
    preds_errors = preds[preds["actual"] != preds["pred_categ"]].copy(deep=True)
    url_actu_1 = set(preds.loc[preds["actual"] == 1, "url"])
    url_actu_0 = set(preds.loc[preds["actual"] == 0, "url"])
    url_fn = set(preds_errors.loc[preds_errors["actual"] == 1, "url"])
    url_fp = set(preds_errors.loc[preds_errors["actual"] == 0, "url"])
    dico_url_to_actu = dict(zip(list(preds["url"]), list(preds["actual"])))
    dico_url_to_lgg = dict(zip(list(preds["url"]), list(preds["language"])))

    # feats
    dico_all_feats = {}
    for urls, fts in feats:
        for i in range(len(urls)):
            dico_all_feats[urls[i]] = fts[i]

    #
    closest_elems = []
    for url in url_fn:
        feat_url = dico_all_feats[url]
        min_dist = 1000000
        min_url = None
        for url_neg in url_actu_0:
            dst = distance_feats(feat_url, dico_all_feats[url_neg])

            if dst < min_dist:
                min_url = url_neg
                min_dist = dst

        closest_elems.append((url, min_url, min_dist))
    for url in url_fp:
        feat_url = dico_all_feats[url]
        min_dist = 1000000
        min_url = None
        for url_pos in url_actu_1:
            dst = distance_feats(feat_url, dico_all_feats[url_pos])

            if dst < min_dist:
                min_url = url_pos
                min_dist = dst

        closest_elems.append((url, min_url, min_dist))

    # enriching pairs
    all_resu = []
    for pair in closest_elems:
        resu = {"url": pair[0], "url_close": pair[1], "dist": pair[2]}

        resu["actual"] = dico_url_to_actu[pair[0]]
        resu["lgg"] = dico_url_to_lgg[pair[0]]
        resu["actual_close"] = dico_url_to_actu[pair[1]]
        resu["lgg_close"] = dico_url_to_lgg[pair[1]]

        dico_feats = feat_to_dico(dico_all_feats[pair[0]], suf="")
        dico_feats_close = feat_to_dico(dico_all_feats[pair[1]], suf="_close")

        resu.update(dico_feats)
        resu.update(dico_feats_close)

        all_resu.append(resu)

    df = pd.DataFrame(all_resu)
    df = df.sort_values(by="dist", ascending=True)
    colo_order = ['url', "actual", 'url_close', "actual_close", 'dist', "lgg", 'n_words', 'n_sent', 'n_pair_same',
                  'n_pair_succ', 'spe_words', "lgg_close", 'n_words_close', 'n_sent_close', 'n_pair_same_close',
                  'n_pair_succ_close', 'spe_words_close']
    df = df[colo_order]

    df.to_csv(fp_out)
    b = 1

    pass
