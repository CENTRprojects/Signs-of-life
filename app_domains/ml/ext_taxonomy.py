"""OLD: script to refine word taxonomy"""
import copy
import time
import string
import sys
import os
from os.path import join
import re
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# import sys
pth = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(pth)
sys.path.append(pth)

from word_forms.word_forms import get_word_forms
# from googletrans import Translator
from config import RUN_CONFIG
from tokenizers import WORD_TOK_ID_TO_TOKENIZER

df_lgg_to_tok = pd.read_csv(join(RUN_CONFIG["MAIN_DIR"], "input", "lgg_tokenizers.csv"))
DICO_LGG_TO_WORD_TOK_ID = dict(zip(list(df_lgg_to_tok["lgg"]), list(df_lgg_to_tok["word_tokenizer"])))

ALL_LGG = ["sq", "am", "ar", "hy", "az", "eu", "be", "bn", "bs", "bg", "ca", "ceb", "zh-cn", "zh-tw", "co",
           "hr", "cs", "da", "nl", "en", "eo", "et", "fi", "fr", "fy", "gl", "ka", "de", "el", "gu", "ht", "ha",
           "haw", "he", "hi", "hmn", "hu", "is", "ig", "id", "ga", "it", "ja", "jw", "kn", "kk", "km", "ko", "ku",
           "ky", "lo", "la", "lv", "lt", "lb", "mk", "mg", "ms", "ml", "mt", "mi", "mr", "mn", "my", "ne", "no",
           "ny", "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sm", "gd", "sr", "st", "sn", "sd", "si", "sk", "sl",
           "so", "es", "su", "sw", "sv", "tl", "tg", "ta", "te", "th", "tr", "uk", "ur", "uz", "vi", "cy", "xh",
           "yi", "yo", "zu"]

ONE_LETTER_FILTER = {"sm", "sd", "mi",  "ig", "si", "", "", "", "", ""}
TWO_LETTER_FILTER = {"vi", "yo", "kk", "mn", "so", "ku", "my", "ku", "mi", "haw", "gd", "sm", "ht", "ro", "ha"}



SET_PUNCTUATION = set(list(string.punctuation) + ["▁"])
pat_punct = string.punctuation.replace("_", "").replace("-", "")
pat_spe = "\n\t\r "
NON_TOK_PAT = "[" + pat_punct + pat_spe + "]+"
RE_NON_TOK_PAT = re.compile(NON_TOK_PAT, re.IGNORECASE)

p_json = join(RUN_CONFIG["MAIN_DIR"], "inter", "ml", "stopwords")
all_json = os.listdir(p_json)
DICO_SW_ID_TO_STOP_WORDS = {}
for js in all_json:
    with open(join(p_json, js), "r", encoding="utf-8") as f:
        try:
            list_wd = json.load(f)
        except:
            print(js)
    DICO_SW_ID_TO_STOP_WORDS[js.split(".")[0]] = list_wd

df_lgg_to_tok = pd.read_csv(join(RUN_CONFIG["MAIN_DIR"], "input", "lgg_tokenizers.csv"))
df_lgg_to_tok = df_lgg_to_tok[df_lgg_to_tok["stop_words"].notnull()]
DICO_LGG_TO_SW_ID = dict(zip(list(df_lgg_to_tok["lgg"]), list(df_lgg_to_tok["stop_words"])))


def root_to_derived():
    p_core  =join(RUN_CONFIG["MAIN_DIR"], "inter", "ml")
    fpin = join(p_core, "vocab_v2.csv")
    fpout = join(p_core, "vocab_derived_v2.csv")
    df_derived = []

    df = pd.read_csv(fpin)
    tot_lg = len(df)
    for i, row in df.iterrows():
        print("{}/{}".format(i, tot_lg))

        root = row["root"]
        to_translate = row["to_translate"]
        to_derive = row["to_derive"]

        if to_derive == "yes":

            derivations = get_word_forms(root)

            for form in derivations.keys():

                new_deriv = list(set(derivations[form]))

                for deriv in new_deriv:
                    df_derived.append({"root": root, "word": deriv, "form": form, "to_derive": to_derive,
                                       "to_translate": to_translate})

        else:
            df_derived.append({"root": root, "word": root, "to_derive": to_derive, "to_translate": to_translate})

    pd.DataFrame(df_derived).to_csv(fpout, index=False)


def unit_translate(row):
    if "word" in row:
        to_translate = row["word"]
    else:
        to_translate = row["context"]

    try:
        translator = Translator()
        trs = translator.translate(to_translate, src="en", dest=row["lgg"]).text

    except Exception as e:
        print("error :{}\t\t{}\t\t{}\t\t{}".format(to_translate, type(e), str(e), row["lgg"]))
        trs = None

    row["trs"] = trs
    return row


def unit_translate_double(row):
    to_translate = row["trs"]

    try:
        translator = Translator()
        trs = translator.translate(to_translate, src="en", dest=row["lgg"]).text

    except Exception as e:
        print("error :{}\t\t{}\t\t{}\t\t{}".format(to_translate, type(e), str(e), row["lgg"]))
        trs = None

    row["trs_dbl_trans"] = trs
    return row


def concat_data(l_direct, l_contex, l_dbl_trans):
    df = pd.DataFrame(l_direct)

    print(l_contex)
    df_ctxt = pd.DataFrame(l_contex)
    df_ctxt = df_ctxt[["index", "trs"]]
    df_ctxt = df_ctxt.rename(columns={"trs": "trs_context"})

    df_dbl_trs = pd.DataFrame(l_dbl_trans)
    df_dbl_trs = df_dbl_trs[["index", "trs_dbl_trans"]]
    # df_dbl_trs = df_dbl_trs.rename(columns={"trs":"trs_dbl_trans"})

    df = pd.merge(df, df_ctxt, on="index", how="outer")
    df = pd.merge(df, df_dbl_trs, on="index", how="outer")

    return df


def build_taxonomy():
    p_core  =join(RUN_CONFIG["MAIN_DIR"], "inter", "ml")
    fpin = join(p_core, "vocab_derived_with_context_v2.csv")
    # fpin_unique_word = r"D:\Documents\freelance\sign_of_life_crawler\inter\ml\vocab_unique_words.csv"
    fpout = join(p_core, "taxonomy_v2.csv")
    fpout_missing = join(p_core, "missing_v2.csv")

    df = pd.read_csv(fpin)

    df = df.head(SAMPLEING)
    df["index"] = df.index

    # add lgg
    for elem in ALL_LGG[0:N_LGG]:
        df[elem] = np.nan

    df = pd.melt(df, id_vars=["form", "root", "index", "to_derive", "to_translate", "word", "context"], var_name="lgg",
                 value_name="translation")

    list_direct = df.drop(["context"], axis=1).copy(deep=True).to_dict(orient="records")
    list_context = df.drop(["word"], axis=1).copy(deep=True).to_dict(orient="records")

    # translation
    print("loop direct")
    count_test = 0
    list_done_direct = []
    while count_test < 10:
        list_direct = Parallel(n_jobs=6)(delayed(unit_translate)(batch) for batch in tqdm(list_direct))

        list_done_direct = list_done_direct + [e for e in list_direct if (e["trs"] is None)]
        list_direct = [e for e in list_direct if (e["trs"] is None)]

        if len(list_direct) == 0:
            break

        count_test += 1
        time.sleep(SLEEP_TIME)

    print("loop context")
    count_test = 0
    list_done_context = []
    while count_test < 10:
        list_context = Parallel(n_jobs=6)(delayed(unit_translate)(batch) for batch in tqdm(list_context))

        list_done_context = list_done_context + [e for e in list_direct if (e["trs"] is None)]
        list_context = [e for e in list_context if (e["trs"] is None)]

        if len(list_context) == 0:
            break

        count_test += 1
        time.sleep(SLEEP_TIME)

    print("loop double trans")
    count_test = 0
    list_done_direct_orig = [copy.deepcopy(e) for e in list_done_direct]
    list_done_dbl_trans = []
    while count_test < 10:
        list_done_direct_orig = Parallel(n_jobs=6)(
            delayed(unit_translate_double)(batch) for batch in tqdm(list_done_direct_orig))

        list_done_dbl_trans = list_done_dbl_trans + [e for e in list_direct if (e["trs_dbl_trans"] is None)]
        list_done_direct_orig = [e for e in list_done_direct_orig if (e["trs_dbl_trans"] is None)]

        if len(list_done_direct_orig) == 0:
            break

        count_test += 1
        time.sleep(SLEEP_TIME)

    # concat
    df_final = concat_data(list_done_direct, list_done_context, list_done_dbl_trans)
    df_final = pd.pivot(df_final, index=["form", "root", "index", "to_derive", "to_translate", "word", "context"],
                        columns=["lgg"], values=["trs", "trs_context", "trs_dbl_trans"])
    df_final.to_csv(fpout, index=False)

    missing = list_direct + list_context + list_done_direct_orig
    pd.DataFrame(missing).to_csv(fpout_missing, index=False)
    print("done")


def concat_translations():
    p_core  =join(RUN_CONFIG["MAIN_DIR"], "inter", "ml")
    fpin = join(p_core, "park_full_tr_v2.csv")
    fpout = join(p_core, "taxonomy_v2.csv")
    fpout_2 = join(p_core, "taxonomy_tocheck_v2.csv")
    colo_core = ["form", "root", "to_derive", "to_translate", "word", "context"]

    df = pd.read_csv(fpin, sep="\t")

    df = pd.melt(df, id_vars=colo_core, var_name="lgg", value_name="trs_direct")

    df = df.sort_values(by=["lgg", "word"], ascending=True)

    df.to_csv(fpout, index=False)

    # df[ind_not_ok].to_csv(fpout_2, index=False)
    pass


def concat_taxonomies_add_features():
    p_core  =join(RUN_CONFIG["MAIN_DIR"], "inter", "ml")
    fpin1 = join(p_core, "taxonomy_v1.csv")
    fpin2 = join(p_core, "taxonomy_v2.csv")

    fpin_feat = join(p_core, "root_categories.csv")

    fpout = join(p_core, "taxonomy.csv")

    df1 = pd.read_csv(fpin1)
    df2 = pd.read_csv(fpin2)

    df = pd.concat([df1, df2], axis=0, sort=False)

    # categories
    df_categ = pd.read_csv(fpin_feat)
    df = pd.merge(df, df_categ, on="root", how="left")

    assert df["word_type"].isnull().astype(int).sum() == 0

    df.to_csv(fpout, index=False)
    print("done")


def remove_stop_words_text(txt, lgg):
    if lgg in DICO_LGG_TO_SW_ID:
        sw_id = DICO_LGG_TO_SW_ID[lgg]
        if sw_id in DICO_SW_ID_TO_STOP_WORDS:
            list_wd = DICO_SW_ID_TO_STOP_WORDS[sw_id]

            while True:
                is_changed = False

                lg_bef = len(txt)
                for wd in list_wd:

                    try:
                        init_text = txt
                        txt = re.sub("^" + wd + " | " + wd + "$", "", txt, flags=re.IGNORECASE)

                        lg_after = len(txt)
                        if lg_after != lg_bef:
                            is_changed = True
                            lg_bef = lg_after
                            print("info removed: \t{}\t from \t{}\t to \t{}".format(wd, init_text, txt))
                    except:
                        print("error with: \t{}".format(wd))
                        continue

                if not is_changed:
                    break

    return txt


def remove_stop_words(txt, lgg):
    # Word tokenization
    if lgg in DICO_LGG_TO_WORD_TOK_ID:
        word_tokenizer = WORD_TOK_ID_TO_TOKENIZER[DICO_LGG_TO_WORD_TOK_ID[lgg]]
    else:
        word_tokenizer = WORD_TOK_ID_TO_TOKENIZER["std"]

    txt_list = word_tokenizer(txt)
    # txt = RE_NON_TOK_PAT.split(txt)

    # clean-up tokenized
    txt_list = [e.lower() for e in txt_list if (len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION)]

    if lgg in DICO_LGG_TO_SW_ID:
        sw_id = DICO_LGG_TO_SW_ID[lgg]
        if sw_id in DICO_SW_ID_TO_STOP_WORDS:
            list_wd = DICO_SW_ID_TO_STOP_WORDS[sw_id]

            while True:
                is_changed = False

                for wd in list_wd:

                    if len(txt_list) == 0:
                        break

                    if wd == txt_list[0]:
                        init_text = txt
                        txt = txt[len(wd)::]
                        if len(txt) > 0 and (txt[0] in [" ", "'"]):
                            txt = txt[1::]
                        is_changed = True
                        print("info removed: \t{}\t from \t{}\t to \t{}".format(wd, init_text, txt))
                        txt_list.pop(0)

                    if len(txt_list) == 0:
                        break

                    if wd == txt_list[-1]:
                        init_text = txt
                        txt = txt[0:-len(wd)]
                        if len(txt) > 0 and (txt[-1] in [" ", "'"]):
                            txt = txt[0:-1]
                        is_changed = True
                        print("info removed: \t{}\t from \t{}\t to \t{}".format(wd, init_text, txt))
                        txt_list.pop()

                    if len(txt_list) == 0:
                        break

                if not is_changed:
                    break

    return txt


def refine_taxonomy():
    p_core  =join(RUN_CONFIG["MAIN_DIR"], "inter", "ml")
    fp_in = join(p_core, "taxonomy.csv")
    fp_out = join(p_core, "taxonomy_refined.csv")

    df = pd.read_csv(fp_in, encoding="utf-8")

    # df = df[["root", "word", "lgg", "trs_direct"]]

    # df["count_words"] = df["trs_direct"].apply(lambda x: len(RE_NON_TOK_PAT.split(x)))
    # ind_multi = df["count_words"] > 1

    df["trs_direct"] = df[["trs_direct", "lgg"]].apply(lambda x: remove_stop_words(x[0], x[1]), axis=1)

    lg_bef = len(df)
    ind_null_text = df["trs_direct"].apply(lambda x: len(x) == 0)
    df = df[~ind_null_text].reset_index(drop=True)
    print("N rows removed:{}".format(len(df) - lg_bef))

    a = 1

    # df = df[ind_multi].reset_index(drop=True)

    # df = df.sort_values(by="count_words", ascending=False)

    df.to_csv(fp_out, index=False)
    print(df.shape)


def is_filter(lgg, txt):
    lg_txt = len(txt)
    if lgg in TWO_LETTER_FILTER:
        return lg_txt<=2
    elif lgg in ONE_LETTER_FILTER:
        return lg_txt<=1
    else:
        return False


def taxo_clear():
    p_core  =join(RUN_CONFIG["MAIN_DIR"], "inter", "ml")
    fp_in = join(p_core, "taxonomy.csv")
    fp_out = join(p_core, "taxonomy_refined_cleared.csv")

    df = pd.read_csv(fp_in)

    ind_to_remove = df[["lgg", "trs_direct"]].apply(lambda x: is_filter(x[0], x[1]), axis=1)
    print(np.sum(ind_to_remove))
    df = df[~ind_to_remove]


    df.to_csv(fp_out, index=False)

    a = 1


if __name__ == '__main__':
    # root_to_derived()

    SLEEP_TIME = 1
    SAMPLEING = None
    N_LGG = 3
    # build_taxonomy()

    # concat_translations()

    # concat_taxonomies_add_features()

    # refine_taxonomy()

    taxo_clear()
