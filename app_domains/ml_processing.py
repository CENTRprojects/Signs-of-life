"""OLD pipeline for feature engineering: replaced by feature_eng.py"""
from os.path import join
import pandas as pd
import pickle
import re
import string

from config import RUN_CONFIG
from tokenizers import *

SET_PUNCTUATION = set(list(string.punctuation) + ["▁"])
pat_punct = string.punctuation.replace("_", "").replace("-", "")
pat_spe = "\n\t\r "
NON_TOK_PAT = "[" + pat_punct + pat_spe + "]+"
RE_NON_TOK_PAT = re.compile(NON_TOK_PAT)
TOK_SENT_SEP = "[.?:!;·˳̥۰ᴉ•․﮳。：；？！」』؟:]"
RE_TOK_SENT_SEP = re.compile(TOK_SENT_SEP + "+", flags=re.IGNORECASE)
RE_TOK_MULTI_SENT_SEP = re.compile(TOK_SENT_SEP + "{2,}", flags=re.IGNORECASE)

# tokenizers to use
df_lgg_to_tok = pd.read_csv(join(RUN_CONFIG["MAIN_DIR"], "input", "lgg_tokenizers.csv"))
DICO_LGG_TO_WORD_TOK_ID = dict(zip(list(df_lgg_to_tok["lgg"]), list(df_lgg_to_tok["word_tokenizer"])))
DICO_LGG_TO_SENT_TOK_ID = dict(zip(list(df_lgg_to_tok["lgg"]), list(df_lgg_to_tok["sentence_tokenizer"])))

# ml
with open(join(RUN_CONFIG["MAIN_DIR"], "input", RUN_CONFIG["name_model"] + ".pkl"), "rb+") as f:
    MDL = pickle.load(f)
MDL.set_params(**{"n_jobs":1})
with open(join(RUN_CONFIG["MAIN_DIR"], "input", "vecto_" + RUN_CONFIG["name_model"] + ".pkl"),
          "rb+") as f:
    VECTORIZER = pickle.load(f)


def get_reference_words():
    """"Parse taxonomy file to get the parking vocabulary
    DICO_TRANSLATION_BY_LGG = dict translations of by Language,
    VOCABULARY = the base vocabulary (before translation)
     DICO_WORD_TYPES = dictionary of words into their core/attribute category
     DICO_WORD_TO_CLASS = dictionary of words into their end class (for sale, registered,...)
     """
    fp_trans = RUN_CONFIG["taxonomy_path"]
    df_trans = pd.read_csv(fp_trans, encoding="utf-8")
    # df_trans = pd.read_csv(fp_trans)

    for colo in ["root", "word", "trs_direct"]:
        df_trans[colo] = df_trans[colo].apply(str.lower)

    # DICO_TRANSLATION = dict(zip(df_trans_list["local_word"], df_trans_list[CONFIG_TR["feature"]]))
    # DICO_TRANSLATION = dict(zip(df_trans["trs_direct"], df_trans[RUN_CONFIG["feature"]]))
    VOCABULARY = sorted(list(df_trans[RUN_CONFIG["feature"]].drop_duplicates()))
    DICO_WORD_TYPES = dict(zip(df_trans["trs_direct"], df_trans["word_type"]))
    DICO_WORD_TYPES.update(dict(zip(df_trans["root"], df_trans["word_type"])))
    DICO_WORD_TO_CLASS = dict(zip(df_trans["trs_direct"], df_trans["end_class"]))
    DICO_WORD_TO_CLASS.update(dict(zip(df_trans["root"], df_trans["end_class"])))

    DICO_TRANSLATION_BY_LGG = {"other": {}}
    all_lggs = list(df_trans["lgg"].drop_duplicates())
    for lgg in all_lggs:
        ind_lgg = df_trans["lgg"] == lgg
        dico_lgg = dict(zip(df_trans.loc[ind_lgg, "trs_direct"], df_trans.loc[ind_lgg, RUN_CONFIG["feature"]]))
        DICO_TRANSLATION_BY_LGG[lgg] = dico_lgg
        DICO_TRANSLATION_BY_LGG["other"].update(dico_lgg)

    # spe words
    spe_words = pd.read_csv(RUN_CONFIG["unique_words"])
    for i, row in spe_words.iterrows():
        wd = row[RUN_CONFIG["feature"]]
        VOCABULARY.append(wd)
        DICO_WORD_TYPES[wd] = row["word_type"]
        DICO_WORD_TO_CLASS[wd] = row["end_class"]

        for k in DICO_TRANSLATION_BY_LGG.keys():
            DICO_TRANSLATION_BY_LGG[k].update({wd: wd})

    # print("\tVocabulary length:{}".format(len(VOCABULARY)))
    return DICO_TRANSLATION_BY_LGG, VOCABULARY, DICO_WORD_TYPES, DICO_WORD_TO_CLASS


DICO_TRANSLATION_BY_LGG, VOCABULARY, DICO_WORD_TYPES, DICO_WORD_TO_CLASS = get_reference_words()
OTHER_FEATURES = ["n_sentences", "n_words", "n_pair_same", "n_pair_successive", "min_sent_size_same",
                  "min_sent_size_successive"]
FEATURES = VOCABULARY + OTHER_FEATURES   # all features used for the ML training/prediction

SPE_TOKENS = [
    ["index of", "index_of"],
    ["hello world", "hello_world"],
    ["lorem ipsum", "lorem_ipsum"],
    ["no website", "no_website"]
]

MAX_SENT = 100

def split_big_sentences(txt):
    """Limit size of sentences to MAX_SENT"""
    txt_cor = []
    for sent in txt:
        if len(sent) > MAX_SENT:
            for i in range(0, (len(sent) // MAX_SENT) + 1):
                txt_spl = sent[i * MAX_SENT: (i + 1) * MAX_SENT]
                if len(txt_spl) > 0:
                    txt_cor.append(txt_spl)
        else:
            txt_cor.append(sent)
    return txt_cor

def translate(w, lgg):
    """dictionnary translation"""
    lgg_to_use = lgg
    if lgg not in DICO_TRANSLATION_BY_LGG:
        lgg_to_use = "other"

    if w in DICO_TRANSLATION_BY_LGG[lgg_to_use]:
        return DICO_TRANSLATION_BY_LGG[lgg_to_use][w]
    else:
        return w

def get_relevant_words(text, lgg):
    """Collect identified parking words"""
    relev = []
    lgg_to_use = lgg
    if lgg not in DICO_TRANSLATION_BY_LGG:
        lgg_to_use = "other"

    for sentence in text:
        for word in sentence:
            if word in DICO_TRANSLATION_BY_LGG[lgg_to_use]:
                relev.append(DICO_TRANSLATION_BY_LGG[lgg_to_use][word])
    return relev


def add_url_token(txt, url):
    """Replace the domain name into the token 'x_url'"""
    txt = re.sub(("www." + url).replace(".", "\."), "x_url", txt, flags=re.IGNORECASE)
    txt = re.sub(url.replace(".", "\."), "x_url", txt, flags=re.IGNORECASE)
    return txt


def add_multi_word_tokens(txt):
    """Replace multiple words parking vocabulary into a one word token"""
    for pair in SPE_TOKENS:
        left = pair[0].split(" ")[0]
        right = pair[0].split(" ")[1]
        txt = re.sub(left + "[ .]+" + right, pair[1], txt, flags=re.IGNORECASE)
    return txt


def spit_sentences(txt):
    """Cut text into sentences"""
    txt = RE_TOK_SENT_SEP.split(txt)
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


def tokenize_and_correct(txt, lgg):
    """Split by words using alphanumeric/dash/underscore and check if in mispelled words"""

    # Sentence tokenization
    # txt = spit_sentences(txt)
    if lgg in DICO_LGG_TO_SENT_TOK_ID:
        sent_tokenizer = eval(SENT_TOK_ID_TO_TOKENIZER[DICO_LGG_TO_SENT_TOK_ID[lgg]])
    else:
        sent_tokenizer = eval(SENT_TOK_ID_TO_TOKENIZER["std"])

    # Word tokenization
    if lgg in DICO_LGG_TO_WORD_TOK_ID:
        word_tokenizer = eval(WORD_TOK_ID_TO_TOKENIZER[DICO_LGG_TO_WORD_TOK_ID[lgg]])
    else:
        word_tokenizer = eval(WORD_TOK_ID_TO_TOKENIZER["std"])

    # utf-8 encoding
    txt = bytes(txt, 'utf-8').decode('utf-8', 'ignore')

    try:
        txt = sent_tokenizer(txt)
    except:
        print("INFO: failed target language tokenization for one domain --> standard used")
        txt = sent_latin_tok(txt)

    # clean-up sentences:
    txt = [e for e in txt if ((len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION))]

    # split abnormally long sentences
    txt = split_big_sentences(txt)

    # remove unusual separator texts
    txt = [RE_TOK_MULTI_SENT_SEP.sub(".", sent) for sent in txt]

    # clean-up sentences
    txt = [e for e in txt if ((len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION))]

    word_tok = []
    for sent in txt:
        try:
            sent_tok = word_tokenizer(sent)
        except:
            sent_tok = word_latin_tok(sent)
        word_tok.append(sent_tok)
    txt = word_tok

    # clean-up tokens
    txt = [[e.lower() for e in sent if (len(e.strip()) > 0) and (e.strip() not in SET_PUNCTUATION)] for sent in txt]

    # partial translation
    txt = [[translate(e, lgg) for e in sent] for sent in txt]
    return txt

if __name__ == '__main__':
    pass
    # txt_test = """html.erstehilfekurs.at.Buy this domain..erstehilfekurs.at.2020 Copyright.  All Rights Reserved..The Sponsored Listings displayed above are served automatically by a third party. Neither the service provider nor the domain owner maintain any relationship with the advertisers. In case of trademark issues please contact the domain owner directly (contact information can be found in whois)..Privacy Policy.."""
    txt_test = """De door u gewenste pagina bevindt zich.hier...."""
    print(tokenize_and_correct(txt_test, "nl"))
