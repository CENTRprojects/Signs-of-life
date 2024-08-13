"""Multi-languages text tokenizers (text to words)"""
import string
import re

import jieba
from inltk.tokenizer import IndicTokenizer

# latin languages + arab
from cltk.tokenize.word import WordTokenizer
from cltk.tokenize.sentence import TokenizeSentence

# thai
from pythainlp import sent_tokenize as thai_sent_tokenize
from pythainlp import word_tokenize as thai_word_tokenize

# vietnamese
from underthesea import sent_tokenize as viet_sent_tokenize
from underthesea import word_tokenize as viet_word_tokenize

# persian
from hazm import sent_tokenize as pers_sent_tokenize
from hazm import word_tokenize as pers_word_tokenize

# hebrew
import hebrew_tokenizer as ht

# japanese
import tinysegmenter

# korean
from soynlp.tokenizer import MaxScoreTokenizer

kor_tokenizer = MaxScoreTokenizer()

# lower level of loggers
import logging

loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)

arab_tokenizer = WordTokenizer('arabic')

bengali_sent_tokenizer = TokenizeSentence('bengali')
hindi_sent_tokenizer = TokenizeSentence('hindi')
telugu_sent_tokenizer = TokenizeSentence('telugu')
marathi_sent_tokenizer = TokenizeSentence('marathi')

jap_segmenter = tinysegmenter.TinySegmenter()

# indian languages
print("INFO: loading indian lgg")
from inltk.inltk import setup as ind_setup

ind_setup("hi")
ind_setup("pa")
ind_setup("gu")
ind_setup("kn")
ind_setup("ml")
ind_setup("mr")
ind_setup("bn")
ind_setup("ta")
ind_setup("ur")
ind_setup("ne")

# tokenizers:
tok_hi = IndicTokenizer("hi")
tok_pa = IndicTokenizer("pa")
tok_gu = IndicTokenizer("gu")
tok_kn = IndicTokenizer("kn")
tok_ml = IndicTokenizer("ml")
tok_mr = IndicTokenizer("mr")
tok_bn = IndicTokenizer("bn")
tok_ta = IndicTokenizer("ta")
tok_ur = IndicTokenizer("ur")
tok_ne = IndicTokenizer("ne")
print("--> all done")

# word
pat_punct = string.punctuation.replace("_", "").replace("-", "")
pat_spe = "\n\t\r ।"
NON_TOK_PAT = "[" + pat_punct + pat_spe + "]+"
RE_NON_TOK_PAT = re.compile(NON_TOK_PAT)

# sent
TOK_SENT_SEP = "[.?:!;·˳̥۰ᴉ•․﮳。：；？！」』؟:।]+"
RE_TOK_SENT_SEP = re.compile(TOK_SENT_SEP)


def sent_latin_tok(txt):
    """text to sentences for latin texts"""
    return RE_TOK_SENT_SEP.split(txt)


def word_latin_tok(txt):
    """sentence to words for latin texts"""
    return RE_NON_TOK_PAT.split(txt)


def sent_bangali_tok(txt):
    """text to sentences for bangali texts"""
    return bengali_sent_tokenizer.tokenize(txt)


def sent_hindi_tok(txt):
    """text to sentences for hindi texts"""
    return hindi_sent_tokenizer.tokenize(txt)


def sent_telugu_tok(txt):
    """text to sentences for telugu texts"""
    return telugu_sent_tokenizer.tokenize(txt)


def sent_marathi_tok(txt):
    """text to sentences for marathi texts"""
    return marathi_sent_tokenizer.tokenize(txt)


def word_arab_tok(txt):
    """sentence to words for arab texts"""
    return arab_tokenizer.tokenize(txt)


def word_hebrew_tok(txt):
    """sentence to words for hebrew texts"""
    return [e[1] for e in list(ht.tokenize(txt))]


def word_jap_tok(txt):
    """sentence to words for japanese texts"""
    return jap_segmenter.tokenize(txt)


# indian
def word_hi_tok(txt):
    """sentence to words for hindi texts"""
    return tok_hi.tokenizer(txt)


def word_pa_tok(txt):
    """sentence to words for punjabi texts"""
    return tok_pa.tokenizer(txt)


def word_gu_tok(txt):
    """sentence to words for gujarati texts"""
    return tok_gu.tokenizer(txt)


def word_kor_tok(txt):
    """sentence to words for korean texts"""
    return kor_tokenizer.tokenize(txt)


def word_ml_tok(txt):
    """sentence to words for malayalam texts"""
    return tok_ml.tokenizer(txt)


def word_mr_tok(txt):
    """sentence to words for marathi texts"""
    return tok_mr.tokenizer(txt)


def word_bn_tok(txt):
    """sentence to words for bengali texts"""
    return tok_bn.tokenizer(txt)


def word_ta_tok(txt):
    """sentence to words for tamil texts"""
    return tok_ta.tokenizer(txt)


def word_ur_tok(txt):
    """sentence to words for urdu texts"""
    return tok_ur.tokenizer(txt)


def word_ne_tok(txt):
    """sentence to words for nepali texts"""
    return tok_ne.tokenizer(txt)


def word_kn_tok(txt):
    """sentence to words for kannada texts"""
    return tok_kn.tokenizer(txt)


def word_chinese_tok(txt):
    """sentence to words for chinese texts"""
    return list([e[0] for e in jieba.tokenize(txt)])


def sent_chinese_tok(txt):
    """sentence to words for chinese texts"""
    return RE_TOK_SENT_SEP.split(txt)


def word_persian_tok(txt):
    """sentence to words for persian texts"""
    return pers_word_tokenize(txt)


def sent_persian_tok(txt):
    """sentence to words for persian texts"""
    return pers_sent_tokenize(txt)


def word_thai_tok(txt):
    """sentence to words for thai texts"""
    return thai_word_tokenize(txt)


def sent_thai_tok(txt):
    """text to sentences for thai texts"""
    return thai_sent_tokenize(txt)


def word_viet_tok(txt):
    """sentence to words for vietnamese texts"""
    return viet_word_tokenize(txt)


def sent_viet_tok(txt):
    """text to sentences for vietnamese texts"""
    return viet_sent_tokenize(txt)


WORD_TOK_ID_TO_TOKENIZER = {
    "std": "word_latin_tok",
    "chn": "word_chinese_tok",
    "ar": "word_arab_tok",
    "fa": "word_persian_tok",
    "hi": "word_hi_tok",
    "pa": "word_pa_tok",
    "gu": "word_gu_tok",
    "kn": "word_kn_tok",
    "ml": "word_ml_tok",
    "mr": "word_mr_tok",
    "bn": "word_bn_tok",
    "ta": "word_ta_tok",
    "ur": "word_ur_tok",
    "ne": "word_ne_tok",
    "he": "word_hebrew_tok",
    "ja": "word_jap_tok",
    "th": "word_thai_tok",
    "vi": "word_viet_tok",
    "ko": "word_kor_tok",
}

SENT_TOK_ID_TO_TOKENIZER = {
    "std": "sent_latin_tok",
    "bn": "sent_bangali_tok",
    "hi": "sent_hindi_tok",
    "te": "sent_telugu_tok",
    "mr": "sent_marathi_tok",
    "fa": "sent_persian_tok",
    "th": "sent_thai_tok",
    "vi": "sent_viet_tok",
    # "ko": "sent_kor_tok",
}

__all__ = ["WORD_TOK_ID_TO_TOKENIZER", "SENT_TOK_ID_TO_TOKENIZER", "word_latin_tok", "word_chinese_tok",
           "word_arab_tok", "word_persian_tok", "word_hi_tok", "word_pa_tok",
           "word_gu_tok", "word_kn_tok", "word_ml_tok", "word_mr_tok", "word_bn_tok", "word_ta_tok", "word_ur_tok",
           "word_ne_tok", "word_hebrew_tok", "word_jap_tok", "word_thai_tok", "word_viet_tok", "word_kor_tok",
           "sent_latin_tok", "sent_bangali_tok", "sent_hindi_tok", "sent_telugu_tok", "sent_marathi_tok",
           "sent_persian_tok", "sent_thai_tok", "sent_viet_tok"
           ]
