"""Script handling the parking classification of a domain
A domain may have up to 3 pages: the original, the redirected, the browser visit
This script also handle the consolidation of the results"""
import re
import numpy as np
import pandas as pd
from os.path import join
import os, sys, traceback
from joblib import Parallel, delayed
from tqdm import tqdm
import json

from config import RUN_CONFIG
from utils import PerformanceLogger, clean_link, link_to_domain, CustomJSONizer
from ml.feature_eng import Featurer, MIN_TEXT_LG
from ml.prediction import Predictor
from links_finder import parse_html
from language_patterns import PATTERN_PARKED, get_unicode_set
from url_visitor import request_full_file_target_links, request_full_file_with_browser, get_sample_filenames
from page_processing import get_page_displayed_text, get_page_language

# LOGGING
# lower level of loggers
import logging
loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)

plog = PerformanceLogger(filename="predictions_perf.log", to_screen=True,
                         enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])
plog2 = PerformanceLogger(filename="unit_predict_park_perf.log", to_screen=False,
                          enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])
plog2.enabled = False  # just turning this off manually for now
sam_plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"], "logging"), filename="sam_p.log",
                             enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])

# Registrars
LIST_REGISTRARS = pd.read_csv(join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input",
                                   "hosting_companies_with_tld.csv"), encoding="latin-1").values.tolist()

DICT_REGISTRARS = {}
DICT_REGISTRARS_TO_NAME = {}
for e in LIST_REGISTRARS:
    if e[0].startswith("www."):
        lk_reg = e[0][4::]
    else:
        lk_reg = e[0]

    for tld in e[1].split(";"):
        lk_reg_with_tld = lk_reg + tld
        lg_reg = len(lk_reg_with_tld)

        DICT_REGISTRARS_TO_NAME[lk_reg_with_tld] = lk_reg

        if lg_reg in DICT_REGISTRARS.keys():
            DICT_REGISTRARS[lg_reg][lk_reg_with_tld] = True
        else:
            DICT_REGISTRARS[lg_reg] = {lk_reg_with_tld: True}

MIN_LENGTH_DISPLAYED_TEXT = 5  # Minimum size of displayed text to use it for pattern detection. Otherwise the html is used
THRESHOLD_PAGE_LENGTH = 1000  # Max entities for pattern detection (700)
MIN_LEGNTH_WORD = 4  # minimum size of a word to be considered meaningfull
THRESHOLD_EMPTY = 5  # page considered empty if THRESHOLD_EMPTY or less words
THRESHOLD_HTML_LINES = 150 # redirections ignored if more than THRESHOLD_HTML_LINES lines
THRESHOLD_MIN_DISPLAYED_WORDS_FOR_PATTERNS = 50
THRESHOLD_N_TAGS = 40
BASIC_TAGS = ["td", "tr", "p", "article", "footer", "li", "span", "noscript"]
BASICCONTENT_ELEMENTS = dict(zip(BASIC_TAGS, [True] * len(BASIC_TAGS)))

LIMIT_PRINT_LINKS = 50
THRESH_LETTERS_FULL_JS = 500
THRESH_JS_PREVALANT = 0.5  # 50 % of the page length

PATTERN_WRITE_1 = "document\.writeln *\("
PATTERN_WRITE_2 = "document\.write *\("
PATTERN_WRITE_3 = "\.createElement *\("
PATTERN_WRITE_4 = "\.appendChild *\("

GDADDY_LANDER = "/lander"

SPE_SET = {}

LIST_SPE_WORDS = ["sale", "blocked", "construction", "expired", "index_of", "other", "reserved", "starter"]

# Machine Learning
FT = Featurer(RUN_CONFIG) # features building
PD = Predictor(RUN_CONFIG) # ML prediction



def ignore_same_domain_redirection(resu):
    """Set redirection flag to false if the redirection points toward the same domain
    AND set source= target for http redirections"""
    # history and URL_history
    if "history" in resu:
        resu.pop("history")
    if "URL_history" in resu:
        resu.pop("URL_history")

    # redirection
    if ("target_url" not in resu):
        return resu
    elif resu["target_url"] is None:
        return resu
    else:
        target_url = resu["target_url"]
        url = resu["url"]

        if target_url:
            target_domain = link_to_domain(target_url)
            initial_domain = link_to_domain(url)

            if target_domain == initial_domain:
                resu["is_redirected"] = False
                resu["source"] = "original"
            else:
                resu["source"] = "target"  # for http redirection

        return resu


def select_features(documents):
    """Filter relevant information for the paarking classification (from the HTTP response of a domain)"""
    X = []
    for doc in documents:
        if not doc.is_error:
            try:
                dico = {"url": doc.url, "clean_text": doc.clean_text, "html_text": doc.raw_text,
                        "language": doc.language, "is_error": doc.is_error, "comment": doc.comment,
                        "to_sample": doc.other_variables["to_sample"]
                        }

                if "history" in doc.other_variables:
                    dico["history"] = doc.other_variables["history"]
                if "URL_history" in doc.other_variables:
                    dico["URL_history"] = doc.other_variables["URL_history"]

            except KeyError as e:
                vars_dict = vars(doc)
                sam_plog.it(f"this document is missing a key ({e}), so I'm going to crash now : " + ''.join(
                    [f"\n\t{k} : {repr(v[:100]) if type(v) == str else v.keys() if type(v) == dict else v}" for k, v in
                     vars_dict.items()]))
                raise

            # Target visits
            if "original_url" in doc.other_variables:
                dico["original_url"] = doc.other_variables["original_url"]
            # non text
            if "non_text" in doc.other_variables:
                dico["non_text"] = doc.other_variables["non_text"]

            X.append(dico)
        else:
            dico = {"url": doc.url, "clean_text": doc.clean_text, "html_text": doc.raw_text,
                    "language": doc.language, "is_error": doc.is_error, "comment": doc.comment,
                    "to_sample": doc.other_variables["to_sample"]}

            # Target visits
            if "original_url" in doc.other_variables:
                dico["original_url"] = doc.other_variables["original_url"]
            X.append(dico)
    return X



def print_missing_languages(missing_languages):
    """Print count of urls for which the language is not available in parking patterns (cf )"""
    df_missing = pd.DataFrame(missing_languages, columns=["missing_languages"])
    df_missing["count"] = df_missing["missing_languages"]
    df_missing = df_missing.groupby("missing_languages", as_index=False).agg({"count": "count"}).sort_values(
        by="count", ascending=False)
    ind_other = df_missing["missing_languages"] == "other"
    df_missing = df_missing[~ind_other]
    if len(df_missing) > 0:
        print("WARNING missing language --> english used :\n{}".format(df_missing.to_string()))


def detect_redirection(url, url_histo, histo, frames, window_loc, meta_refresh, n_lines_html):
    """ Identify different types of redirection: HTTP, Meta-Refresh, Iframe, Window Loc"""

    # HTTP redirection
    is_redirected = False
    redirection_type = ""
    target_link = None

    if histo is not None:
        if histo > 0:
            last_url = str(url_histo).split("___")[0]
            # last_url = link_to_domain(last_url)
            last_url = clean_link(last_url)

            # if not last_url.endswith(link_to_domain(url)):  # domain has changed
            if last_url != url:  # if url has changed: need to revisit target page
                is_redirected = True
                redirection_type = "http"
                target_link = last_url

    # meta refresh
    if (not is_redirected) and (meta_refresh is not None):
        target_link = meta_refresh["link"]
        # target_link_clean = link_to_domain(target_link)
        target_link_clean = clean_link(target_link)
        # url_clean = link_to_domain(url)
        url_clean = clean_link(url)
        # if not target_link_clean.startswith(url_clean):
        if target_link_clean != url_clean:
            is_redirected = True
            redirection_type = "meta_refresh"

    # Iframe redirection
    total_significance = sum([0] + [e["significance"] for e in frames])
    if len(frames) > 0:
        if not is_redirected:
            # The total number of frame + frameset must be equal to 1
            if total_significance == 1:
                # limit on size of HTML
                if n_lines_html < THRESHOLD_HTML_LINES:
                    for tag in frames:
                        if tag["significance"] == 1:
                            # not inside a basic content tag
                            if tag["parent"] not in BASICCONTENT_ELEMENTS:

                                if tag["type"] == "iframe":
                                    target_link = tag["link"]
                                elif tag["type"] == "frameset":
                                    for frame in tag["frames"]:
                                        if frame["significance"] == 1:
                                            target_link = frame["link"]
                                            break

                                target_link_clean = clean_link(target_link)
                                url_clean = clean_link(url)
                                if target_link_clean != url_clean:
                                    is_redirected = True
                                    target_link = target_link
                                    redirection_type = "iframe"
                            break

    # window location redirection
    if (not is_redirected) and (len(window_loc) == 1) and (n_lines_html < THRESHOLD_HTML_LINES):
        target_link = window_loc[0]["link"]

        # target_link_clean = link_to_domain(target_link)
        # url_clean = link_to_domain(url)
        # if not target_link_clean.startswith(url_clean):
        target_link_clean = clean_link(target_link)
        url_clean = clean_link(url)
        if target_link_clean != url_clean:
            is_redirected = True
            redirection_type = "window_location"

    # print("{}\t{}\t{}".format(url, is_redirected, redirection_type))
    return is_redirected, redirection_type, target_link


def detect_registrar_link(all_links, target_link):
    """Identify links to a registrar"""
    park_service = None
    registrar_found = False
    is_redirected_to_registrar = False

    # redirection link
    if target_link is not None:
        if isinstance(target_link, str):
            lk = link_to_domain(target_link)
            len_lk = len(lk)
            for lg_ref in DICT_REGISTRARS.keys():
                # if lk[0:lg_ref] in DICT_REGISTRARS[lg_ref].keys():
                if lk[-lg_ref::] in DICT_REGISTRARS[lg_ref].keys():
                    # full registrar link
                    if len_lk <= lg_ref:
                        is_redirected_to_registrar = True
                        park_service = lk[-lg_ref::]
                        registrar_found = True
                    # subdomain of registrar link
                    elif lk[-lg_ref - 1] == ".":
                        is_redirected_to_registrar = True
                        park_service = lk[-lg_ref::]
                        registrar_found = True

    if not is_redirected_to_registrar:
        # all links
        list_registrar = []
        for lk in all_links:
            lk = link_to_domain(lk)
            len_lk = len(lk)
            for lg_ref in DICT_REGISTRARS.keys():
                # if lk[0:lg_ref] in DICT_REGISTRARS[lg_ref].keys():
                if lk[-lg_ref::] in DICT_REGISTRARS[lg_ref].keys():
                    # full registrar link
                    if len_lk <= lg_ref:
                        list_registrar.append(lk[-lg_ref::])
                    # subdomain of registrar link
                    elif lk[-lg_ref - 1] == ".":
                        list_registrar.append(lk[-lg_ref::])

        list_name_of_registrars = [DICT_REGISTRARS_TO_NAME[e] for e in list_registrar]
        if len(list(set(list_name_of_registrars))) == 1:
            registrar_found = True
            park_service = list_registrar[0]
        elif len(list_name_of_registrars) > 0:
            park_service = "__".join(list_registrar)[0:LIMIT_PRINT_LINKS]

    return park_service, registrar_found, is_redirected_to_registrar


def detect_parking_pattern(html, txt, lg, url, n_entities):
    """Detect Core + attribute keywords pattern
     = indicate if a "core" word and an "attribute" word are in the same sentence """
    kw_parked = False
    kw_park_notice = None
    missing_language = None
    if (n_entities < THRESHOLD_PAGE_LENGTH) or (lg in ["zh-cn", "zh-tw"]):
        if len(txt) <= MIN_LENGTH_DISPLAYED_TEXT:
            text_to_scan = html
        else:
            text_to_scan = txt

        text_to_scan = re.sub(" {2,}", ".", text_to_scan)

        additional = [url.replace(".", "\."), ("www." + url).replace(".", "\.")]
        list_kw_park_notice = []
        if lg not in PATTERN_PARKED.keys():
            missing_language = lg
            lg = "en"

        charset = get_unicode_set(lg)

        # specific language
        for idx, kw in enumerate(PATTERN_PARKED[lg]["core"] + additional):
            length = len(kw)
            # sentences = re.findall("([a-z ]*" + kw + "[a-z ]*)", text_to_scan, re.IGNORECASE)
            sentences = re.findall("(" + charset + "*" + kw + charset + "*)", text_to_scan, flags=re.IGNORECASE)

            # complexe pattern (with
            if len(sentences) > 0:
                if isinstance(sentences[0], tuple):
                    sentences = [e[0] for e in sentences]

            sentences = [e for e in sentences if (len(e) > 0)]

            if len(sentences) > 0:
                for sent in sentences:
                    # ---
                    for attr in PATTERN_PARKED[lg]["attributes"]:

                        # Ignore if core-attribute excluded
                        if len(attr) > 2:
                            if idx in attr[2]:
                                continue

                        if re.search(attr[0], sent, re.IGNORECASE):

                            # exclusion if a given word is found in the sentence
                            if len(attr) > 3:
                                to_exclude = False
                                for word in attr[3]:
                                    if re.search(word, sent, re.IGNORECASE):
                                        to_exclude = True

                                if to_exclude:
                                    continue

                            kw_parked = True
                            # print(attr[0])
                            # list_kw_park_notice.append(sent)
                            list_kw_park_notice.append(attr[1])
                            break

                    if kw_parked:
                        break
                    # ---

            if kw_parked:
                break

        # English by default
        if (len(list_kw_park_notice) == 0) & (lg != "en"):
            for idx, kw in enumerate(PATTERN_PARKED["en"]["core"]):

                length = len(kw)
                # sentences = re.findall("([a-z ]*" + kw + "[a-z ]*)", text_to_scan, re.IGNORECASE)
                sentences = re.findall("(" + charset + "*" + kw + charset + "*)", text_to_scan, flags=re.IGNORECASE)

                # complexe pattern (with
                if len(sentences) > 0:
                    if isinstance(sentences[0], tuple):
                        sentences = [e[0] for e in sentences]

                sentences = [e for e in sentences if (len(e) > 0)]

                if len(sentences) > 0:
                    for sent in sentences:

                        # ---
                        for attr in PATTERN_PARKED["en"]["attributes"]:

                            # Ignore if core-attribute excluded
                            if len(attr) > 2:
                                if idx in attr[2]:
                                    continue

                            if re.search(attr[0], sent, re.IGNORECASE):
                                # print(attr[0])
                                kw_parked = True
                                # list_kw_park_notice.append(sent)
                                list_kw_park_notice.append(attr[1])
                                break

                        if kw_parked:
                            break
                        # ---

                if kw_parked:
                    break

        kw_park_notice = "___".join(sorted(list(set(list_kw_park_notice))))
    return kw_park_notice, kw_parked, missing_language


def is_javascript_needed(html, no_script_text, lg_displayed_text):
    """
    Indicates if a page needs to be revisited with javascript in order to get full relevant content
    :param html:
    :param no_script_text:
    :return:
    """
    to_revisit_with_js = False
    document_write = None

    if lg_displayed_text < THRESH_LETTERS_FULL_JS:

        # dynamic changes
        pat_w1 = re.search(PATTERN_WRITE_1, html, re.IGNORECASE) is not None
        pat_w2 = re.search(PATTERN_WRITE_2, html, re.IGNORECASE) is not None
        pat_w3 = re.search(PATTERN_WRITE_3, html) is not None
        pat_w4 = re.search(PATTERN_WRITE_4, html) is not None

        if pat_w1 or pat_w2 or pat_w3 or pat_w4:
            document_write = True
            to_revisit_with_js = True
        else:
            document_write = False

        # no script warning
        pat1 = re.search("enable", no_script_text, re.IGNORECASE)
        pat2 = re.search("require", no_script_text, re.IGNORECASE)
        pat3 = re.search("javascript", no_script_text, re.IGNORECASE)
        if (pat1 or pat2) and pat3:
            to_revisit_with_js = True

    return document_write, to_revisit_with_js


def get_spe_words(vocab_array, vocabulary, dico_word_to_class):
    """Identify the presence of the words in LIST_SPE_WORDS"""
    resu = dict(zip(LIST_SPE_WORDS, [0] * len(LIST_SPE_WORDS)))

    non_null_indexes = np.where(vocab_array[0] > 0)
    for idx in list(non_null_indexes[0]):

        word = vocabulary[idx]
        end_class = dico_word_to_class[word]

        if end_class in resu:
            resu[end_class] += 1

    return resu


def check_non_text_park(txt):
    """Check if the word 'park' is in txt"""
    if txt is None:
        return False
    else:
        return (re.search("[^a-z]park", txt, re.IGNORECASE) is not None)


def get_n_unique_words(tokens):
    """Return number of unique words"""
    n = 0
    set_words = set()
    for sent in tokens:
        for wd in sent:
            set_words.add(wd)
    return len(set_words)


class PageParkClassifier():
    """Handles parking classification: collect the various clues THEN apply the classification logic"""
    def __init__(self, feats):
        # All single page data
        self.url = feats["url"]
        self.to_sample = feats["to_sample"]
        self.is_error = feats["is_error"]
        self.comment = feats["comment"]

        # features (typically not present for DNS errors)
        self.txt = str(feats["clean_text"]) if ("clean_text" in feats) else None
        self.html = str(feats["html_text"]) if ("html_text" in feats) else None
        self.lg = feats["language"] if ("language" in feats) else None
        self.histo = feats["history"] if ("history" in feats) else None
        self.url_histo = str(feats["URL_history"]) if ("URL_history" in feats) else None
        self.non_text = feats["non_text"] if ("non_text" in feats) else None

        # general
        self.missing_language = None
        # self.sample_data = {}
        self.html_data = {}
        self.pred_is_parked = False

        # Redirections
        self.is_redirected = None
        self.redirection_type = None
        self.target_link = None
        self.is_redirected_different_domain = False

        # Registrar
        self.park_service = None
        self.registrar_found = None
        self.is_redirected_to_registrar = None

        # JS
        self.document_write = None
        self.to_revisit_with_js = None
        self.is_full_js_parked = None
        self.is_non_text_park = None

        # ML classification
        self.n_words = None
        self.n_unique_words = None
        self.n_sentences = None
        self.text_feat = None
        self.pred_ml_park = False
        self.ml_feat_special_words = {}

        # Emptiness
        self.js_or_iframe_found = None
        self.pred_is_empty = None

        # parking pattern
        self.kw_park_notice = None
        self.kw_parked = None

        # source or target page
        if ("original_url" in feats) and (feats["original_url"] is not None):
            self.original_url = feats["original_url"]
            self.source = "target"
        else:
            self.original_url = None
            self.source = "original"

        plog2.it(f"Performing unit predict parking on {self.url}")

    def save_sample_data(self, resu):
        """Save page data for reuse in Labtool"""
        if self.source == "original":
            original_url = self.url
        else:
            original_url = self.original_url
        ss_filename, raw_filename, clean_filename, json_filename = get_sample_filenames(original_url)

        if RUN_CONFIG["DO_SAMPLING"] and self.to_sample:
            sam_plog.it(f"writing raw and clean html for : {original_url}")
            with open(raw_filename, 'w', encoding='UTF8') as f:
                f.write(self.html)
            with open(clean_filename, 'w', encoding='UTF8') as f:
                f.write(self.txt)

            sample_data = {"ss_filename": ss_filename, "raw_filename": raw_filename,
                           "clean_filename": clean_filename, "json_filename": json_filename}
            resu.update(sample_data)

            with open(json_filename, 'w', encoding='UTF8') as f:
                json.dump(resu, f, cls=CustomJSONizer)

        return resu

    def extract_html_data(self):
        """Extract various data from the HTML by parsing it"""
        html_data = parse_html(self.html, self.url)
        expected_data = ["all_links", "frames", "flag_iframe", "flag_js_found", "window_loc", "meta_refresh",
                         "n_lines_html", "dico_html_complexity", "no_script_text", "lg_inline_script"]
        for colo in expected_data:
            assert colo in html_data, f"Value error: missing {colo} in html_data"

        self.html_data = html_data

    def identify_redirection(self):
        """Identify presence of a redirection
        (= the current page is only used to send to another page) """
        assert len(self.html_data) > 0

        url = self.url
        url_histo = self.url_histo
        histo = self.histo
        frames = self.html_data["frames"]
        window_loc = self.html_data["window_loc"]
        meta_refresh = self.html_data["meta_refresh"]
        n_lines_html = self.html_data["n_lines_html"]
        self.is_redirected, self.redirection_type, self.target_link = detect_redirection(url,
                                                                                         url_histo,
                                                                                         histo,
                                                                                         frames,
                                                                                         window_loc, meta_refresh,
                                                                                         n_lines_html)

        # is the page redirected to a different domain
        if (self.source == "target") or (self.redirection_type == "http"):
            if self.redirection_type == "http":
                original_domain = link_to_domain(url)
                target_domain = link_to_domain(str(url_histo).split("___")[0])
            else:
                original_domain = link_to_domain(self.original_url)
                target_domain = link_to_domain(url)

            if target_domain != original_domain:
                self.is_redirected_different_domain = True
            plog2.it(f"original_domain: {original_domain} | target_domain: {target_domain}")

    def identify_registrar(self):
        """Detect present of registrar links from the reference list"""
        # Registrar research
        all_links = self.html_data["all_links"]
        target_link = self.target_link
        # general registrar links
        self.park_service, self.registrar_found, self.is_redirected_to_registrar = detect_registrar_link(all_links,
                                                                                                         target_link)

        # godaddy lander page
        if self.target_link is not None:
            if self.target_link.lower().endswith(self.url + GDADDY_LANDER):
                self.park_service = "GoDaddy"
                self.registrar_found = True
                self.is_redirected_to_registrar = True

    def detect_javascript_requirement(self):
        """Detect clues that a browser visit is necessary"""
        html = self.html
        no_script_text = self.html_data["no_script_text"]
        lg_text = len(self.txt)
        self.document_write, self.to_revisit_with_js = is_javascript_needed(html, no_script_text, lg_text)

    def is_redirection_necessary(self):
        """Conclude on redirection page visit decision"""
        is_first_redirection = self.source == "original"
        is_redirection = self.redirection_type in ["window_loc", "iframe", "meta_refresh"]
        is_not_registrar = not self.registrar_found
        return is_first_redirection and is_redirection and is_not_registrar

    def ml_classification(self):
        """Classify the page by Machine Learning approach"""
        # ML features
        all_ml_features, infos_params = FT.transform(self.txt, self.url, self.lg)
        self.n_words = infos_params["n_words"]
        self.n_unique_words = infos_params["n_unique_words"]
        self.n_sentences = infos_params["n_sentences"]
        self.text_feat = infos_params["text_feat"]

        # ML prediction
        if self.n_sentences > 0 and self.n_words > MIN_TEXT_LG:  # non trivial page
            self.pred_ml_park = PD.predict(all_ml_features)
            self.ml_feat_special_words = get_spe_words(self.text_feat, FT.VECTORIZER.vocabulary, FT.DICO_WORD_TO_CLASS)

        else:
            self.pred_ml_park = False
            self.ml_feat_special_words = get_spe_words(np.zeros((1)), FT.VECTORIZER.vocabulary,
                                                       FT.DICO_WORD_TO_CLASS)

    def identify_empty_page(self):
        """Conclude on wether a page is empty"""
        flag_js_found = self.html_data["flag_js_found"]
        flag_iframe = self.html_data["flag_iframe"]
        all_links = self.html_data["all_links"]
        n_unique_words = self.n_unique_words

        # EMPTINESS
        # Javascript/Iframe flag
        self.js_or_iframe_found = False
        if flag_js_found or flag_iframe:
            self.js_or_iframe_found = True
        # Empty page
        self.pred_is_empty = (n_unique_words <= THRESHOLD_EMPTY) and (not self.js_or_iframe_found) and (
                len(all_links) == 0)
        plog2.it(f"pred_is_empty : {self.pred_is_empty}")

    def identify_parking_pattern(self):
        """Detect Core + attribute keywords pattern"""
        html = self.html
        txt = self.txt
        url = self.url
        lg = self.lg
        n_words = self.n_words

        # Pattern research
        self.kw_park_notice, self.kw_parked, self.missing_language = detect_parking_pattern(html, txt, lg, url, n_words)
        plog2.it(f"parking pattern: {self.kw_park_notice} | {self.kw_parked} | {self.missing_language}")

    def validate_registrar(self):
        """Validate if the page is a registrar by cross-checking clues"""
        n_tags = (self.html_data["dico_html_complexity"])["tag_quantity"]
        registrar_found = self.registrar_found
        if registrar_found:
            if self.is_redirected_to_registrar:
                registrar_found = True
            elif self.pred_ml_park:
                registrar_found = True
            elif self.n_words > THRESHOLD_MIN_DISPLAYED_WORDS_FOR_PATTERNS:  # non trivial page with no parking pattern
                registrar_found = False
            elif n_tags > THRESHOLD_N_TAGS:  # generated by website builder
                registrar_found = False

        self.registrar_found = registrar_found
        plog2.it(f"registrar_found: {registrar_found}")

    def validate_javascript_requirement(self):
        """Validate if the page requires a browser visit by cross-checking clues"""
        lg_html = len(self.html)
        lg_txt = len(self.txt)
        non_text = self.non_text
        to_sample = self.to_sample
        lg_inline_script = self.html_data["lg_inline_script"]
        registrar_found = self.registrar_found
        pred_ml_park = self.pred_ml_park
        pred_is_empty = self.pred_is_empty
        to_revisit_with_js = self.to_revisit_with_js

        # Ratio JS
        ratio_script_to_html = lg_inline_script / (lg_html + 1e-5)
        is_js_prevalant = (ratio_script_to_html > THRESH_JS_PREVALANT) & (lg_txt < THRESH_LETTERS_FULL_JS)
        plog2.it(f"JS ratio: {ratio_script_to_html} | is_js_prevalant : {is_js_prevalant}")

        # NON-TEXT
        # Non-text_parking clues
        is_non_text_park = check_non_text_park(non_text)
        is_full_js_parked = is_non_text_park & (lg_txt < THRESH_LETTERS_FULL_JS)

        # No need to interpret javascript if enough evidence to conclude
        if registrar_found or pred_ml_park or is_full_js_parked or pred_is_empty:
            to_revisit_with_js = False
        elif is_js_prevalant:
            to_revisit_with_js = True
        plog2.it(
            f"is_non_text_park : {is_non_text_park} | is_full_js_parked : {is_full_js_parked} | to_revisit_with_js : {to_revisit_with_js} | to_sample : {to_sample}")

        self.to_revisit_with_js = to_revisit_with_js
        self.is_full_js_parked = is_full_js_parked
        self.is_non_text_park = is_non_text_park

    def gather_error_result(self):
        """Gather output result in case of error at request stage"""
        resu = {
            "url": self.url,
            "source": self.source,
            "original_url": self.original_url,
            "to_sample": False,

            "is_redirected": False,
            "to_revisit_with_js": False,
            "pred_is_parked": True,
        }

        # Propagate error if redirection to an error
        if self.source == "target":
            resu["is_error"] = self.is_error
            resu["comment"] = self.comment

        return resu

    def gather_normal_result(self):
        """Gather output result in case of normal page"""
        # final
        resu = {
            # general
            "url": self.url,
            "to_sample": self.to_sample,
            "source": self.source,
            "original_url": self.original_url,

            # content data
            "flag_iframe": self.html_data["flag_iframe"],
            "flag_js_found": self.html_data["flag_js_found"],
            "n_letter": len(self.txt),
            "n_words": self.n_words,

            # redirection
            "is_redirected": self.is_redirected,
            "redirection_type": self.redirection_type,
            "target_url": self.target_link,
            "is_redirected_to_registrar": self.is_redirected_to_registrar,
            "is_redirected_different_domain": self.is_redirected_different_domain,

            # registrar
            "park_service": self.park_service,
            "registrar_found": self.registrar_found,

            # javascript
            "document_write": self.document_write,
            "to_revisit_with_js": self.to_revisit_with_js,
            "is_full_js_parked": self.is_full_js_parked,
            "is_non_text_park": self.is_non_text_park,

            # parking pattern
            "kw_parked": self.kw_parked,
            "kw_park_notice": self.kw_park_notice,

            # emptiness
            "js_or_iframe_found": self.js_or_iframe_found,
            "pred_is_empty": self.pred_is_empty,

            # ml
            "pred_ml_park": self.pred_ml_park,

            "pred_is_parked": self.pred_is_parked,
        }

        # extra data
        # special words
        for k, v in self.ml_feat_special_words.items():
            resu["ml_feat_" + k] = v

        # maintain history redirections during browser visit
        if self.to_revisit_with_js:
            resu["history"] = self.histo
            resu["URL_history"] = self.url_histo

        # adding html structure
        for k, v in self.html_data["dico_html_complexity"].items():
            resu[k] = v

        # SSL enabled
        try:
            resu["ssl_enabled"] = self.url_histo.startswith("https:")
        except:
            resu["ssl_enabled"] = None

        return resu

    def gather_handle_classification_error(self, type_error, str_error, exc_info, tback):
        """Gather output result in case of unexpected error"""
        url = self.url
        plog.it(f"ERROR in parking classification --> {url} not classified : type:{type_error} message:{str_error}",
                is_error=True)

        plog.it(f"ERROR sys.exc_info: {exc_info}", is_error=True)
        plog.it(f"ERROR traceback: {tback}", is_error=True)

        resu = {
            "url": url,
            "source": self.source,
            "original_url": self.original_url,
            "to_sample": self.to_sample,

            "is_redirected": False,
            "to_revisit_with_js": False,
            "pred_is_parked": True,
        }
        return resu

    def gather_tempo_redirection_result(self):
        """Gather temporary result in case of redirection"""
        resu = {
            "url": self.target_link,
            "source": "target",
            "original_url": self.url,
            "to_sample": self.to_sample,
            "to_revisit_with_js": False
        }
        return resu

    def classify(self):
        """Parking classification logic: collecting clues, classifying, validating some clues"""
        resu = {}
        try:
            if self.is_error:
                # DNS or HTTP error page
                resu = self.gather_error_result()

            else:
                # normal page
                self.extract_html_data()

                self.identify_redirection()

                self.identify_registrar()

                self.detect_javascript_requirement()

                if self.is_redirection_necessary():
                    # the current page is irrelevant for classification
                    resu = self.gather_tempo_redirection_result()
                else:
                    #
                    self.ml_classification()

                    self.identify_empty_page()

                    self.identify_parking_pattern()

                    self.validate_registrar()

                    # parking conclusion
                    if self.registrar_found or self.kw_parked or self.pred_is_empty or self.pred_ml_park:
                        self.pred_is_parked = True
                    else:
                        self.pred_is_parked = False

                    self.validate_javascript_requirement()

                    # gather results
                    resu = self.gather_normal_result()

        except Exception as e:

            # exception information
            type_error = str(type(e))
            str_error = str(e)
            exc_info = str(sys.exc_info()[:1])
            tback = str(traceback.print_exc(limit=2, file=sys.stdout))

            resu = self.gather_handle_classification_error(type_error, str_error, exc_info, tback)

        resu = self.save_sample_data(resu)

        return resu, self.missing_language


def single_page_classify_parking(feats):
    """Full parking classification logic application on a Page."""
    clf = PageParkClassifier(feats)
    resu, miss_lgg = clf.classify()
    return resu, miss_lgg


def consolidate_original_and_target_results(preds, preds_second):
    """Overwrite original url result by the redirection target results when non-HTTP redirection identified"""
    print("-------consolidation of source/target links")
    dico_url_to_target_results = dict()
    for resu in preds_second:
        if ("original_url" in resu) and (resu["original_url"] is not None):
            replacement = resu.copy()
            replacement["target_url"] = replacement["url"]
            replacement["url"] = replacement["original_url"]
            dico_url_to_target_results[resu["original_url"]] = replacement

    for rf in preds:
        if rf["original_url"] is not None:
            if rf["original_url"] in dico_url_to_target_results.keys():
                for ky, val in dico_url_to_target_results[rf["original_url"]].items():
                    if ky != "to_sample":
                        rf[ky] = val

        # add target link for http  redirection (target is visited by default)
        if "http_redirection_link" in rf:
            if rf["http_redirection_link"] is not None:
                rf["source"] = "target"
                rf["target_url"] = rf["http_redirection_link"]
    return preds


def consolidate_original_and_js_rendered_results(preds, preds_js):
    """Overwrite original url result by the redirection target results when non-HTTP redirection identified"""
    print("-------consolidation of source pages AND js rendered pages")
    dico_url_to_target_results = dict()
    for resu in preds_js:
        replacement = resu.copy()
        if ("original_url" in replacement) and (replacement["original_url"] is not None):
            replacement["target_url"] = replacement["url"]
            replacement["url"] = replacement["original_url"]
            dico_url_to_target_results[resu["original_url"]] = replacement
        else:
            dico_url_to_target_results[resu["url"]] = replacement

    for rf in preds:
        # link target results to original url
        if ("original_url" in rf) and (rf["original_url"] is not None):
            if rf["original_url"] in dico_url_to_target_results:
                for ky, val in dico_url_to_target_results[rf["original_url"]].items():
                    rf[ky] = val
                rf["is_js_rendered_results"] = True
            else:
                rf["is_js_rendered_results"] = False
        else:
            if rf["url"] in dico_url_to_target_results:
                for ky, val in dico_url_to_target_results[rf["url"]].items():
                    rf[ky] = val
                rf["is_js_rendered_results"] = True
            else:
                rf["is_js_rendered_results"] = False

    return preds


def predict_parking(documents):
    """Predict the category of a Domain: HTTP Errors, Parking Notice, Normal...
    Attempt the original Page first,
    if a redirection is needed,Visit and attempt the Redirected Page
    if a Browser visit is necessary,Visit and attempt the Browsed Page"""
    plog.perf_go(f"Logging on {len(documents)} urls {documents[0].url} -> {documents[-1].url}")

    # Original Page classification
    print("-------parking classification of original links")
    plog.it("Attempting to classify original links")
    # features
    X_first = select_features(documents)

    plog.it(f"Performing first classification. MULTI_PROCESSING: {RUN_CONFIG['MULTI_PROCESSING']}")
    plog2.perf_go(f"Performing first classification on {len(documents)} urls")
    if RUN_CONFIG["MULTI_PROCESSING"]:
        plog.it("Performing multi-processing first classification")
        list_res = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"], require='sharedmem')(
            delayed(single_page_classify_parking)(batch) for batch in
            tqdm(X_first))
    else:
        # non parallel
        list_res = []
        for doc in X_first:
            plog.it(f"performing single-thread first classification on {doc['url']}")
            if 'original_url' in doc:
                plog.it(f"original_url : {doc['original_url']} -> {doc['url']}")

            list_res.append(single_page_classify_parking(doc))

    plog2.perf_end("Completed first classification")
    plog.it(f"completed first classification with {len(list_res)} results")
    plog.it("checking languages")

    preds = [e[0] for e in list_res]

    # Redirection Pages visit & classification
    target_links_to_visit = [e for e in preds if (e["source"] == "target") or e["to_sample"]]
    plog.it(f"Redirection targets to visit: {len(target_links_to_visit)}")
    if len(target_links_to_visit) > 0:
        plog.it("-------Visit of target links for {} urls".format(len(target_links_to_visit)))
        list_target_websites = request_full_file_target_links(target_links_to_visit)

        plog.it("-------page processing of  target links for {} urls".format(len(list_target_websites)))
        list_target_websites = get_page_displayed_text(list_target_websites)
        list_target_websites = get_page_language(list_target_websites)

        plog.it("-------parking classification of  target links")
        # Second classification
        X_target_websites = select_features(list_target_websites)
        plog.it("completed second classification")

        plog2.perf_go(f"Performing second classification on {len(target_links_to_visit)} urls")
        # predict
        if RUN_CONFIG["MULTI_PROCESSING"]:
            plog.it("Performing multi-processing second classification")
            list_res_second = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"], require='sharedmem')(
                delayed(single_page_classify_parking)(website) for website in tqdm(X_target_websites))
        else:
            # --non parallel
            list_res_second = []
            for website in X_target_websites:
                list_res_second.append(single_page_classify_parking(website))
                plog.it(f"performing single-thread second classification on {website['url']}")

        plog2.perf_end("second classification complete")
        plog.it("completed second classification")
        plog.it("second language check")

        preds_second = [e[0] for e in list_res_second]

        # consolidation
        plog.it("consolidating original and target results")
        preds = consolidate_original_and_target_results(preds, preds_second)

    # Revisit with Browser and classification
    if ("DO_JS_INTERPRETATION" in RUN_CONFIG) and RUN_CONFIG["DO_JS_INTERPRETATION"]:
        plog.it("performing JS interpretation")
        links_to_revisit_with_js = [e for e in preds if (e["to_revisit_with_js"] or e["to_sample"])]

        if len(links_to_revisit_with_js) > 0:
            print("-------Javascript interpretation for {} urls".format(len(links_to_revisit_with_js)))
            list_js_rendered_websites = request_full_file_with_browser(links_to_revisit_with_js)

            print("-------page processing of  js rendered pages for urls")
            list_js_rendered_websites = get_page_displayed_text(list_js_rendered_websites)
            list_js_rendered_websites = get_page_language(list_js_rendered_websites)

            print("-------parking classification of  js rendered pages")
            # Second classification
            X_js_rendered_websites = select_features(list_js_rendered_websites)

            plog2.perf_go(f"Performing JS classification on {len(links_to_revisit_with_js)} urls")
            # predict
            if RUN_CONFIG["MULTI_PROCESSING"]:
                list_res_js = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"], require='sharedmem')(
                    delayed(single_page_classify_parking)(website) for website in tqdm(X_js_rendered_websites))
            else:
                # --non parallel
                list_res_js = []
                for website in X_js_rendered_websites:
                    list_res_js.append(single_page_classify_parking(website))

            plog2.perf_end("JS Classification complete")
            # print("PREDS_JS: ")
            # pprint(list_res_js)

            preds_js = [e[0] for e in list_res_js]

            # consolidation
            preds = consolidate_original_and_js_rendered_results(preds, preds_js)

    plog.it("doing last cleanup")
    # Remove redirection to same domain
    preds = [ignore_same_domain_redirection(resu) for resu in preds]

    df_resu = pd.DataFrame(preds)

    plog.perf_end("Returning result, all done.")

    return df_resu


if __name__ == '__main__':
    pass
