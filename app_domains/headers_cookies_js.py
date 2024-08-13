"""Script to collect and analyse Headers, Cookies and Javascript libraries"""

import re
from os.path import join
import pandas as pd

from ref_headers_cookies_js import LIST_HCJ_FEATURES
from url_visitor import PREF_HDRS, CL_HDRS, PREF_COOK, CL_COOKS, dico_to_website
from config import MAIN_DIR

from js_links_finder import find_js_links
from utils import load_obj, normalize_hcj_list, normalize_hcj_keys, normalize_hcj_value

PREF_HCJ = "fx_hcj_"


def extract_jslibs_single_page(doc):
    """Exctract the javascript libraries used in one page"""
    dico_resu = {"url": doc.url}
    if (doc.raw_text is not None) & (re.search("[a-zA-Z]", str(doc.raw_text)) is not None):
        all_links, contents, flag_js_found, flag_php_found = find_js_links(doc.raw_text)

        libs = [str(e).lower() for e in all_links if e.endswith(".js")]
        for i in range(len(libs)):
            if libs[i].endswith(".min.js"):
                libs[i] = libs[i][0:-7] + ".js"

        libs = sorted(libs)

        dico_resu["fx_js_n_libs"] = len(libs)
        dico_resu["fx_js_libraries"] = libs

    else:
        dico_resu["fx_js_libraries"] = []

    return dico_resu


def identify_page_services(set_js, set_hdr, set_cook):
    """Identify all features from Headers/Cookies/Javascript as defined in LIST_HCJ_FEATURES"""
    dico_hcj = {}

    for feature in LIST_HCJ_FEATURES:
        n_clues = 0
        feature_name = f"{PREF_HCJ}_{feature['type']}_{feature['name']}"

        # headers clues
        for ky in feature["HD"]:
            if normalize_hcj_value(f"{PREF_HDRS}{ky}") in set_hdr:
                n_clues += 1

        # cookies clues
        for ky in feature["CK"]:
            if normalize_hcj_value(f"{PREF_COOK}{ky}") in set_cook:
                n_clues += 1

        # jslibs clues
        for ky in feature["JS"]:
            if normalize_hcj_value(ky) in set_js:
                n_clues += 1

        # wrap
        dico_hcj[feature_name] = True if n_clues > 0 else False

    return dico_hcj


def analyse_headers_cookies_javascript(documents):
    """Extract HCJ features from all domains"""
    # HCJ = Headers, Cookies, Javascript
    # quantities
    # features

    url = "url"
    df = []
    for doc in documents:
        # if CL_HDRS in doc.other_variables:
        dico = {url: doc.url}

        if doc.is_error:
            dico_hcj = {}
            for feature in LIST_HCJ_FEATURES:
                feature_name = f"{PREF_HCJ}_{feature['type']}_{feature['name']}"
                dico_hcj[feature_name] = False
        else:

            # JS libraries
            if doc.raw_text is not None:
                set_js = normalize_hcj_list(extract_jslibs_single_page(doc))
            else:
                set_js = []

            # headers
            if CL_HDRS in doc.other_variables:
                set_hdr = normalize_hcj_keys(doc.other_variables[CL_HDRS])
            else:
                set_hdr = {}

            # cookies
            if CL_COOKS in doc.other_variables:
                set_cook = normalize_hcj_keys(doc.other_variables[CL_COOKS])
            else:
                set_cook = {}

            # Identify features
            dico_hcj = identify_page_services(set_js, set_hdr, set_cook)

        dico.update(dico_hcj)
        df.append(dico)

    if len(df) > 0:
        df = pd.DataFrame(df)
    else:
        df = pd.DataFrame([], columns=["url"])

    return df


if __name__ == "__main__":
    p_in = join(MAIN_DIR, "inter")
    obj_name = r"documents_commini_5.csv.chunk.0.csv"
    fp_out = join(MAIN_DIR, "output", "hcj_output.csv")

    obj = load_obj(obj_name, p_in)

    obj = [dico_to_website(e) for e in obj]

    dft = analyse_headers_cookies_javascript(obj)
    dft.to_csv(fp_out, index=False)
