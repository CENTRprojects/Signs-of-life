"""Script handling the social media data extraction steps"""
import re
import numpy as np
import pandas as pd
import tldextract
from Levenshtein._levenshtein import distance
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)

from config import RUN_CONFIG, PLOG
from utils import clean_link, gather_url_saved, remove_url_saved, link_to_domain, save_obj
from links_finder import parse_html_for_sm
from url_visitor import request_full_file_target_links

THRESHOLD_HTML_LINES = 150
BASIC_TAGS = ["td", "tr", "p", "article", "footer", "li", "span", "noscript"]
BASICCONTENT_ELEMENTS = dict(zip(BASIC_TAGS, [True] * len(BASIC_TAGS)))

PREFS_SM = "https*://w*w*w*\.*"
PATTERN_FACEBOOK = "facebook.com/([^';!,\"\?<>\n\t\r /\\^]*)"
PATTERN_TWITTER = "twitter.com/([^';!,\"\?<>\n\t\r /\\^]*)"
PATTERN_LINKEDIN = "linkedin.com/([^';!,\"\?<>\n\t\r /\\^]*)"
PATTERN_REDDIT = "reddit.com/([^';!,\"\?<>\n\t\r /\\^]*)"
PATTERN_INSTAGRAM = "instagram.com/([^';!,\"\?<>\n\t\r /\\^]*)"
PATTERN_GITHUB = "github.com/([^';!,\"\?<>\n\t\r /\\^]*)"

SMS = ["fb", "tw", "lk", "rd", "ig", "gh"]
DICO_SM_FULL_NAME = {"fb": "facebook", "tw": "twitter", "lk": "linkedin", "rd": "reddit",
                     "ig": "instagram", "gh": "github"}
DICO_SMS_PATS = {"fb": PATTERN_FACEBOOK, "tw": PATTERN_TWITTER, "lk": PATTERN_LINKEDIN, "rd": PATTERN_REDDIT,
                 "ig": PATTERN_INSTAGRAM, "gh": PATTERN_GITHUB}


def identify_social_media(documents):
    """Identify social media activity of a list of domains"""

    print("-------SM identification of original links")
    # features
    X_first = select_features_sm(documents)

    # First classification
    if RUN_CONFIG["MULTI_PROCESSING"]:
        list_res = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
            delayed(unit_identify_sm)(batch) for batch in tqdm(X_first))
    else:
        # non parallel
        list_res = []
        for doc in X_first:
            # ----------------------
            # if SPE_URL is not None:
            # if doc["url"] != "16808d.com":
            # if doc["url"] not in  SPE_SET:
            #     continue
            # ----------------------

            list_res.append(unit_identify_sm(doc))

    preds = list_res

    # Redirection targets visit
    target_links_to_visit = [e for e in preds if (e["source"] == "target")]
    if len(target_links_to_visit) > 0:
        print("-------Visit of target links for {} urls".format(len(target_links_to_visit)))
        list_target_websites = request_full_file_target_links(target_links_to_visit)

        print("-------SM identification of  target links")
        # Second classification
        X_target_websites = select_features_sm(list_target_websites)

        # predict
        if RUN_CONFIG["MULTI_PROCESSING"]:
            list_res_second = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
                delayed(unit_identify_sm)(website) for website in tqdm(X_target_websites))
        else:
            # --non parallel
            list_res_second = []
            for website in X_target_websites:
                list_res_second.append(unit_identify_sm(website))
        preds_second = list_res_second

        # consolidation
        preds = consolidate_original_and_target_results(preds, preds_second)

    # Remove redirection to same domain
    preds = [correct_and_cleanup(resu) for resu in preds]

    df_resu = pd.DataFrame(preds)
    return df_resu


def correct_and_cleanup(resu):
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

        if target_url and url:
            target_domain = link_to_domain(target_url)
            initial_domain = link_to_domain(url)

            if target_domain == initial_domain:
                resu["is_redirected"] = False
                resu["source"] = "original"
            else:
                resu["source"] = "target"  # for http redirection
        return resu


def select_features_sm(documents):
    """Filter relevant information for the social media extraction (from the HTTP response of a domain)"""
    X = []
    for doc in documents:
        if not doc.is_error:
            dico = {"url": doc.url, "html_text": doc.raw_text, "is_error": doc.is_error}

            if "history" in doc.other_variables:
                dico["history"] = doc.other_variables["history"]
            if "URL_history" in doc.other_variables:
                dico["URL_history"] = doc.other_variables["URL_history"]

            # Target visits
            if "original_url" in doc.other_variables:
                dico["original_url"] = doc.other_variables["original_url"]

            X.append(dico)
        else:
            dico = {"url": doc.url, "is_error": doc.is_error}
            X.append(dico)
    return X


def consolidate_original_and_target_results(preds, preds_second):
    """Overwrite original url result by the redirection target results when non-HTTP redirection identified"""
    print("-------consolidation of source/target links")
    dico_url_to_target_results = dict()
    for resu in preds_second:
        if ("original_url" in resu) and (resu["original_url"] is not None): # in case of error with target page visit
            replacement = resu.copy()
            replacement["target_url"] = replacement["url"]
            replacement["url"] = replacement["original_url"]
            dico_url_to_target_results[resu["original_url"]] = replacement

    for rf in preds:
        # link target results to original url
        if rf["original_url"] is not None:
            if rf["original_url"] in dico_url_to_target_results.keys():
                for ky, val in dico_url_to_target_results[rf["original_url"]].items():
                    rf[ky] = val

        # add target link for http  redirection (target is visited by default)
        if "http_redirection_link" in rf:
            if rf["http_redirection_link"] is not None:
                rf["source"] = "target"
                rf["target_url"] = rf["http_redirection_link"]
    return preds


def identifiy_redirection(url, url_histo, histo, frames, window_loc, meta_refresh, n_lines_html):
    """ Identify type of redirection: HTTP, Meta-Refresh, Iframe, Window Loc"""

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


def clean_up_url_sm(url_extension):
    """Remove url parameters from url"""
    url_extension = url_extension.split("?")[0]
    url_extension = url_extension.split("&")[0]
    return url_extension.lower()


def get_sm_features(all_links):
    """Identify social medias among all_links"""
    feats = {}

    for sm in SMS:
        sm_links = []

        # filter sm
        for link, path, tp in all_links:
            # loop sm
            pat = DICO_SMS_PATS[sm]
            res_pat = re.search(pat, link, re.IGNORECASE)
            if res_pat:
                lk_name = res_pat.groups()[0]
                lk_name = clean_up_url_sm(lk_name)
                sm_links.append((lk_name, path, tp))

        # cnt
        cnt = len(sm_links)

        # cnt
        cnt_unique = len(list(set([e[0] for e in sm_links])))

        # links
        lks = "__".join([e[0] for e in sm_links])

        # paths
        path = "__".join([e[1] for e in sm_links])

        # type
        tp = "__".join([e[2] for e in sm_links])

        resu_sm = {
            "feat_" + sm + "_cnt": cnt,
            "feat_" + sm + "_cntunq": cnt_unique,
            "feat_" + sm + "_lks": lks,
            "feat_" + sm + "_paths": path,
            "feat_" + sm + "_types": tp,
        }
        feats.update(resu_sm)

    return feats


def remove_tld(url):
    """Remove TLD of url"""
    return tldextract.extract(url)[1]

def custo_distance(name, url_host):
    """Custom distance between two text (name and url_host) with Levenshtein distance as general case"""
    if name == url_host:
        return -1
    elif name in url_host:
        return 0
    elif url_host in name:
        return 0
    else:
        return distance(name, url_host)

def normalize_name(nm, sm):
    """Clean up any generic key word from the social account name"""
    nm = nm.replace ("\\", "/")

    if nm.startswith("/"):
        nm = nm[1::]
    if nm.endswith("/"):
        nm = nm[0:-1]

    # linkedin
    if sm =="lk":
        if (nm.split("/")[0] in {"company", "in"}):
            if ("/" in nm):
                nm = nm.split("/")[1]
            else:
                nm = ""
    if sm =="rd":
        if nm.split("/")[0] == "r":
            nm = nm.split("/")[1]
    return nm


def dist_norm(url):
    """normalize url"""
    return url.lower().replace("-", "").replace("_", "")

def format_social_medias(sm_features, url):
    """format social medias columns: boolean of social media presence + account name"""
    sm_output = {}
    sm_to_fill = SMS.copy()

    # url
    url = dist_norm(remove_tld(url))

    # empty
    to_remove = set()
    for sm in sm_to_fill:
        if sm_features["feat_" + sm + "_cntunq"] == 0:
            sm_output["has_" + DICO_SM_FULL_NAME[sm]] = False
            sm_output["feat_" + sm + "_lks"] = ""
            to_remove.add(sm)
    sm_to_fill = [e for e in sm_to_fill if e not in to_remove]

    # best name match
    for sm in sm_to_fill:
        names = list(set(sm_features["feat_" + sm + "_lks"].split("__")))

        # filter .php
        names = [nm for nm in names if not nm.endswith(".php")]

        # normalize
        names = [normalize_name(nm, sm) for nm in names]

        # filter
        names = [nm for nm in names if len(nm)>2]

        if len(names) == 0:
            final_name = ""
            sm_output["has_" + DICO_SM_FULL_NAME[sm]] = False
        elif len(names) == 1:
            final_name = names[0]
            sm_output["has_" + DICO_SM_FULL_NAME[sm]] = True
        else:
            final_name = names[np.argmin([custo_distance(dist_norm(nm), url) for nm in names])]
            sm_output["has_" + DICO_SM_FULL_NAME[sm]] = True

        sm_output["feat_" + sm + "_lks"] = final_name

    return sm_output


def unit_identify_sm(feats):
    """Extract social Media of one page:
    = Redirection Identification + Social media extraction and filtering
    feats: features of one url
    """
    url = feats["url"]

    # source or target page
    if ("original_url" in feats) and (feats["original_url"] is not None):
        original_url = feats["original_url"]
        source = "target"
        # source_results = feats["source_results"]
    else:
        original_url = None
        # source_results = None
        source = "original"

    try:
        is_error = feats["is_error"]
        if not is_error:

            html = str(feats["html_text"])
            histo = feats["history"]
            url_histo = str(feats["URL_history"])

            # links extraction
            all_links, frames, window_loc, meta_refresh, n_lines_html, sm_meta_tags = parse_html_for_sm(
                html, url)

            # Redirections
            is_redirected, redirection_type, target_link = identifiy_redirection(url, url_histo,
                                                                                 histo,
                                                                                 frames,
                                                                                 window_loc, meta_refresh,
                                                                                 n_lines_html)

            if (source == "original") and (redirection_type in ["window_loc", "iframe", "meta_refresh"]):
                resu = {"source": "target", "url": target_link, "original_url": url}
            else:
                # REDIRECTION
                # redirection to different domain
                is_redirected_different_domain = False
                if (source == "target") or (redirection_type == "http"):
                    if redirection_type == "http":
                        original_domain = url
                        target_domain = link_to_domain(str(url_histo).split("___")[0])
                    else:
                        original_domain = link_to_domain(original_url)
                        target_domain = link_to_domain(url)
                    if target_domain != original_domain:
                        is_redirected_different_domain = True

                # Social Media direct search
                sm_features = get_sm_features(all_links)

                # sm extraction
                sm_outputs = format_social_medias(sm_features, url)

                # final
                resu = {"url": url, "original_url": original_url, "is_redirected": is_redirected,
                        "redirection_type": redirection_type, "source": source,
                        "is_redirected_different_domain": is_redirected_different_domain, "target_url": target_link,
                        }

                # feats
                # resu.update(sm_features)
                resu.update(sm_outputs)
                resu.update(sm_meta_tags)

        else:
            resu = {"url": url, "source": source, "original_url": original_url,  "is_redirected": False}

    except Exception as e:
        PLOG.it("ERROR during social media identification --> {} not classified : type:{} message:{}".format(feats["url"],
                                                                                                      type(e),
                                                                                                      str(e)))
        resu = { "url": url, "source": source, "original_url": original_url, "is_redirected": False}

    return resu


if __name__ == '__main__':
    pass
