"""Script handling Python and Chrome visit of a domain + Visit of redirected pages orchestration"""

import re
import socket
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from os.path import join
from pathlib import Path
import time
import random
from random import shuffle
from concurrent import futures
from datetime import datetime
import asyncio
import requests
import aiohttp
from aiohttp import ClientSession
import asyncpool
from selenium import webdriver
import tracemalloc

from config import RUN_CONFIG
from ref_headers_cookies_js import SET_ALL_CONSIDERED_HDS, SET_ALL_CONSIDERED_CKS
from utils import save_obj, load_obj, gather_url_saved, remove_url_saved, convert_idna, is_str_full_digit, \
    normalize_hcj_value
from status_codes import HTTP_STATUS
from joblib import Parallel, delayed
from utils import PerformanceLogger

# import psutil
PREF_HDRS = "fx_hdr_"
PREF_COOK = "fx_cks_"
PREF_JS = "fx_jsc_"
CL_HDRS = "headers"
CL_COOKS = "cookies"
SUSPICIOUS_COLS = ["kw_park_notice"]  # kw_park_notice may contain other languages

# Attempt UVLOOP setup
try:
    if RUN_CONFIG["USE_UVLOOP"]:
        import uvloop  # available on Linux only

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("uvloop activated")
except:
    pass

# logging
import logging
loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)

req_plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"], "logging"), filename="req_p.log",
                             enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])
main_plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"], "logging"), filename="main_p.log",
                              enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])
sam_plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"], "logging"), filename="sam_p.log",
                             enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])
tracemalloc.start()


class RateLimiter:
    def __init__(self):
        self.started = 0
        self.completed = 0
        if RUN_CONFIG['PARALLEL_PREFER'] == 'threads':
            request_period = 1 / RUN_CONFIG[
                'REQUEST_RATE_LIMIT']  # time between each request to hit the request_rate_limit
        else:
            request_period = RUN_CONFIG['MAX_PROCESSES'] / RUN_CONFIG[
                'REQUEST_RATE_LIMIT']  # time between each request to hit the request_rate_limit

        # RUN_CONFIG['MAX_PROCESSES']
        self.request_time_block = request_period
        self.next_tick = int(time.perf_counter())  # start on a nice round number

    def __call__(self):
        self.started += 1
        time_now = time.perf_counter()
        if self.next_tick < time_now:
            self.next_tick = int(
                time_now) + 1 + self.request_time_block  # claim this tick in case someone else comes along while I'm waiting
        else:
            self.next_tick += self.request_time_block
        return self.started, self.next_tick  # send back an id, and the green light time

    def im_done(self):
        self.completed += 1
        return self.status()

    def status(self):
        # return f"{self.completed}/{self.started} => {self.started-self.completed}"
        return "{}/{}({}%2f)".format(self.completed, self.started, 100 * self.completed / self.started)

    def reset(self):
        self.__init__()


rate_limiter = RateLimiter()

log_manager = logging.getLogger()
# logging.FileHandler('logger.log')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
log_manager.addHandler(handler)
log_manager.setLevel(logging.WARNING)

USER_AGENT = RUN_CONFIG["USER_AGENT"]
LOADING_TIME = 15  # 15 seconds to interpret the full page

DEFAULT_BOOLEAN_VALUES = {
    "to_sample": False,
    # park
    "pred_is_parked": False,
    "ind_non_schema": True,
    "is_error": False,
    "is_redirected": False,
    "is_redirected_different_domain": False,
    "registrar_found": False,
    "kw_parked": False,
    "pred_ml_park": False,
    "pred_is_empty": False,
    "is_full_js_parked": False,
    "flag_js_found": False,

    # sm
    "has_facebook": False, "has_twitter": False,
    "has_linkedin": False, "has_reddit": False, "has_instagram": False, "has_github": False,
    "has_open_graph": False, "has_twitter_card": False, "has_schema_tag": False,

    # mail
    "has_mx_record": False,
    "has_own_mx_record": False,

}
THRESH_TOO_LONG = 10000000  # discarded too long pages (considered as error)
RE_HTML = re.compile("^[ \n\r\t]*<[ \n\r\t]*html", flags=re.IGNORECASE)
RE_HTMLDOCT = re.compile("^[ \n\r\t]*<[ \n\r\t]*!doctype", flags=re.IGNORECASE)


def fill_default_values(df):
    """Fill NaN values of dataframe df with default values in DEFAULT_BOOLEAN_VALUES"""
    df = df.copy(deep=True)

    for colo in DEFAULT_BOOLEAN_VALUES.keys():
        if colo in list(df.columns):
            df[colo] = df[colo].fillna(value=DEFAULT_BOOLEAN_VALUES[colo])

    # ml key word features
    for colo in list(df.columns):
        if colo.startswith("ml_feat_"):
            df[colo] = df[colo].fillna(value=0)
    return df


def interpret_boolean(df):
    """Normalize boolean values"""
    df = df.copy(deep=True)

    for colo in DEFAULT_BOOLEAN_VALUES.keys():
        if colo in list(df.columns):
            ind_true = df[colo].isin([True, "True", "TRUE"])
            ind_false = df[colo].isin([False, "False", "FALSE"])
            ind_nul = df[colo].isnull()
            df.loc[ind_true, colo] = True
            df.loc[ind_false, colo] = False
            df.loc[ind_nul, colo] = DEFAULT_BOOLEAN_VALUES[colo]
            df[colo] = df[colo].astype(bool)
        else:
            print("WARNING no column {} --> filled with default value".format(colo))
            df[colo] = DEFAULT_BOOLEAN_VALUES[colo]

    # ml key word features
    for colo in list(df.columns):
        if colo.startswith("ml_feat_"):
            ind_full_digit = df[colo].apply(is_str_full_digit)
            df.loc[ind_full_digit, colo] = df.loc[ind_full_digit, colo].apply(lambda x: int(x))
            df[colo] = df[colo].fillna(value=0)

    return df


class Website:
    def __init__(self, url):
        self.url = url

        # General metrics
        self.count_words = None
        self.comment = None
        self.is_error = None
        self.to_revisit = False
        # self.special_revisit = None
        self.language = None
        self.raw_text = None
        self.clean_text = None
        self.flag_css = None
        self.flag_js = None
        self.other_variables = dict()
        self.other_variables['to_sample'] = False


def website_to_dico(website):
    return vars(website)


def dico_to_website(dico):
    """Dictiory of a web page to Website object"""
    wb = Website(dico["url"])
    for var in list(vars(wb).keys()):
        if var in dico.keys():
            vars(wb)[var] = dico[var]
    return wb


def correct_encoding(x):
    """Correct non-utf-8 encoding"""
    if x is None:
        return x
    elif len(str(x)) == 0:
        return x
    else:
        return str(x).encode("utf-8", errors="ignore").decode("utf-8")


def split_by_batches(list_dict_unique, n_batch_split, n_to_visit_round):
    """Split list of unique URLs into batches"""
    batch_size = int(np.ceil(n_to_visit_round / n_batch_split))
    url_batches = []
    for i in range(0, n_batch_split):
        partial_dict_unique = list_dict_unique[i * batch_size:(i + 1) * batch_size]
        url_batches.append(partial_dict_unique)
    url_batches = [e for e in url_batches if len(e) > 0]
    return url_batches


def track_performance(all_res, loop_duration, n_to_visit_round, fname):
    """ Print descriptive performance metrics for the request round"""
    cnt = 0
    for e in all_res:
        # if re.search("Timeout", str(e.comment)) or re.search("DNS resolution", str(e.comment)):
        if e.to_revisit == True:
            cnt += 1

    achieved = len(all_res)
    if achieved > 0:
        error_rate = int(round(100 * cnt / achieved, 1))
    else:
        error_rate = 100
    if loop_duration is not None:
        speed = int(round((n_to_visit_round / loop_duration) * 60, 0))
    else:
        speed = "NA"
    print("achieved: \t{}\terror rate : \t{}\tround_visit :\t{}\tavg_visit_per_min : \t{}".format(achieved,
                                                                                                  error_rate,
                                                                                                  n_to_visit_round,
                                                                                                  speed))


class FileData:
    """Track all data from one file chunk: initial URL + all intermediary results of each steps + crawler output"""
    def __init__(self, file):
        self.file = file
        self.initial_table = None
        self.documents = []
        self.output_table = None
        # pass

    def init_io_files(self, path_csv):
        """ Read file containing unique and valid urls to visit"""
        req_plog.perf_go(f"Initializing {path_csv}")

        df = pd.read_csv(path_csv, encoding="utf-8")

        req_plog.it(f"dataframe(df) : {df}")
        # Remove point and ignonre wrong urls
        df["url"] = df["url"].astype(str)
        df["url"] = df["url"].apply(lambda x: x[0:-1] if x.endswith(".") else x)
        df["url"] = df["url"].apply(convert_idna)
        ind_valid_schema = df["url"].apply(lambda x: (re.search("\.", x) is not None))

        df["ind_non_schema"] = True
        df.loc[~ind_valid_schema, "ind_non_schema"] = False

        self.initial_table = df
        self.output_table = df

    def file_full_request(self):
        """
        Launch all requests of a file's domains by batches
        :param force_new: If True, any website already visited will be overwritten
        :return: save HTML pages in self.documents
        """
        # All urls
        df = self.initial_table
        ind_valid_schema = df["ind_non_schema"] != False
        df_unique = df[ind_valid_schema].drop_duplicates(subset=["url"]).dropna(subset=["url"])
        list_dict_unique = df_unique.reindex(np.random.permutation(df_unique.index)).to_dict(orient="records")

        all_res = []
        if not RUN_CONFIG["force_new_visit"]:
            # Reuse urls already done in a former run at file level
            l_1 = len(list_dict_unique)
            if os.path.isfile(join(RUN_CONFIG["PATH_DOC_SAVE"], "obj_documents_{}.json".format(self.file))):
                self.to_json()
                url_alr = dict()
                for e in self.documents:
                    if not e.to_revisit:
                        url_alr[e.url] = 1
                        all_res.append(e)
                list_dict_unique = [e for e in list_dict_unique if (e["url"] not in url_alr.keys())]
            l_2 = len(list_dict_unique)

            # Reuse urls already done in a former run at  URL level
            list_existing_url = [dico_to_website(d) for d in gather_url_saved(RUN_CONFIG["PATH_URL_SAVE"])]
            if len(list_existing_url) > 0:
                url_alr = dict()
                for e in list_existing_url:
                    if not e.to_revisit:
                        url_alr[e.url] = 1
                list_dict_unique = [e for e in list_dict_unique if (e["url"] not in url_alr.keys())]
            l_3 = len(list_dict_unique)

            print("Total urls in file : {}".format(l_1))
            print("Urls already in document pickles : {}".format(l_1 - l_2))
            print("Urls already in url pickles : {}".format(l_2 - l_3))

        shuffle(list_dict_unique)
        n_to_visit_round = len(list_dict_unique)
        print("Total urls to visit : {}".format(n_to_visit_round))
        loop_duration = None
        if n_to_visit_round > 0:

            # split by batches
            url_batches = split_by_batches(list_dict_unique, RUN_CONFIG["BATCH_SPLIT"], n_to_visit_round)

            print("debug MAX_WORKERS : " + str(RUN_CONFIG["MAX_WORKERS"]))
            print("debug LIMIT_REQUEST : " + str(RUN_CONFIG["LIMIT_REQUEST"]))
            print("debug MINUTES_TO_TIMEOUT : " + str(RUN_CONFIG["MINUTES_TO_TIMEOUT"]))

            # Multi process/Mono process asynchronous
            tt = time.time()
            if RUN_CONFIG["MULTI_PROCESSING"]:
                Parallel(n_jobs=RUN_CONFIG["MAX_PROCESSES"])(
                    delayed(async_url_batch_visit)(batch, do_sampling=True) for batch in url_batches)
            else:
                async_url_batch_visit(list_dict_unique, do_sampling=True)

            loop_duration = time.time() - tt
            print("--------------------outside async/parallel ZONE--------------------")

        # gather url files
        list_existing_url = [dico_to_website(d) for d in gather_url_saved(RUN_CONFIG["PATH_URL_SAVE"])]
        all_res = all_res + list_existing_url

        # Performance track
        track_performance(all_res, loop_duration, n_to_visit_round, self.file)

        self.documents = all_res

    def to_csv(self, path_csv):
        """ Save output csv file"""
        # generic metrics
        list_res = []
        for doc in self.documents:
            dict_to_save = {"url": doc.url,
                            "comment": doc.comment,
                            "language": doc.language,
                            "is_error": doc.is_error
                            }
            for addit, value in doc.other_variables.items():
                if RUN_CONFIG["DO_HCJ_EXTRACTION"] and addit in [CL_HDRS, CL_COOKS]:
                    # for hdr, hdr_val in value.items():
                    #     dict_to_save[hdr] =  hdr_val
                    pass

                elif (not addit.startswith("cookie")) and (not addit.startswith("Header")) and (addit != "non_text"):
                    dict_to_save[addit] = value

            list_res.append(dict_to_save)

        df_gen = pd.DataFrame(list_res)

        # handle case redirection to an error:
        redirection_to_error = False
        if "is_error" in list(self.output_table.columns):
            redirection_to_error = True
        if redirection_to_error:
            self.output_table = self.output_table.rename(columns={"is_error": "is_error_2", "comment": "comment_2"})

        # merge with initial table
        df_gen_col = list(df_gen.columns)
        addit_col = [e for e in list(self.output_table.columns) if e != "url"]
        for colo in addit_col:
            if colo in df_gen_col:
                self.output_table = self.output_table.drop([colo], axis=1)

        self.output_table = pd.merge(self.output_table, df_gen, how="left", on="url")

        if redirection_to_error:
            # replace error of source domain only when target domain has an error
            ind_error = self.output_table["is_error_2"].notnull()
            self.output_table.loc[ind_error, "is_error"] = self.output_table.loc[ind_error, "is_error_2"]
            self.output_table.loc[ind_error, "comment"] = self.output_table.loc[ind_error, "comment_2"]
            self.output_table = self.output_table.drop(["is_error_2", "comment_2"], axis=1)

        # CLean boolean columns
        self.output_table = fill_default_values(self.output_table)

        # save
        try:
            self.output_table.to_csv(path_csv, encoding="utf-8-sig", index=False)
        except:
            try:
                for colo in SUSPICIOUS_COLS:
                    self.output_table[colo] = self.output_table[colo].apply(correct_encoding)

                self.output_table.to_csv(path_csv, encoding="utf-8-sig", index=False)
            except:
                for colo in list(self.output_table.columns):
                    self.output_table[colo] = self.output_table[colo].apply(correct_encoding)

                self.output_table.to_csv(path_csv, encoding="utf-8-sig", index=False)

    def to_pickle(self):
        save_obj([website_to_dico(w) for w in self.documents], "documents_" + self.file, RUN_CONFIG["PATH_DOC_SAVE"])

        # delete url pickles
        remove_url_saved(RUN_CONFIG["PATH_URL_SAVE"])

    def to_json(self):
        self.documents = [dico_to_website(d) for d in load_obj("documents_" + self.file, RUN_CONFIG["PATH_DOC_SAVE"])]

    def cleanup(self):
        req_plog.perf_end("Finished")


def async_url_batch_visit(partial_dict_unique, do_sampling=False):
    """ Start Asynchronous loop"""
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main_async_batch_visit(partial_dict_unique, loop, do_sampling))
    loop.close()


async def main_async_batch_visit(dict_unique, loop, do_sampling):
    """ Launch pool of asynchronous requests"""
    connector = aiohttp.TCPConnector(ssl=False, loop=loop, resolver=aiohttp.AsyncResolver(loop=loop),
                                     family=socket.AF_INET, limit=RUN_CONFIG["LIMIT_REQUEST"],
                                     # force_close=True)
                                     keepalive_timeout=5)

    timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_read=None, sock_connect=None)

    # pid for logging
    pid = os.getpid()

    # start Session
    async with ClientSession(connector=connector, loop=loop, timeout=timeout, connector_owner=False,
                             headers={"User-Agent": USER_AGENT}) as session:
        # start Queue Pool: When a request is being processed by the remote server, the Python worker
        # is released to be able to work on another request until the answer is received
        async with asyncpool.AsyncPool(loop=loop,
                                       num_workers=RUN_CONFIG["MAX_WORKERS"],
                                       name="Pool_{}".format(str(pid)),
                                       logger=log_manager,
                                       worker_co=single_url_visit,
                                       # max_task_time=RUN_CONFIG["MINUTES_TO_TIMEOUT"] * 60,
                                       max_task_time=None,
                                       log_every_n=RUN_CONFIG["LOG_EVERY_N"],
                                       expected_total=len(dict_unique)) as pool:
            cnt = 0
            for doc in dict_unique:
                # print(f"Pushing {doc['url']}")
                cnt += 1
                await pool.push(doc, session, do_sampling)

    await asyncio.sleep(1)
    # closing independant connector
    await connector.close()


async def single_url_visit(domain, session, do_sampling):
    """
    Launch one HTTP request to one domain
    """
    url = domain["url"]
    if RUN_CONFIG["DEBUG_PRINT"]:
        print("Starting url: {}".format(url))
    ts = time.time()
    wb = Website(url)
    page = None
    enable_revisiting = RUN_CONFIG["ENABLE_REVISITING"]

    # select for sampling
    if do_sampling:  # explicit entry so we only do this once per run
        if random.random() < RUN_CONFIG['SAMPLING_RATE']:
            wb.other_variables["to_sample"] = True
            sam_plog.it(f"Selected for sampling : {domain['url']}")

    # hash naming by original url (for redirection targets)
    if ("original_url" in domain.keys()) and (domain["original_url"] is not None):
        save_path = RUN_CONFIG["PATH_URL_REVISIT_SAVE"]
        ref_url = domain["original_url"]
        wb.other_variables["original_url"] = ref_url
    else:
        save_path = RUN_CONFIG["PATH_URL_SAVE"]
        ref_url = url
    url_result_name = convert_idna(ref_url)

    # request URL page
    try:
        full_url = url
        if re.search("^https*:", full_url, re.IGNORECASE) is None:
            full_url = "http://" + url

        max_range_bytes = RUN_CONFIG["MAX_MB_SINGLE_URL"] * 1024 * 1024 - 10
        async with session.get(full_url, timeout=RUN_CONFIG["MINUTES_TO_TIMEOUT"] * 60) as response:
            # OBTAIN HEADER CONTENT-LENGTH TO CHECK SIZE BEFORE RETRIEVING ENTIRE OBJECT
            content_length = response.headers.get('Content-Length')

            if RUN_CONFIG["DO_HCJ_EXTRACTION"]:  # get cookies & headers
                dico_hd = {}
                for k, v in dict(response.headers).items():
                    if normalize_hcj_value(k) in SET_ALL_CONSIDERED_HDS:
                        dico_hd[PREF_HDRS + k] = 1
                wb.other_variables[CL_HDRS] = dico_hd

                dico_cks = {}
                for k, v in dict(response.cookies).items():
                    if normalize_hcj_value(k) in SET_ALL_CONSIDERED_CKS:
                        dico_cks[PREF_COOK + k] = 1
                wb.other_variables[CL_COOKS] = dico_cks

            # IF THERE IS CONTENT-LENGTH IN HEADER AND IT IS LESS THAN 5MB
            if content_length is not None and int(content_length) <= max_range_bytes:

                page = dict()
                page["text"] = await response.text(errors="ignore")
                page["url"] = response.url
                page["history"] = response.history
                page["status"] = response.status

            # IF THERE IS NO CONTENT-LENGTH IN HEADER
            elif content_length is None:
                total_bytes = 0
                chunks = []

                # ITERATE PROGRESSIVELY THE RESPONSE CONTENT
                async for chunk in response.content.iter_any():

                    if chunk:
                        total_bytes += len(chunk)

                        # CHECK IF LENGTH IS LARGER THAN 5MB
                        if total_bytes > max_range_bytes:
                            # SET VALUES TO IGNORE THE DOMAIN
                            wb.is_error = True
                            wb.comment = "Page size too large error"
                            wb.to_revisit = False
                            break
                        chunks.append(chunk)

                # CHECK IF RESPONSE IN LESS THAN MAX_MB TO LATER PROCESS THE DOMAIN PAGE
                if total_bytes < max_range_bytes:
                    async with session.get(full_url, timeout=RUN_CONFIG["MINUTES_TO_TIMEOUT"] * 60) as response:
                        page = dict()
                        page["text"] = b"".join(chunks).decode(errors="ignore")
                        page["url"] = response.url
                        page["history"] = response.history
                        page["status"] = response.status
                        # wb.is_error = False

    # Catching errors
    except aiohttp.client.ClientConnectorError as e:
        if "Connect call failed" in str(e):
            wb.is_error = True
            wb.comment = "Connection error"
            wb.to_revisit = enable_revisiting
        else:
            wb.is_error = True
            wb.comment = "DNS resolution error"
            wb.to_revisit = enable_revisiting
    except futures.TimeoutError as e:
        # print("timeout {}".format(type(e)))
        wb.is_error = True
        wb.comment = "TimeoutError"
        wb.to_revisit = enable_revisiting
    except futures._base.TimeoutError as e:
        # print("timeout {}".format(type(e)))
        wb.is_error = True
        wb.comment = "TimeoutError"
        wb.to_revisit = enable_revisiting
    except aiohttp.client.ServerDisconnectedError:
        wb.is_error = True
        wb.comment = "ServerDisconnected"
        wb.to_revisit = enable_revisiting
    except aiohttp.client.ClientOSError as e:
        wb.is_error = True
        if ("WinError 10054" in str(e)) or ("WinError 10054" in str(e)):
            wb.comment = "Connection closed"
            wb.to_revisit = enable_revisiting
        elif "Connection reset by peer" in str(e):
            wb.is_error = True
            wb.comment = "Connection error"
            wb.to_revisit = False
        else:
            wb.comment = "error of type {} message: {}".format(type(e), str(e))
            wb.to_revisit = False
    except aiohttp.client.ClientConnectorCertificateError:
        wb.is_error = True
        wb.comment = "SSL error"
        wb.to_revisit = False
    except requests.exceptions.ConnectionError:
        wb.is_error = True
        wb.comment = "Connection error"
        wb.to_revisit = enable_revisiting
    except aiohttp.client.ClientConnectionError:
        wb.is_error = True
        wb.comment = "Connection error"
        wb.to_revisit = enable_revisiting
    except UnicodeDecodeError:
        wb.is_error = True
        wb.comment = "Decoding error"
        wb.to_revisit = False
    except aiohttp.client.ClientPayloadError:
        wb.is_error = True
        wb.comment = "Payload error"
        wb.to_revisit = False
    except Exception as e:
        wb.is_error = True
        wb.comment = "error of type {} message: {}".format(type(e), str(e))
        if "Session is closed" in str(e):
            wb.to_revisit = enable_revisiting
        else:
            wb.to_revisit = False

    if not wb.is_error:
        wb = complete_data_of_successful_requests(page, wb)
    if not wb.is_error:
        wb = handle_too_long_pages(wb)

    # print("url done: {}".format(url))
    # save file
    save_obj(website_to_dico(wb), "res_{}".format(url_result_name), save_path)
    if RUN_CONFIG["DEBUG_PRINT"]:
        print("Ending url: \t{} - {}".format(url, time.time() - ts))

def is_html_start(raw_text):
    """Check if website response looks like HTML page"""
    beg_text = raw_text[0:50]
    return (RE_HTML.search(beg_text) is not None) or (RE_HTMLDOCT.search(beg_text) is not None)


def handle_too_long_pages(wb):
    """
    Exclude non html pages that are too long
    """
    if wb.raw_text:
        if not is_html_start(wb.raw_text) and len(wb.raw_text) > THRESH_TOO_LONG:
            wb.is_error = True
            wb.raw_text = "Error: too long"
            wb.comment = "Error: too long"

    return wb


def complete_data_of_successful_requests(page, wb):
    """Get additional information on request success: status code and history of redirections"""
    if (page["status"] < 200) | (page["status"] >= 300):
        wb.is_error = True
        st_code = page["status"]
        if st_code in HTTP_STATUS.keys():
            detail = ": " + HTTP_STATUS[st_code]
        else:
            detail = ""
        wb.comment = "Error status code :" + str(st_code) + detail

        try:
            wb.other_variables["all_status_codes"] = "_".join(
                [str(page["status"])] + [str(e.status) for e in page["history"]])
            wb.other_variables["history"] = len(page["history"])
            wb.other_variables["URL_history"] = "___".join(
                [str(page["url"])] + [str(e.url) for e in page["history"]])

        except  Exception as e:
            print("error with other variables : {}".format(str(e)))
            wb.comment += ".Other variables not available --> IGNORING :{}".format(str(e))

        if page["text"] is not None:
            if len(page["text"]) > 0:
                wb.raw_text = page["text"]
    else:
        wb.raw_text = page["text"]

        try:
            wb.other_variables["all_status_codes"] = "_".join(
                [str(page["status"])] + [str(e.status) for e in page["history"]])
            wb.other_variables["history"] = len(page["history"])
            wb.other_variables["URL_history"] = "___".join(
                [str(page["url"])] + [str(e.url) for e in page["history"]])

        except Exception as e:
            print("error with other variables : {}".format(str(e)))
            wb.comment = "Other variables not available --> IGNORING :{}".format(str(e))
    return wb


def request_full_file_target_links(target_links_to_visit):
    """Split by batches and visit redirected pages"""
    # batches split
    batch_size = int(np.ceil(len(target_links_to_visit) / RUN_CONFIG["BATCH_SPLIT"]))
    url_batches = []
    for i in range(0, RUN_CONFIG["BATCH_SPLIT"]):
        url_batches.append(target_links_to_visit[i * batch_size:(i + 1) * batch_size])
    url_batches = [e for e in url_batches if len(e) > 0]

    print("-------visits of target links")
    remove_url_saved(RUN_CONFIG["PATH_URL_REVISIT_SAVE"])

    # todo: add NON-PARALLEL option
    Parallel(n_jobs=RUN_CONFIG["MAX_PROCESSES"])(delayed(async_url_batch_visit)(batch) for batch in url_batches)
    list_target_websites = [dico_to_website(d) for d in gather_url_saved(RUN_CONFIG["PATH_URL_REVISIT_SAVE"])]

    remove_url_saved(RUN_CONFIG["PATH_URL_REVISIT_SAVE"])

    return list_target_websites


def single_url_browser_visit(link, webdriver):
    """Visit one URL with Chrome webdriver"""
    # link to visit (original or redirection)
    if link["source"] == "target":
        url = link["target_url"]
        resu = Website(url)
        resu.other_variables["original_url"] = link["original_url"]
    else:
        url = link["url"]
        resu = Website(url)

    full_url = url
    if re.search("^https*:", full_url, re.IGNORECASE) is None:
        full_url = "http://" + url

    # loading timeout
    webdriver.implicitly_wait(LOADING_TIME)  # wait for full loading

    # visit url
    webdriver.get(full_url)

    # loading timeout
    webdriver.implicitly_wait(0)

    try:
        html = webdriver.execute_script("return document.documentElement.innerHTML")
        resu.is_error = False
    except Exception as e:
        html = f"JS CRAWLING ERROR: {e}"
        resu.is_error = True

    # results
    resu.comment = None
    resu.to_revisit = False
    resu.raw_text = html
    resu.other_variables["history"] = link["history"] if hasattr(link, "history") else 0
    resu.other_variables["URL_history"] = link["URL_history"] if hasattr(link, 'URL_history') else 0

    return resu


def request_full_file_with_browser(links):
    """Visit a list of URLs with Chrome webdriver"""
    # visiting links
    all_resu = []
    all_resu = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
        delayed(single_url_browser_load_visit)(link) for link in tqdm(links))

    all_resu = [e for e in all_resu if (e is not None)]

    return all_resu


'''
example resu/link going into single_url_browser_load_visit: link = {
    'park_service': None, 
    'registrar_found': False, 
    'kw_parked': False, 
    'kw_park_notice': '', 
    'is_redirected': False, 
    'redirection_type': '', 
    'flag_iframe': False,
	'flag_js_found': True,
	'n_letter': 199,
	'n_words': 34,
	'js_or_iframe_found': True,
	'pred_is_empty': False,
	'pred_is_parked': False,
	'source': 'original',
	'original_url': None,
	'url': 'mammachia.com.au',
	'document_write': False,
	'target_url': None,
	'is_redirected_to_registrar': False,
	'is_redirected_different_domain': False,
	'to_revisit_with_js': False,
	'is_non_text_park': False,
	'is_full_js_parked': False,
	'pred_ml_park': True,
	'to_sample': True,
	'ml_feat_sale': 0,
	'ml_feat_blocked': 0,
	'ml_feat_construction': 0,
	'ml_feat_expired': 0,
	'ml_feat_index_of': 0,
	'ml_feat_other': 1,
	'ml_feat_reserved': 0,
	'ml_feat_starter': 0,
	'history': 0,
	'URL_history': 'http://mammachia.com.au',
	'tag_quantity': 7,
	'ssl_enabled': False
}
'''


def clean_url(url):
    return re.sub('[/\:*?"<>|]', '_', url)


# store sample filenames as url-type-datetime.ext
def get_sample_filenames(url):
    """Get file names for the sample data associated to url"""
    cleaned_url = clean_url(url)
    date_str = datetime.now().strftime(
        "%Y-%m-%d")  # this will crash if called twice and the clock ticks over past midnight - html and ss are written in different steps, after some time related to the batch size.
    # assure samples directory exists
    try:
        Path(RUN_CONFIG["SAMPLING_LOCAL_FOLDER"]).mkdir(parents=True, exist_ok=False)
        if (RUN_CONFIG["DEBUG_PRINT"]):
            print(f'Sample output directory created {RUN_CONFIG["SAMPLING_LOCAL_FOLDER"]}')
    except FileExistsError:
        pass

    ss_filename = Path(RUN_CONFIG["SAMPLING_LOCAL_FOLDER"]).joinpath(
        Path(f"{cleaned_url}.ss.{date_str}.png")).relative_to(Path(RUN_CONFIG["MAIN_DIR"]))
    raw_filename = Path(RUN_CONFIG["SAMPLING_LOCAL_FOLDER"]).joinpath(
        Path(f"{cleaned_url}.raw.{date_str}.txt")).relative_to(Path(RUN_CONFIG["MAIN_DIR"]))
    clean_filename = Path(RUN_CONFIG["SAMPLING_LOCAL_FOLDER"]).joinpath(
        Path(f"{cleaned_url}.clean.{date_str}.txt")).relative_to(Path(RUN_CONFIG["MAIN_DIR"]))
    json_filename = Path(RUN_CONFIG["SAMPLING_LOCAL_FOLDER"]).joinpath(
        Path(f"{cleaned_url}.resu.{date_str}.json")).relative_to(Path(RUN_CONFIG["MAIN_DIR"]))
    # sam_plog.it(f"Providing filenames for {url} : {ss_filename} | {raw_filename} | {clean_filename} | {json_filename}")
    return (ss_filename.as_posix(), raw_filename.as_posix(), clean_filename.as_posix(), json_filename.as_posix())


def single_url_browser_load_visit(link):
    """Visit one url with Chrome webdriver"""
    try:
        webdriver = initiate_browser_driver()
        resu = single_url_browser_visit(link, webdriver)

        # screenshot save
        if RUN_CONFIG["DO_SAMPLING"] and link['to_sample']:
            sam_plog.it(f"({link['url']}) | PREPARING FOR SCREENSHOT : {link['ss_filename']}")
            '''
            # use a scrolling trick to make sure page has fully loaded before taking the screenshot (so we don't get an empty page)
            driver.execute_script("""
                (function () {
                    var y = 0;
                    var step = 100;
                    window.scroll(0, 0);

                    function f() {
                        if (y < document.body.scrollHeight) {
                            y += step;
                            window.scroll(0, y);
                            setTimeout(f, 100);
                        } else {
                            window.scroll(0, 0);
                            document.title += "scroll-done";
                        }
                    }

                    setTimeout(f, 1000);
                })();
            """)

            for i in range(30):
                if "scroll-done" in driver.title:
                    break
                time.sleep(1)
            '''
            for i in range(30):
                if webdriver.execute_script("""document.onreadystatechange = function () {
                                               if (document.readyState == "complete") {
                                                    return "complete";
                                                }
                                            }""") == 'complete':
                    break
                time.sleep(1)
            total_height = webdriver.execute_script("""
                if (document.scrollingElement){
                    return document.scrollingElement.scrollHeight;
                } 
                return document.body.offsetHeight;
            """)
            max_height = RUN_CONFIG["SAMPLING_MAX_SCREENSHOT_HEIGHT_PX"]
            webdriver.set_window_size(1200, total_height if total_height <= max_height else max_height)
            webdriver.save_screenshot(link['ss_filename'])
            sam_plog.it(f"({link['url']}) | SCREENSHOT SAVED")

        webdriver.quit()

    except Exception as e:
        print("JS error with {} of type: {} : {} --> js interpretation removed".format(link["url"], type(e), str(e)))
        # raise
        resu = None
        try:
            webdriver.quit()  # force close
        except:
            print("INFO: failed to close driver")
            pass

    return resu


def initiate_browser_driver():
    """Open a Browser session"""
    # options
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('headless')
    options.add_argument("disable-extensions")
    options.add_argument('silent')
    options.add_argument('window-size=1200x600')
    options.add_argument(f'user-agent={USER_AGENT}')
    # options.add_argument("--incognito")
    # launch Chrome
    driver = webdriver.Chrome(chrome_options=options)
    # loading timeout
    driver.implicitly_wait(LOADING_TIME)  # wait for full loading
    return driver
