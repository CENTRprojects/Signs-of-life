"""Main script to run the full Sign Of Life crawler
Set the configuration parameters in config.py before running it
"""
import re
import pandas as pd
import numpy as np
import os
from os.path import join
import time
import sys

from config import RUN_CONFIG, MAIN_DIR
from utils import remove_url_saved, PerformanceLogger
import output_processing as op

# steps
from url_visitor import FileData, interpret_boolean                     # DO_REQUESTS
from page_processing import get_page_displayed_text, get_page_language  # DO_PAGE_PROCESSING
from classification_parked import predict_parking                       # DO_CONTENT_CLASSIFICATION
from social_media_extraction import identify_social_media               # DO_SOCIAL_MEDIA
from headers_cookies_js import analyse_headers_cookies_javascript       # DO_HCJ_EXTRACTION
from mail_exchange import identify_mx_records_all_domains               # DO_MAIL_EXCHANGE
from formatting import final_formatting                                 # DO_CONCAT_FORMAT

# logging
import logging
loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)
plog = PerformanceLogger(filename="main_perf.log", enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])

EXPECTED_ORDER = ["url", "ind_non_schema", "pred_is_parked", "is_error", "comment", "is_redirected",
                  "redirection_type", "is_redirected_different_domain",
                  "flag_park", "flag_iframe", "flag_js_found",
                  "URL_history", "all_status_codes", "history", "registrar_found", "park_service", "kw_parked",
                  "kw_park_notice", "flag_js", "pred_is_empty", "n_letter", "n_words",
                  "ssl_enabled", "language", "cookies", "headers", "links",
                  "has_facebook", "has_twitter", "has_linkedin", "has_reddit", "has_instagram", "has_github",
                  "feat_fb_lks", "feat_tw_lks", "feat_lk_lks", "feat_rd_lks", "feat_ig_lks", "feat_gh_lks",
                  "has_open_graph", "has_twitter_card", "has_schema_tag", "to_sample", "ss_filename", "raw_filename",
                  "clean_filename", "json_filename"
                  ]


class Controller:
    """
    Main class for running each STEPS of the crawler
    """

    def __init__(self, file):
        """
        :param file: each csv filepath in the input directory
        :param config: run configuration dictionnary
        """
        self.file = file
        plog.it(f"Controller file set to {self.file}")
        self.file_data = FileData(file)
        self.load_csv()

    def load_json(self):
        """Load all current documents from JSON"""
        self.file_data.to_json()
        return True

    def save_json(self):
        self.file_data.to_pickle()

    def save_csv(self, path_csv):
        """Save current crawler results in CSV"""
        self.file_data.to_csv(path_csv)

    def load_csv(self):
        """Load current crawler results from CSV"""
        filepath = join(RUN_CONFIG['input_folder'], self.file)
        plog.it(f"Loading {filepath}")
        self.file_data.init_io_files(filepath)
        plog.it(f"{filepath} Initialized - {len(self.file_data.initial_table)} URLs loaded")

    def launch_requests_engine(self):
        """Main function for Python visit step"""
        self.file_data.file_full_request()
        return True

    def html_to_clean_text(self):
        """ Extract displayed text from each html pages"""
        self.file_data.documents = get_page_displayed_text(self.file_data.documents)

    def detect_language(self):
        """ Detect the main language of the page"""
        self.file_data.documents = get_page_language(self.file_data.documents)

    # ---------------------------------------------------------------------------------------------
    # ------------------------------------------MODELS METHODS-------------------------------------
    # ---------------------------------------------------------------------------------------------
    def predict_is_parked(self):
        """ Predict parking category of all the url of the current documents"""
        plog.it("loading documents to predict_parking")
        df_pred = predict_parking(self.file_data.documents)
        plog.it("creating output table")
        out_df = self.file_data.output_table
        plog.it("merging output table")

        out_df = pd.merge(out_df, df_pred, on="url", how="left")

        self.file_data.output_table = out_df
        plog.it(f"returning df_pred: {df_pred}")
        return df_pred

    def identify_social_media_activity(self):
        """ extract social medias data from all the url of the current documents"""
        df_media = identify_social_media(self.file_data.documents)
        col_to_drop = ['original_url', 'is_redirected', 'redirection_type', 'source', 'is_redirected_different_domain',
                       'target_url']
        for colo in col_to_drop:
            if colo in df_media.columns:
                df_media = df_media.drop([colo], axis=1)

        # filter existing column (common to parked)
        out_df = self.file_data.output_table
        exist_col = set(list(out_df.columns))

        dropped_col = [e for e in list(df_media.columns) if (e in exist_col) and (e != "url")]
        if dropped_col:
            print("INFO: in identify_social_media_activity: columns have been dropped: {}".format(dropped_col))
        df_media = df_media.drop(dropped_col, axis=1)
        self.file_data.output_table = pd.merge(out_df, df_media, on="url", how="left")
        return df_media

    def ctl_analyse_hcj(self):
        """ extract HCJ features from all the url of the current documents
        HCJ = Headers Cookies Javascript
        """
        df_headers = analyse_headers_cookies_javascript(self.file_data.documents)
        out_df = self.file_data.output_table
        # filter existing column (common to parked)
        exist_col = set(list(out_df.columns))
        dropped_col = [e for e in list(df_headers.columns) if (e in exist_col) and (e != "url")]
        if dropped_col:
            print("INFO: in analyse_headers: columns have been dropped: {}".format(dropped_col))
        df_headers = df_headers.drop(dropped_col, axis=1)
        self.file_data.output_table = pd.merge(out_df, df_headers, on="url", how="left")
        return df_headers

    def collect_mx_records(self):
        """Collect MX records from all the url of the current documents"""
        df_mx = identify_mx_records_all_domains(self.file_data.documents)
        out_df = self.file_data.output_table
        self.file_data.output_table = pd.merge(out_df, df_mx, on="url", how="left")
        return df_mx


def printAllDone():
    plog.it("-" * 64 + "\n" + "-" * 64 + "\n" + "-" * 24 + "ALL DONE" + "-" * 32 + "\n" + "-" * 64 + "\n" + "-" * 64)


def OutputProcessing(filename=None):
    """ OutputProcessing takes any csvs produced by this program and inserts them into the database defined in RUN_CONFIG[USE_DB] """
    if RUN_CONFIG["USE_DB"] == True:
        plog.it("---------------------USING DATABASE------------------------")
        op.DoIt(filename)
    else:
        plog.it("---------------------NO DATABASE CONFIGURED------------------------")
        plog.it("------------------RESULTS LEFT IN OUTPUT FOLDER--------------------")


def main():
    """
    Main function for the full crawler run (parking + social media + mails + Headers/Cookies/JS)
    :param config: run configuration dictionnary
    :return: save result
    """
    plog.perf_go("Starting Crawler")  # profiles code until perf_end
    print("Version: {}".format(RUN_CONFIG["version"]))
    super_raw_files = os.listdir(RUN_CONFIG["input_folder"])
    super_raw_files = sorted(super_raw_files, reverse=False)

    if len(super_raw_files) == 0:
        print("NO FILES FOUND AT GIVEN INPUT FOLDER: RUN ABORTED")
        return None

    # clean up
    plog.it("cleaning up pickles")
    # delete url pickles
    remove_url_saved(RUN_CONFIG["PATH_URL_SAVE"])
    plog.perf_lap('Deleted Pickles')

    plog.it("processing files into chunks")
    # remove old chunks
    for f in super_raw_files:
        if ".chunk." in f:
            f_path = join(RUN_CONFIG["input_folder"], f)
            os.remove(f_path)
    # get folder listing of proper csvs
    raw_files = sorted(os.listdir(RUN_CONFIG["input_folder"]), reverse=False)
    raw_files = [f for f in raw_files if (re.search("chunk.\d+.csv$", f) is None)]
    plog.it("N_files : {} \tfiles :{}".format(len(raw_files), raw_files))

    # File check - preprocess all the files to make sure we aren't going to break ourselves. 
    c_limit = RUN_CONFIG["CONTROLLER_LIMIT"]  # max number of domains the controller can handle at one time
    files = []
    for raw_file in raw_files:
        raw_path = join(RUN_CONFIG["input_folder"], raw_file)
        plog.it("PREPROCESSING {}".format(raw_path))

        df_domain = pd.read_csv(raw_path, encoding="utf-8")
        chunk = 0
        # special brakes for .ee rate limit test
        if ".ee." in raw_file:
            c_limit = 50
        else:
            c_limit = RUN_CONFIG["CONTROLLER_LIMIT"]  # max number of domains the controller can handle at one time
        while chunk * c_limit < len(df_domain):
            filename = f"{raw_file}.chunk.{chunk}.csv"
            filepath = join(RUN_CONFIG["input_folder"], filename)
            df_domain.iloc[chunk * c_limit:(chunk + 1) * c_limit].to_csv(filepath, index=False)
            files.append(filename)
            chunk += 1

    plog.it(
        f"PREPROCESSING COMPLETE - {len(raw_files)} file(s) broken down into {len(files)} chunks with up to {c_limit} urls each.")

    for f in files:
        input_path = join(RUN_CONFIG["input_folder"], f)
        host_name = str(os.getenv("HOSTNAME")) if os.getenv("HOSTNAME") is not None else ""
        output_path = join(MAIN_DIR, "output", host_name + "_res_" + f)
        RUN_CONFIG['output_final_file_path'] = join(MAIN_DIR, "output", host_name + "_final_" + f)
        plog.perf_lap("Starting on new file: " + f)
        plog.it("\ninput_file: {}\noutput_file: {}".format(input_path, output_path))

        # Main class
        ctl = Controller(f)
        if RUN_CONFIG["DO_REQUESTS"]:
            print("------------------REQUESTING------------------------")

            tic_req = time.time()
            cnt_while = 0
            are_requests_done = False
            while True:
                cnt_while += 1
                # try:
                are_requests_done = ctl.launch_requests_engine()

                # if any website needs a new visit, a second round is launched
                for doc in ctl.file_data.documents:
                    if doc.to_revisit:
                        are_requests_done = False
                        break

                if cnt_while > RUN_CONFIG["MAX_REVISIT"]:
                    are_requests_done = True

                # except Exception as e:
                #     plog.it("Error in loop: {} - {}".format(type(e), str(e)))
                #     plog.it("\t while")
                #     time.sleep(3)

                if are_requests_done:
                    break

            plog.it("Total request time in seconds : {}".format(time.time() - tic_req))
            plog.it("Number of revisit loop{}".format(cnt_while))

            if RUN_CONFIG["DEBUG_MODE"]:
                ctl.save_csv(output_path)
                ctl.save_json()

        else:
            print("INFO: ------------------REQUESTS DISABLED IN CONFIG------------------------")
        plog.perf_lap("Requests Complete")

        # post-processing
        plog.it("Starting Post Processing")

        if RUN_CONFIG["DO_CLASSIFICATION"] or RUN_CONFIG["DO_PAGE_PROCESSING"]:
            plog.it("POSTPROCESSING {}".format(f))

            if not RUN_CONFIG["DO_REQUESTS"]:
                try:
                    ctl.load_json()
                except Exception as e:
                    plog.it("ERROR while loading {}\t type:{}\t{} --> IGNORED".format(f, type(e), str(e)),
                            is_error=True)
                    continue
                plog.it("loaded pickles")

            if RUN_CONFIG["DO_PAGE_PROCESSING"]:
                print("{} -----PAGE PROCESSING------------------------".format(f))
                plog.it(
                    f"Starting Page Processing on {len(ctl.file_data.documents)} from {ctl.file_data.documents[0]} to {ctl.file_data.documents[-1]}")
                # html to clean text
                plog.it("html to text")
                ctl.html_to_clean_text()
                plog.it('html to clean text complete')

                # language detection
                plog.it("detect language")
                ctl.detect_language()
                plog.it('language detection complete')

            else:
                print("-------------------- PAGE PROCESSING DISABLED --------------------")
            plog.perf_lap("Page Processing Complete")

            if RUN_CONFIG["DEBUG_MODE"]:
                ctl.save_csv(output_path)
                ctl.save_json()
            plog.it("Starting Classification")

            if RUN_CONFIG["DO_CLASSIFICATION"]:
                print("{} -----CLASSIFICATION------------------------".format(f))
                if RUN_CONFIG["DO_CONTENT_CLASSIFICATION"]:
                    try:
                        print("prediction parking")
                        a = time.time()
                        _ = ctl.predict_is_parked()
                        print("Parking prediction time : {}".format(time.time() - a))
                    except Exception as e:
                        print("CRITICAL ERROR: parking prediction failed: {}\t{}".format(type(e), str(e)))
                        plog.it("CRITICAL ERROR: parking prediction failed: {}\t{}".format(type(e), str(e)),
                                is_error=True)
                        print(f"")
                        raise

                if RUN_CONFIG["DO_SOCIAL_MEDIA"]:
                    try:
                        print("Social Media Identification")
                        a = time.time()
                        _ = ctl.identify_social_media_activity()
                        print("Social Media Identification time : {}".format(time.time() - a))
                    except Exception as e:
                        print("CRITICAL ERROR: Social Media Identification failed: {}\t{}".format(type(e), str(e)))
                        plog.it("CRITICAL ERROR: Social Media Identification failed: {}\t{}".format(type(e), str(e)),
                                is_error=True)
                        raise

                if RUN_CONFIG["DO_MAIL_EXCHANGE"]:
                    try:
                        print("Mail Exchange Extraction")
                        a = time.time()
                        _ = ctl.collect_mx_records()
                        print("Mail Exchange Extraction time : {}".format(time.time() - a))
                    except Exception as e:
                        print("CRITICAL ERROR: Mail Exchange Extraction failed: {}\t{}".format(type(e), str(e)))
                        plog.it("CRITICAL ERROR: Mail Exchange Extraction failed: {}\t{}".format(type(e), str(e)),
                                is_error=True)
                        raise

                if RUN_CONFIG["DO_HCJ_EXTRACTION"]:
                    try:
                        print("Headers/Cookies/Javascript analysis")
                        a = time.time()
                        _ = ctl.ctl_analyse_hcj()
                        print("Headers/Cookies/Javascript analysis time : {}".format(time.time() - a))
                    except Exception as e:
                        print("CRITICAL ERROR: Headers/Cookies/Javascript analysis failed: {}\t{}".format(type(e),
                                                                                                          str(e)))
                        plog.it("CRITICAL ERROR: Headers/Cookies/Javascript analysis failed: {}\t{}".format(type(e),
                                                                                                            str(e)),
                                is_error=True)
                        raise

                plog.it("Saving classification output")
                ctl.save_csv(output_path)
                plog.it("file {} COMPLETED\n\n\n".format(f))
            else:
                print("-------------------- CLASSIFICATION DISABLED --------------------")
            plog.perf_lap("Classification Complete")

        plog.it("cleaning up pickles")
        # delete url pickles
        remove_url_saved(RUN_CONFIG["PATH_URL_SAVE"])
        plog.perf_lap('Deleted Pickles')

        # concat results
        plog.it("Starting Concatenation")
        if RUN_CONFIG["DO_CONCAT_FORMAT"]:
            print("------------------CONCATENATING------------------------")
            df_final = None
            # cnt_concat = 0
            host_name = str(os.getenv("HOSTNAME")) if os.getenv("HOSTNAME") is not None else ""
            output_path = join(MAIN_DIR, "output", host_name + "_res_" + f)
            plog.it(f"Saving data to {output_path}")
            df_tempo = pd.read_csv(output_path, encoding="utf-8-sig")
            df_tempo = interpret_boolean(df_tempo)

            if df_final is None:
                df_final = df_tempo.copy(deep=True)
            else:
                df_final = pd.concat([df_final, df_tempo], axis=0, sort=False)

            # Column formatting
            plog.it("Applying column formatting")
            # Column formatting
            expected_order = EXPECTED_ORDER
            for colo in expected_order:
                if colo not in df_final:
                    df_final[colo] = np.nan
            all_col = list(df_final.columns)
            other_col = [e for e in all_col if e not in expected_order]
            df_final = df_final[expected_order + other_col]
            plog.it("Column formatting complete")

            if RUN_CONFIG["DEBUG_MODE"]:
                debug_filepath = RUN_CONFIG["output_final_file_path"][0:-4] + "_tempo.csv"
                plog.it(f"Saving debug data to {debug_filepath}")
                df_final.to_csv(debug_filepath, encoding="utf-8-sig", index=False)

            print("---------------------FORMATTING------------------------")
            final_formatting(df_final, RUN_CONFIG["output_final_file_path"])
            plog.it(f"PARKING and SOCIAL MEDIA scopes saved to {RUN_CONFIG['output_final_file_path']}")

        else:
            print("------------------- CONCATENATION & FORMATTING DISABLED --------------------")

        plog.perf_lap('Concatenation Complete')
        plog.it(f'sending {RUN_CONFIG["output_final_file_path"]} to output processing')
        OutputProcessing(RUN_CONFIG["output_final_file_path"])
        plog.perf_lap('Output Processing Complete')

    plog.perf_end("Crawler Finished")


def printAllDone():
    plog.it("-" * 64 + "\n" + "-" * 64 + "\n" + "-" * 24 + "ALL DONE" + "-" * 32 + "\n" + "-" * 64 + "\n" + "-" * 64)


def OutputProcessing(filename=None):
    """ OutputProcessing takes any csvs produced by this program and inserts them into the database defined in RUN_CONFIG[USE_DB] """
    if RUN_CONFIG["USE_DB"] == True:
        plog.it("---------------------USING DATABASE------------------------")
        op.DoIt(filename)
    else:
        plog.it("---------------------NO DATABASE CONFIGURED------------------------")
        plog.it("------------------RESULTS LEFT IN OUTPUT FOLDER--------------------")


def get_container_name():
    container_name = ""
    for arg in sys.argv:
        if arg.startswith('-container_name='):
            container_name = "/" + arg.split('=')[1]
            break
    return container_name


if __name__ == '__main__':
    RUN_CONFIG['input_folder'] = RUN_CONFIG['input_folder'] + get_container_name()

    tic_tic = time.time()
    main()
    tic_tic2 = time.time()
    printAllDone()
    plog.it("Total time in seconds (main): {}".format(tic_tic2 - tic_tic))
