"""Main script for the Mail Exchange server detection
Independant of the crawler"""

import re
import pandas as pd
import numpy as np
import os
from os.path import join
import time
import sys
import dns
import dns.resolver as reso

# change path
pth = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(pth)
sys.path.append(pth)

from mails.config_mail import CONFIG_MAIL
from utils import link_to_domain

EXPECTED_ORDER = ["url", "has_mx_record", 	"MXRecord", "has_own_mx_record", "mail_comment", "ind_non_schema"]

def query_mail_exchange(url):
    """Send a query for MX record of the url"""
    mxRecord = None
    comment = ""
    try:
        # records = dns.resolver.query(url, 'MX')
        records = reso.query(url, 'MX')
        mxRecord = records[0].exchange
        mxRecord = str(mxRecord)
        a = 1
    except reso.NoAnswer:
    # except dns.resolver.NoAnswer:
        comment = "No answer"
    except reso.NoNameservers as e:
    # except dns.resolver.NoNameservers as e:
        comment = "No name server"
    except dns.exception.Timeout:
        comment = "Timeout"
    except dns.resolver.NXDOMAIN:
        comment = "No existing query name"
    except Exception as e:
        print("Special error: {} - {}".format(type(e), str(e)))
        comment = str(e)

    has_mx_record = mxRecord is None

    rs = {"url": url, "has_mx_record": has_mx_record,"MXRecord": mxRecord, "mail_comment": comment}
    return rs


def load_preproc(fpath):
    df = pd.read_csv(fpath, encoding="utf-8")

    # Remove point and ignonre wrong urls
    df["url"] = df["url"].astype(str)
    df["url"] = df["url"].apply(lambda x: x[0:-1] if x.endswith(".") else x)
    ind_valid_schema = df["url"].apply(lambda x: (re.search("\.", x) is not None))

    df["ind_non_schema"] = True
    df.loc[~ind_valid_schema, "ind_non_schema"] = False
    return df


def main():
    """
    Main function for Mail Exchange server detection (independant of the crawler)
    :return: save result
    """
    files = os.listdir(CONFIG_MAIL["input_folder"])
    files = sorted(files, reverse=False)

    print("N_files : {} \tfiles :{}".format(len(files), files))

    if len(files) == 0:
        print("NO FILES FOUND AT GIVEN INPUT FOLDER: RUN ABORTED")
        return None

    for f in files:
        print("PREPROCESSING for {}".format(f))
        input_path = join(CONFIG_MAIL["input_folder"], f)
        output_path = join(CONFIG_MAIL["MAIN_DIR"], "output", "res_" + f)
        print("input_file: {}\noutput_file: {}".format(input_path, output_path))

        df = load_preproc(input_path)

        # requests
        print("MX REQUESTS for {}".format(f))
        urls = list(df["url"])
        cnt = 0
        tot_cnt = len(urls)
        all_resu = []

        for url in urls:

            resu = query_mail_exchange(url)

            has_own_mx_record = False
            if resu["MXRecord"] is not None:

                mxrec = resu["MXRecord"].lower()
                if mxrec.endswith("."):
                    mxrec = mxrec[0:-1]

                input_domain = link_to_domain(url.lower())

                if mxrec.endswith(input_domain):
                    has_own_mx_record = True

            resu["has_own_mx_record"] =  has_own_mx_record

            time.sleep(CONFIG_MAIL["TIMEOUT_BETWEEN_VISIT"])

            all_resu.append(resu)

            cnt+=1
            if cnt % CONFIG_MAIL["LOG_EVERY_N"] ==0:
                print("progress: {}/{}".format(cnt, tot_cnt))

        # merge
        all_resu = pd.DataFrame(all_resu)
        df = pd.merge(df, all_resu, on="url", how="left")

        df.to_csv(output_path, encoding="utf-8", index=False)

    # concat results
    print("------------------CONCATENATING------------------------")
    df_final = None
    cnt_concat = 0
    for f in files:
        output_path = join(CONFIG_MAIL["MAIN_DIR"], "output", "res_" + f)
        df_tempo = pd.read_csv(output_path, encoding="utf-8")

        if df_final is None:
            df_final = df_tempo.copy(deep=True)
        else:
            df_final = pd.concat([df_final, df_tempo], axis=0, sort=False)
        cnt_concat += 1

    print("Total files concatenated : {}".format(cnt_concat))

    # quick format
    for colo in EXPECTED_ORDER:
        if colo not in df_final:
            df_final[colo] = np.nan
    all_col = list(df_final.columns)
    other_col = [e for e in all_col if e not in EXPECTED_ORDER]
    df_final = df_final[EXPECTED_ORDER + other_col]

    df_final.to_csv(CONFIG_MAIL["output_final_file_path"], encoding="utf-8",index=False)

    printAllDone()

def printAllDone():
    print("-" * 64 + "\n" + "-" * 64 + "\n" + "-" * 24 + "ALL DONE" + "-" * 32 + "\n" + "-" * 64 + "\n" + "-" * 64)


def OutputProcessing():
    """ OutputProcessing takes any csvs produced by this program and inserts them into the database defined in RUN_CONFIG[USE_DB] """
    import output_processing as op
    if CONFIG_MAIL["USE_DB"] == True:
        print("---------------------USING DATABASE------------------------")
        op.DoIt()
    else:
        print("---------------------NO DATABASE CONFIGURED------------------------")
        print("------------------RESULTS LEFT IN OUTPUT FOLDER--------------------")


if __name__ == '__main__':
    tic_tic = time.time()
    main()
    tic_tic2 = time.time()
    OutputProcessing()
    printAllDone()
    print("Total time in seconds (main): {}".format(tic_tic2 - tic_tic))
    print("Total time in seconds (DBinsert): {}".format(time.time() - tic_tic2))
