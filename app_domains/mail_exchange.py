"""Function that query the MX records of a domain"""
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# lower level of loggers
import logging
import dns
import dns.resolver as reso

loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)

from config import RUN_CONFIG, PLOG
from utils import link_to_domain

MAIL_EXPECTED_COLS = ["url", "has_mx_record", "MXRecord", "has_own_mx_record", "mail_comment"]


def identify_mx_records_all_domains(documents):
    """Query the MX records of a list of domain"""

    # features
    X_first = select_features_mail(documents)

    # First classification
    if RUN_CONFIG["MULTI_PROCESSING"]:
        list_res = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
            delayed(identify_mx_records_one_domain)(batch) for batch in tqdm(X_first))
    else:
        # non parallel
        list_res = []
        for doc in X_first:
            list_res.append(identify_mx_records_one_domain(doc))
    return pd.DataFrame(list_res)


def select_features_mail(documents):
    return [{"url": doc.url} for doc in documents]


def query_mail_exchange(url):
    """Send a query for MX record of the url"""
    mxRecord = None
    comment = ""
    try:
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
    except reso.NXDOMAIN:
        comment = "No existing query name"
    except Exception as e:
        print("Special error: {} - {}".format(type(e), str(e)))
        comment = str(e)

    has_mx_record = mxRecord is not None

    rs = {"url": url, "has_mx_record": has_mx_record, "MXRecord": mxRecord, "mail_comment": comment}
    return rs


def identify_mx_records_one_domain(feats):
    """Query the MX record of one domain:
    = Redirection Identification + Social media extraction and filtering
    feats: features of one url
    """
    url = feats["url"]

    try:

        # get mx
        resu = query_mail_exchange(url)

        #
        has_own_mx_record = False
        if resu["MXRecord"] is not None:

            mxrec = resu["MXRecord"].lower()
            if mxrec.endswith("."):
                mxrec = mxrec[0:-1]

            input_domain = link_to_domain(url.lower())

            if mxrec.endswith(input_domain):
                has_own_mx_record = True

        resu["has_own_mx_record"] = has_own_mx_record

    except Exception as e:
        PLOG.it("ERROR during mail record extraction --> {} not classified : type:{} message:{}".format(feats["url"],
                                                                                                        type(e),
                                                                                                        str(e)))
        print("ERROR during mail record extraction --> {} not classified : type:{} message:{}".format(feats["url"],
                                                                                                      type(e),
                                                                                                      str(e)))
        # "has_mx_record", 	"MXRecord", "has_own_mx_record", "mail_comment", "ind_non_schema"
        resu = {"url": url, "has_mx_record": False, "has_own_mx_record": False,
                "mail_comment": "error during mail extraction"}
    return resu
