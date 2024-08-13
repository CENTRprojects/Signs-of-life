"""Tests on parking classification: redirections and empty pages identification"""
import unittest
import os
from os.path import join
import pandas as pd

from classification_parked import single_page_classify_parking
from config import RUN_CONFIG
from utils import load_obj

PATH_TESTS = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")

RES_REF = {"awvpghvh.loan": ["pred_is_empty", False],
           "emigrationdirect.com": ["redirection_type", "http"],
           "houseofbrowsamsterdam.com": ["redirection_type", "iframe"]}
RES_REF_REDIR = {
    "dnoone.com": ["is_redirected", False],
    "zz3z3.com": ["is_redirected", False],
    "teluguvaakili.com": ["is_redirected", False],
    "jpincemin.com": ["is_redirected", True],  # frameset with one significant frame and other src=""
    "reliancemultispecialityhospitals.com": ["redirection_type", "iframe"],
    "vivisin.com": ["redirection_type", "iframe"],
    "gublogs.com": ["redirection_type", "iframe"],  # frameset frame redirection
    "mypropaneprice.com": ["redirection_type", "iframe"],  # frameset frame redirection Majuscule
    "cibobuonochefabene.com": ["redirection_type", "window_location"],  # double slash
}


class ParkingTree(unittest.TestCase):

    def test_subcategories(self):

        ref = load_obj("tests_parking_tree", PATH_TESTS)

        docs_ref = []
        for doc in ref:
            docs_ref.append({"url": doc["url"], "clean_text": doc["clean_text"], "html_text": doc["raw_text"],
                             "history": doc["other_variables"]["history"],
                             "URL_history": doc["other_variables"]["URL_history"],
                             "language": doc["language"], "is_error": doc["is_error"]})

        # feat
        preds = []
        for feats in docs_ref:
            preds.append(single_page_classify_parking(feats))

        # companrison
        failed_tests = []
        for resu, _ in preds:
            url = resu["url"]
            if url in RES_REF:
                param = RES_REF[url][0]
                expected = RES_REF[url][1]
                if param in resu.keys():
                    predicted = resu[param]
                    if predicted == expected:
                        continue
                else:
                    predicted = "MISSING"

                failed_tests.append({"url": url, "param": param, "expected": expected, "predicted": predicted})

        self.assertEqual(len(failed_tests) == 0, True,
                         msg="The subcategories are not predicted correctly:\n{}".format(
                             "\n".join([e["url"] + "\t\t\t" + e["param"] + "\t expected :" + str(e[
                                                                                                     "expected"]) + "\t predicted :" + str(
                                 e["predicted"]) for e in
                                        failed_tests])))


def manual_test():
    fpout = RUN_CONFIG["output_final_file_path"][0:-4] + "_DEBUG.csv"

    list_res = pd.read_csv(fpout).to_dict(orient="records")

    # companrison
    failed_tests = []
    for resu in list_res:
        url = resu["url"]
        if url in RES_REF_REDIR.keys():
            param = RES_REF_REDIR[url][0]
            expected = RES_REF_REDIR[url][1]
            if param in resu.keys():
                predicted = resu[param]
                if predicted == expected:
                    continue
            else:
                predicted = "MISSING"

            failed_tests.append({"url": url, "param": param, "expected": expected, "predicted": predicted})


    assert len(failed_tests) == 0,"The subcategories are not predicted correctly:\n{}".format(
                         "\n".join([e["url"] + "\t\t\t" + e["param"] + "\t expected :" + str(e["expected"]) + "\t predicted :" + str(
                             e["predicted"]) for e in
                                    failed_tests]))
    print("validated")

if __name__ == '__main__':
    unittest.main()
    manual_test()
