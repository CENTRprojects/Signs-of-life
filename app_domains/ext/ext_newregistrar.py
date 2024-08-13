import pandas as pd

import tldextract
import logging

from utils import link_to_domain

log_tld = logging.getLogger("tldextract")
log_tld.setLevel(logging.ERROR)

import os
from os.path import join

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def agg_func(x):
    return ";".join(list(x))


def extdent_registrars_part1():
    """ Add new registrars """

    fp_new_regist = join(MAIN_DIR, r"input\pl_registrars.csv")
    fp_current = join(MAIN_DIR, r"input\hosting_companies_with_tld.csv")

    df_new = pd.read_csv(fp_new_regist, encoding="utf-8")
    list_curr = pd.read_csv(fp_current, encoding="latin-1").values.tolist()

    df_new = df_new.dropna(subset=["url"])

    print("curr ref : ({},{})".format(len(list_curr), len(list_curr[0])))
    print("new: {}".format(df_new.shape))

    # curr
    list_all_urls = []
    for name_tld in list_curr:
        for tld in name_tld[1].split(";"):
            list_all_urls.append(name_tld[0] + tld)

    # pd.DataFrame(list_all_urls, columns=["url"]).to_csv(r"D:\Documents\freelance\sign_of_life_crawler\input\flattened_registrars.csv", index=False)
    # import sys
    # sys.exit()

    for url in list(df_new["url"].drop_duplicates()):
        list_all_urls.append(url)

    # new
    df_new_ref = pd.DataFrame(list_all_urls, columns=["url"])
    df_new_ref = df_new_ref.drop_duplicates()
    df_new_ref["tld"] = df_new_ref["url"].apply(lambda x: "." + tldextract.extract(x)[2])
    df_new_ref["hosting_name"] = df_new_ref["url"].apply(lambda x: tldextract.extract(x)[1])

    print(df_new_ref.shape)

    dict_agg = {"tld": agg_func}
    df = df_new_ref.groupby("hosting_name", as_index=False).agg(dict_agg)

    # empty
    ind_null = df["tld"] == ""
    df = df[~ind_null]
    ind_null = df["hosting_name"] == ""
    df = df[~ind_null]

    print("new ref: {}".format(df.shape))

    df.to_csv(fp_current, index=False)
    a = 1

    pass


def extdent_registrars_part2():
    """Generate all combinations"""

    fp_new_regist = join(MAIN_DIR, r"input\pl_registrars.csv")
    fpin_cctld = join(MAIN_DIR, r"input\cctlds.csv")
    fpin_gtld = join(MAIN_DIR, r"input\gtlds.csv")

    # fpin_deriv_cctld = r"D:\Documents\freelance\sign_of_life_crawler\input\most_famous_websites_by_country.csv"
    fpout = join(MAIN_DIR, r"input\registrar\all_combi.csv")

    # names
    df = pd.read_csv(fp_new_regist, encoding="utf-8")

    df["url"] = df["url"].apply(lambda x: link_to_domain(x))

    df["tld_full"] = df["url"].apply(lambda x: "_".join(list(tldextract.extract(x))))
    df["tld"] = df["url"].apply(lambda x: tldextract.extract(x)[2])
    df["hosting_companies"] = df["url"].apply(lambda x: tldextract.extract(x)[1])

    df["id"] = df[["hosting_companies", "tld"]].apply(lambda x: x[0] + "__" + x[1], axis=0)
    existing_url = list(df["url"].drop_duplicates())
    names = list(set(list(df["hosting_companies"].dropna())))
    names = [e for e in names if len(e) > 0]

    # cctlds
    df_cctlds = pd.read_csv(fpin_cctld, encoding="latin-1")

    # gtlds
    df_gtlds = pd.read_csv(fpin_gtld, encoding="latin-1")

    list_tlds = list(set(list(df_cctlds["ccTLD"].dropna()) + list(df_gtlds["gtld"].dropna())))

    # derived cctlds --> included in cctls
    # df_deriv_cctlds = pd.read_csv(fpin_deriv_cctld)
    # df_deriv_cctlds["tld"] = df_deriv_cctlds["final_url"].apply(lambda x: tldextract.extract(x)[2])
    # df_deriv_cctlds["hosting_companies"] = df_deriv_cctlds["final_url"].apply(lambda x: tldextract.extract(x)[1])
    # df_deriv_cctlds.to_csv(r"D:\Documents\freelance\sign_of_life_crawler\input\dcc.csv")

    # combi
    all_names = []
    for tld in list_tlds:
        for nm in names:
            all_names.append(nm + tld)

    df = pd.DataFrame(all_names, columns=["url"])

    # remove existing
    ind_exist = df["url"].isin(existing_url)
    df = df[~ind_exist]

    print(df.shape)
    print(df.head())
    df.to_csv(fpout, index=False)
    print("done")


def extdent_registrars_part3():
    # fp_new_regist = r"D:\Documents\freelance\sign_of_life_crawler\input\nz_registrars.csv"
    fp_new_regist_combi = join(MAIN_DIR, r"output\new_combi.csv")
    fp_current = join(MAIN_DIR, r"input\hosting_companies_with_tld.csv")

    list_curr = pd.read_csv(fp_current, encoding="latin-1").values.tolist()
    df_new = pd.read_csv(fp_new_regist_combi, encoding="latin-1")

    df_new = df_new.dropna(subset=["url"])

    print(len(list_curr), ",", len(list_curr[0]))
    print(df_new.shape)

    # curr
    list_all_urls = []
    for name_tld in list_curr:
        for tld in name_tld[1].split(";"):
            list_all_urls.append(name_tld[0] + tld)

    for url in list(df_new["url"].drop_duplicates()):
        list_all_urls.append(url)

    # new
    df_new_ref = pd.DataFrame(list_all_urls, columns=["url"])
    df_new_ref = df_new_ref.drop_duplicates()
    df_new_ref["tld"] = df_new_ref["url"].apply(lambda x: "." + tldextract.extract(x)[2])
    df_new_ref["hosting_name"] = df_new_ref["url"].apply(lambda x: tldextract.extract(x)[1])

    print(df_new_ref.shape)

    dict_agg = {"tld": agg_func}
    df = df_new_ref.groupby("hosting_name", as_index=False).agg(dict_agg)

    # empty
    ind_null = df["tld"] == ""
    df = df[~ind_null]
    ind_null = df["hosting_name"] == ""
    df = df[~ind_null]

    print("new ref: {}".format(df.shape))

    a = 1
    df.to_csv(fp_current, index=False)
    print("done")


if __name__ == '__main__':
    # extdent_registrars_part1()
    # extdent_registrars_part2()
    # manually filter parking registrar
    extdent_registrars_part3()
