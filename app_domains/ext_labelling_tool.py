"""Unofficial script to visit a list of URLs with a browser"""

import re
import pandas as pd
from selenium import webdriver

# options
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('window-size=1200x600')

ALLOWED_CATEGORIES = [str(e) for e in list(range(1, 10))]


def single_domain_browser_visit(link, driver, cnt):
    try:
        print("{}\t{}".format(cnt, link))
        full_url = link
        if re.search("^https*:", full_url, re.IGNORECASE) is None:
            full_url = "http://" + full_url
        driver.get(full_url)
        while True:
            # print("while")
            category = str(input())
            if category in ALLOWED_CATEGORIES:
                resu = category
                break
            else:
                print("not correct: {}".format(category))



    except Exception as e:
        print("{} : JS error with of type: {} : {} --> js interpretation removed".format(link, type(e), str(e)))
        resu = None

    return resu

def remove_point(x):
    if x.endswith("."):
        return x[0:-1]
    else:
        return x

def labelling_by_input():
    """Label a list of URLs by user input in console command"""
    fpin = r"D:\freelance\sign_of_life_crawler\project\fr_check\fr_check_dns_error.csv"
    fpout = r"D:\freelance\sign_of_life_crawler\project\fr_check\ground_truth_20191110.csv"

    list_urls = list(pd.read_csv(fpin)["url"])

    list_urls = [remove_point(e) for e in list_urls]

    driver = webdriver.Chrome(chrome_options=OPTIONS)

    cnt = 1
    all_resu = []
    for link in list_urls:

        # visit
        categ = single_domain_browser_visit(link, driver, cnt)

        # save
        all_resu.append({"url": link, "ACT_SUB_CATEGORY": categ})
        # if cnt % 2 == 0:
        if cnt % 15 == 0:
            pd.DataFrame(all_resu).to_csv(fpout[0:-4] + "_tempo.csv", index=False)

        # init driver
        # if cnt % 3 == 0:
        if cnt % 30 == 0:
            driver.close()
            driver = webdriver.Chrome(chrome_options=OPTIONS)

        cnt += 1

    pd.DataFrame(all_resu).to_csv(fpout, index=False)
    print("Done")


if __name__ == '__main__':
    labelling_by_input()
