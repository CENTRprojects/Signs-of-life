"""Standalone script to label a list of URLS using user inputs and Chrome webdriver"""
import re
import pandas as pd
from selenium import webdriver

# options
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('window-size=1200x600')

ALLOWED_CATEGORIES = [str(e) for e in list(range(0, 6))]


def single_url_browser_visit_and_label(link, driver, cnt):
    """Visit link with browser and request user input (as label)"""
    try:
        print("{}\t\t{}".format( link, cnt))
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

def labelling():
    """Label the list of URLs in fpin and save user input in fpout"""
    fpin = r"/media/totoyoutou/ssd/freelance/sign_of_life_crawler/input/focused_label/focused_labelling_2.csv"
    fpout = r"/media/totoyoutou/ssd/freelance/sign_of_life_crawler/output/ground_truth_focused_20200423.csv"
    N_start = 0

    df_in = pd.read_csv(fpin)
    df_in = df_in.drop_duplicates(subset=["url"])
    list_urls = list(df_in["url"])

    list_urls = [remove_point(e) for e in list_urls]

    driver = webdriver.Chrome(chrome_options=OPTIONS)

    cnt = 1
    cnt_final = 0
    all_resu = []
    for link in list_urls[N_start::]:

        try:
            # visit
            categ = single_url_browser_visit_and_label(link, driver, cnt)

            if categ is None:
                # driver.close()
                driver = webdriver.Chrome(chrome_options=OPTIONS)

            # save
            all_resu.append({"url": link, "ACT_SUB_CATEGORY": categ})

            if str(categ) in {"0", "1", "2"}:
                cnt_final+=1

            # if cnt % 2 == 0:
            if cnt % 15 == 0:
                pd.DataFrame(all_resu).to_csv(fpout[0:-4] + "_tempo.csv", index=False)

            # init driver
            # if cnt % 3 == 0:
            if cnt_final % 50 == 0:
                print("Total up to now:{}".format(cnt_final))

            if cnt % 30 == 0:
                driver.close()
                driver = webdriver.Chrome(chrome_options=OPTIONS)
            cnt += 1

        except Exception as e:
            print("Error with \t{}\t{}\t{}".format(link, type(e), str(e)))
            # driver.close()
            driver = webdriver.Chrome(chrome_options=OPTIONS)

    pd.DataFrame(all_resu).to_csv(fpout, index=False)
    print("Done")


if __name__ == '__main__':
    labelling()
