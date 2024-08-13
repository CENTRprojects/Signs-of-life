"""Script that loads manually labelled training dataset"""
import pandas as pd

from utils import load_obj


def get_training_data():
    """loads manually labelled training dataset"""
    fpin = r"D:\Documents\freelance\sign_of_life_crawler\output\perf_20191111_v2_tempo.csv"
    p_pickle = r"D:\Documents\freelance\sign_of_life_crawler\inter"
    fname_pickle = r"documents_perf_bench"
    # fpout = r"D:\Documents\freelance\sign_of_life_crawler\inter\ml\dataset.csv"
    fpout = r"D:\Documents\freelance\sign_of_life_crawler\inter\ml\dataset.json"

    df = pd.read_csv(fpin)
    # df = df.head(20)
    print(list(df.columns))
    list_wanted = [2,3,5]
    ind_wanted = df["ACT_SUB_CATEGORY"].isin(list_wanted)
    wanted_urls = set(df.loc[ind_wanted, "url"])
    df = df[ind_wanted].reset_index(drop=True)

    df_text = []
    dct = load_obj(fname_pickle, p_pickle)
    for doc in dct:
        if doc["url"] in wanted_urls:
            # list_pck.append([doc.url, doc.language, doc.raw_text, doc.clean_text])
            df_text.append({"url": doc["url"], "text": doc["clean_text"]})
    df_text = pd.DataFrame(df_text)


    df = pd.merge(df, df_text, how="left", on ="url")
    df = df[["url", "text", "language", "ACT_SUB_CATEGORY"]]
    df["target"] = 0
    df.loc[df["ACT_SUB_CATEGORY"].isin([2,3]), "target"] = 1

    # df_to_json = df.to_dict(orient="records")
    # with open(fpout, "r") as f:
    #     json.dump(f, df_to_json)

    jss = df.to_dict(orient="records")
    import json
    with open(r"D:\Documents\freelance\sign_of_life_crawler\inter\ml\dataset_cor.json", "w") as f:
        json.dumps(jss, indent=3)

    # df.to_json(fpout,  orient="records")
    # df.to_csv(fpout,  index=False, encoding="utf-8")
    print(df.shape)
    print(df["target"].sum())
    pass



if __name__ == '__main__':
    # sample()
    get_training_data()
    pass
