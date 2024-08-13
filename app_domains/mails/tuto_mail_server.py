"""Basic mail exchange server detection"""
import dns.resolver
import os
import pandas as pd
from os.path import join
from concurrent import futures


def query_mail_exchange(url):
    """Send a query for MX record of the url"""
    mxRecord = None
    comment = ""
    try:
        records = dns.resolver.query(url, 'MX')
        mxRecord = records[0].exchange
        mxRecord = str(mxRecord)
        # print("{}\t\t{}".format(url, mxRecord))
    except dns.resolver.NoAnswer:
        # print("no answer for " + url)
        comment = "No answer"
    except dns.resolver.NoNameservers as e:
        # print("no name server " + url)
        comment = "No name server"
    except dns.exception.Timeout:
        comment = "Timeout"
        # print("Timeout for " + url)
    except dns.resolver.NXDOMAIN:
        print("No existing query name for " + url)
        comment = "No existing query name"
    except Exception as e:
        print(str(e))
        comment = str(e)

    rs = {"url": url, "MXRecord": mxRecord, "comment": comment}
    return rs


def test_dnsresolve():
    input_path = join(os.path.dirname(__file__), "input", "batch_dns_error", "dns_error.csv")
    resu_path_tempo = join(os.path.dirname(__file__), "output", "mail_resolver_tempo.csv")
    resu_path = join(os.path.dirname(__file__), "output", "mail_resolver.csv")

    df = pd.read_csv(input_path, encoding="latin-1")
    all_urls = list(df.tail(len(df) - 2500)["url"])

    all_res = []
    cnt = 0
    tot_cnt = len(all_urls)
    print("Total of " + str(tot_cnt))
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(query_mail_exchange, all_urls):
            cnt += 1
            all_res.append(res)

            if cnt % 500 == 0:
                pd.DataFrame(all_res).to_csv(resu_path_tempo, index=False)
                print("{}/{}".format(cnt, tot_cnt))

    df_res = pd.DataFrame(all_res)
    df = pd.merge(df, df_res, on="url", how="left")
    df.to_csv(resu_path, index=False)


if __name__ == '__main__':
    test_dnsresolve()
