import requests

from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

from config import RUN_CONFIG, PLOG

def get_http_to_https_redirection_status(documents):

    # features
    X_first = select_features_redirect(documents)

    # First classification
    if RUN_CONFIG["MULTI_PROCESSING"]:
        list_res = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
            delayed(check_http_to_https_redirect)(batch) for batch in tqdm(X_first))
    else:
        # non parallel
        list_res = []
        for doc in X_first:
            list_res.append(check_http_to_https_redirect(doc))
    return pd.DataFrame(list_res)

def select_features_redirect(documents):
    return [{"url": doc.url} for doc in documents]

def check_http_to_https_redirect(feats):
    url = feats["url"]
    # Make an HTTP request
    response = requests.get(f'http://{url}', allow_redirects=False)

    # Check if the status code is 3xx (redirection)
    if response.status_code == 301 or response.status_code == 302:
        # Check if the 'Location' header contains 'https'
        if 'https' in response.headers.get('Location', ''):
            return True  # It's redirecting from HTTP to HTTPS
    return False  # No HTTP to HTTPS redirect found

if __name__ == "__main__":
    url = 'isoc.org.il'
    if check_http_to_https_redirect(url):
        print(f"{url} redirects from HTTP to HTTPS.")
    else:
        print(f"{url} does not redirect from HTTP to HTTPS.")