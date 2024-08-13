"""Page Processing step = Script that extract displayed text and detect language from an HTML page"""
import re
from bs4 import BeautifulSoup
from langdetect import detect
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import tag_visible
from config import RUN_CONFIG, PLOG

MIN_LENGTH_LGG_DETECTION = 10
MAX_FIRST_LETTERS_LGG_DETECTION = 3000
MIN_LETTER_SENTENCE = 4
LIMIT_JS_SIZE = 10000

HTML_FORMAT_TAGS = ["u", "b", "em", "sub", "sup", "strong", "br"]

RE_HAS_LETTER = re.compile("[A-Z]", flags=re.IGNORECASE)

def get_page_displayed_text(list_websites):
    """Extract hard-coded displayed text from a list of HTML pages"""
    # html to text
    list_url_to_text = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
        delayed(unit_html_to_text)(batch) for batch in tqdm(list_websites))
    dict_url_to_text = dict()
    for e in list_url_to_text:
        dict_url_to_text[e[0]] = e[1::]
    for doc in list_websites:
        doc.clean_text = dict_url_to_text[doc.url][0]
        doc.other_variables["non_text"] = dict_url_to_text[doc.url][1]
    return list_websites


def get_page_language(list_websites_with_displayed_text):
    """Detect the language of a list of HTML pages with their associated displayed texts"""
    # language detection
    list_url_to_lgg = Parallel(n_jobs=RUN_CONFIG["WORKERS_POST_PROCESSING"])(
        delayed(unit_detect_language)(doc) for doc in tqdm(list_websites_with_displayed_text))
    # format results
    dict_url_to_lgg = dict()
    for e in list_url_to_lgg:
        dict_url_to_lgg[e[0]] = e[1]
    for doc in list_websites_with_displayed_text:
        doc.language = dict_url_to_lgg[doc.url]
    return list_websites_with_displayed_text


def unit_html_to_text(doc):
    """Extract hard-coded displayed text from one HTML pages"""
    """Extract text that is displayed on screen through associated HTML tags"""
    clean_text = None
    non_text = None

    if (doc.raw_text is not None) & (RE_HAS_LETTER.search(str(doc.raw_text)) is not None):
        try:
            raw_text = doc.raw_text

            # removing format tags
            raw_text = remove_format_tags(raw_text)

            soup = BeautifulSoup(raw_text, 'html.parser')

            texts = soup.findAll(text=True)
            titles_of_links = soup.findAll("a", title=True)
            titles_of_links = ".".join([str(e.attrs["title"]) for e in titles_of_links])

            alt_of_imgs = soup.findAll("img", alt=True)
            alt_of_imgs = ".".join([str(e.attrs["alt"]) for e in alt_of_imgs])

            meta_name_descr = soup.find("meta", content=True, attrs={"name": "description"})
            if meta_name_descr is not None:
                meta_content = "." + str(meta_name_descr.attrs["content"])
            else:
                meta_content = ""

            visible_texts = filter(tag_visible, texts)
            clean_text = ".".join([e for e in [t.strip() for t in visible_texts] if len(e) > 0])
            clean_text = ".".join([clean_text, titles_of_links, alt_of_imgs])
            clean_text += meta_content

            # non text
            non_text = ""
            for elem in soup.findAll():
                elem_non_text = ""

                if elem.attrs:
                    elem_non_text += "__".join([str(k) + ":" + str(v) for k, v in elem.attrs.items()])

                if elem.name == "script":
                    elem_non_text += "::" + elem.text

                non_text += elem_non_text

            non_text = non_text[0:LIMIT_JS_SIZE]

        except Exception as e:
            PLOG.it("Error with html parsing for {} \t {} \t{}".format(doc.url, type(e), str(e)), is_error=True)
            clean_text = None
            non_text = None

    return [doc.url, clean_text, non_text]


def remove_format_tags(raw_text):
    """Removing some formatting HTML tags"""
    for tag in HTML_FORMAT_TAGS:
        raw_text = re.sub("<" + tag + ">", "", raw_text)
        raw_text = re.sub("</" + tag + ">", "", raw_text)
    return raw_text


def unit_detect_language(doc):
    """Detect the language of a list of HTML page using its displayed texts"""
    lgg = "other"
    if doc.clean_text is not None:
        if len(doc.clean_text) > MIN_LENGTH_LGG_DETECTION:

            tempo_displayed_text = re.sub("^[0-9]*", "", doc.clean_text)[0:MAX_FIRST_LETTERS_LGG_DETECTION]

            sentences = re.split("[?.!]", tempo_displayed_text)
            dico_seps = {}
            index_sep = 0
            for sent in sentences:
                index_sep += len(sent)
                if index_sep < len(tempo_displayed_text):
                    dico_seps[sent] = tempo_displayed_text[index_sep]
                else:
                    dico_seps[sent] = ""
                index_sep +=1

            # sentences = sorted(sentences, key= lambda x: -len(x))
            final_sorted_text = ""
            for e in sentences:
                if len(e)> MIN_LETTER_SENTENCE:
                    final_sorted_text += e + dico_seps[e]

            try:
                lgg = detect(final_sorted_text)
                # lgg = detect(
                #     re.sub("^[ 0-9]*", "", doc.clean_text)[0:MAX_FIRST_LETTERS_LGG_DETECTION])  # remove unecessary text
            except Exception as e:
                if not re.search("No features in text", str(e)):
                    PLOG.it("ERROR during language detection --> other used : type:{} message:{}".format(type(e),
                                                                                                       str(e)), is_error=True)
    return [doc.url, lgg]
