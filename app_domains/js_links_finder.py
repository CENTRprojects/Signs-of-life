"""Script to parse HTML page to extract script tags"""
import re
from html.parser import HTMLParser

from utils import save_obj

THRESHOLD_LINK = 5


class LibrariesParser(HTMLParser):
    """HTML tag parser"""
    def __init__(self):
        HTMLParser.__init__(self)
        self.links = []
        self.flag_js_found = 0
        self.flag_php_found = 0
        self.record = 0
        self.contents = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.flag_js_found = 1
            self.record += 1

            for at in attrs:
                if at[0] == "src":
                    txt = at[1]
                    if re.search("\.", str(txt)):
                        # remove full path
                        cln = re.split("[/\\\]+", txt)[-1]

                        # remove version after question
                        clnq = cln.split("?")[0]

                        # remove - version
                        final = re.sub("-[0-9.]+", ".", clnq)

                        self.links.append(final)
                        # self.data.append(clean_link(txt))

        if tag == "php":
            self.flag_php_found = 1

    def handle_endtag(self, tag):
        if tag == "script":
            if self.record:
                self.record -= 1

    def handle_data(self, data):
        if self.record:
            str_data = str(data)
            if len(str_data) > 0:
                self.contents.append(str_data)

    def error(self, message):
        print("ERROR: {}".format(message))


def find_js_links(html_text):
    """Extract javascript libraries from HTML tags 'SCRIPT'"""
    parser = LibrariesParser()
    try:
        parser.feed(html_text)
    except:
        print("ERROR: Html parsing failed for :\n{}".format(html_text))

    return list(set(parser.links)), parser.contents, parser.flag_js_found, parser.flag_php_found
