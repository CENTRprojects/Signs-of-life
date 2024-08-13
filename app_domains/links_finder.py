"""HTML page parser that extract all links and other relevant HTML information"""
import re
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

from utils import clean_link

FLAG_WINDOW_LOCATION = "window\.location(\.href*)* *=[\]*[\"']([^\"']+)[\]*[\"']"

html_repetitive_and_unsignificant_tags = ["td", "tr", "li", "ul", "b", "br", "em", "sub", "sup", "p", "u", "strong"]
DICO_TRIVIAL_TAGS = dict(
    zip(html_repetitive_and_unsignificant_tags, [True] * len(html_repetitive_and_unsignificant_tags)))

RE_REFRESH = re.compile("refresh", flags=re.IGNORECASE)


def is_trivial_link(lk):
    """ Identify trivial links like mail and same page element, as opposed to links to other pages"""
    if lk == "":
        return True
    elif lk.startswith("#"):
        return True
    elif lk.startswith("mailto"):
        return True
    elif re.search("blank|empty|notfound", lk.split("/")[-1]):
        return True
    else:
        return False


def relative_to_absolute(url, base_url):
    """Turn relative URL to absolute by adding base reference url"""
    url = url.replace("\\", "/")

    if re.search("^https{0,1}:", url, re.IGNORECASE):
        return url
    elif url.startswith("//"):
        return url
    else:
        if url.startswith("./"):
            url = url[1::]

        if url.startswith("/"):
            if base_url.endswith("/"):
                full_url = "http://" + base_url[0:-1] + url
            else:
                full_url = "http://" + base_url + url
        else:
            if base_url.endswith("/"):
                full_url = "http://" + base_url + url
            else:
                full_url = "http://" + base_url + "/" + url

        return full_url


def parse_html(html_text, url):
    """Extract miscellaneous TAGs information on an HTML page
    = Links + Iframes/Framesets + Javascript tags & functions + Meta tags"""
    # a tag and iframe
    # parser = ContentParser(url)
    # parser.feed(html_text)

    soup = BeautifulSoup(html_text, "html.parser")

    # parsing
    a__tags = soup.findAll("a", href=True)
    # links_tags = soup.findAll("link", href=True)
    links_area = soup.findAll("area", href=True)
    frameset_tags = soup.findAll("frameset")
    iframe_tags = soup.findAll("iframe", src=True)
    script_tag = soup.findAll("script")
    no_script = soup.findAll("noscript")
    base_url = soup.find("base", href=True)
    if base_url:
        reference_url = base_url.attrs["href"]
    else:
        reference_url = url

    # length_html
    try:
        normalized_html = soup.prettify()
    except:
        normalized_html = re.sub("\n+", "\n", html_text)

    n_lines_html = len(normalized_html) - len(normalized_html.replace("\n", ""))

    # links
    # links = [a.attrs["href"] for a in a__tags] + [a.attrs["href"] for a in links_tags]
    all_links = [a.attrs["href"] for a in a__tags] + [a.attrs["href"] for a in links_area]
    all_links = [lk for lk in all_links if not is_trivial_link(lk)]
    all_links = [relative_to_absolute(lk, reference_url) for lk in all_links]
    all_links = [clean_link(lk) for lk in all_links]

    # meta refresh url
    metas = soup.find("meta", {"http-equiv": RE_REFRESH}, url=True)
    if metas is None:
        meta_refresh = None
    else:
        src = metas.attrs["url"]
        if not is_trivial_link(src):
            src_full = relative_to_absolute(src, reference_url)
            meta_refresh = {"link": src_full}
        else:
            meta_refresh = None

    if meta_refresh is None:
        # meta refresh content
        metas = soup.find("meta", {"http-equiv": RE_REFRESH}, content=True)
        if metas is not None:
            src_content = metas.attrs["content"]
            link_pattern = re.search("url\t* *=([^'\"]*)", src_content, re.IGNORECASE)

            if link_pattern:
                src = link_pattern.groups()[0].strip()

                if not is_trivial_link(src):
                    src_full = relative_to_absolute(src, reference_url)
                    meta_refresh = {"link": src_full}
                else:
                    meta_refresh = None

    # window locs
    window_loc = []
    for script in script_tag:
        window_loc_tag = re.findall(FLAG_WINDOW_LOCATION, script.text)
        for wloc in window_loc_tag:
            src = wloc[1]
            if not is_trivial_link(src):
                src_full = relative_to_absolute(src, reference_url)
                window_loc.append({"link": src_full})

    # flag_js
    if len(script_tag) > 0:
        flag_js_found = True
    else:
        flag_js_found = False

    # length JS
    lg_inline_script = 0
    for js_elem in script_tag:
        lg_inline_script += len(js_elem.text)

    # iframes
    iframes = []
    for fr in iframe_tags:
        src = fr.attrs["src"]
        significance = 0

        # is video
        is_video = False
        if "allowfullscreen" in fr.attrs:
            is_video = True

        # path
        parent = fr.parent.name

        # add to links
        if (not is_trivial_link(src)) and (not is_video):
            all_links.append(clean_link(src))

            significance = 1

        src_full = relative_to_absolute(src, reference_url)
        iframes.append({"link": src_full, "parent": parent, "type": "iframe", "significance": significance})

    # framesets
    framesets = []
    for frset in frameset_tags:
        significance = 0

        frames_tags = frset.findAll("frame", src=True)

        frames = []
        for fr in frames_tags:
            src = fr.attrs["src"]
            fr_significance = 0

            # non trivial
            if not is_trivial_link(src):
                all_links.append(clean_link(src))
                fr_significance = 1

            src_full = relative_to_absolute(src, reference_url)
            frames.append({"link": src_full, "type": "frame", "significance": fr_significance})

            significance += fr_significance

        # path
        parent = frset.parent.name

        framesets.append({"type": "frameset", "frames": frames, "parent": parent, "significance": significance})

    frames = framesets + iframes
    flag_iframe = len(frames) > 0

    dico_html_complexity = dev_html_complexity(soup)

    # enable javascript
    no_script_text = ".".join([tag.text for tag in no_script])

    # output
    html_data = {"all_links": all_links,
                 "frames": frames,
                 "flag_iframe": flag_iframe,
                 "flag_js_found": flag_js_found,
                 "window_loc": window_loc,
                 "meta_refresh": meta_refresh,
                 "n_lines_html": n_lines_html,
                 "dico_html_complexity": dico_html_complexity,
                 "no_script_text": no_script_text,
                 "lg_inline_script": lg_inline_script}

    return html_data
    # return links, all_frames, flag_iframe, flag_js_found, window_locs, meta_refresh, n_lines_html, dico_html_complexity, no_script_text, lg_inline_script


def get_path(elem):
    """Returns path of positions in the HTML tree"""
    path = ""
    for tag in elem.parents:
        # get position
        pos = 0
        curr_tag = tag
        while curr_tag.previousSibling is not None:
            curr_tag = curr_tag.previousSibling
            pos += 1

        if path:
            path = str(pos) + "_" + path
        else:
            path = str(pos)

    return path


def parse_html_for_sm(html_text, url):
    """ Extract all relevant information/elements out of the HTML page
    = Links + Iframe + Window.Location + Social Media Tags"""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception as e:
        print("Error with html parsing for {} \t {} \t{} --> no links extracted".format(url, type(e), str(e)))
        soup = BeautifulSoup("", "html.parser")

    # parsing
    a__tags = soup.findAll("a", href=True)
    links_area = soup.findAll("area", href=True)
    frameset_tags = soup.findAll("frameset")
    iframe_tags = soup.findAll("iframe", src=True)
    script_tag = soup.findAll("script")
    base_url = soup.find("base", href=True)
    if base_url:
        reference_url = base_url.attrs["href"]
    else:
        reference_url = url

    # length_html
    try:
        normalized_html = soup.prettify()
    except:
        normalized_html = re.sub("\n+", "\n", html_text)

    n_lines_html = len(normalized_html) - len(normalized_html.replace("\n", ""))

    # links
    links = [(a.attrs["href"], a) for a in a__tags] + [(a.attrs["href"], a) for a in links_area]
    links = [pair for pair in links if not is_trivial_link(pair[0])]
    links = [(relative_to_absolute(pair[0], reference_url), pair[1]) for pair in links]
    links = [(clean_link(pair[0]), pair[1]) for pair in links]

    # meta refresh url
    metas = soup.find("meta", {"http-equiv": RE_REFRESH}, url=True)
    if metas is None:
        meta_refresh = None
    else:
        src = metas.attrs["url"]
        if not is_trivial_link(src):
            src_full = relative_to_absolute(src, reference_url)
            meta_refresh = {"link": src_full}
        else:
            meta_refresh = None

    if meta_refresh is None:
        # meta refresh content
        metas = soup.find("meta", {"http-equiv": RE_REFRESH}, content=True)
        if metas is not None:
            src_content = metas.attrs["content"]
            link_pattern = re.search("url\t* *=([^'\"]*)", src_content, re.IGNORECASE)

            if link_pattern:
                src = link_pattern.groups()[0].strip()

                if not is_trivial_link(src):
                    src_full = relative_to_absolute(src, reference_url)
                    meta_refresh = {"link": src_full}
                else:
                    meta_refresh = None

    # window locs
    window_locs = []
    for script in script_tag:
        window_loc_tag = re.findall(FLAG_WINDOW_LOCATION, script.text)
        for wloc in window_loc_tag:
            src = wloc[1]
            if not is_trivial_link(src):
                src_full = relative_to_absolute(src, reference_url)
                window_locs.append({"link": src_full})

    # iframes
    iframes = []
    for fr in iframe_tags:
        src = fr.attrs["src"]
        significance = 0

        # is video
        is_video = False
        if "allowfullscreen" in fr.attrs:
            is_video = True

        # path
        parent = fr.parent.name

        # add to links
        if (not is_trivial_link(src)) and (not is_video):
            links.append((clean_link(src), fr))

            significance = 1

        src_full = relative_to_absolute(src, reference_url)
        iframes.append({"link": src_full, "parent": parent, "type": "iframe", "significance": significance})

    # framesets
    framesets = []
    for frset in frameset_tags:
        significance = 0

        frames_tags = frset.findAll("frame", src=True)

        frames = []
        for fr in frames_tags:
            src = fr.attrs["src"]
            fr_significance = 0

            # non trivial
            if not is_trivial_link(src):
                links.append((clean_link(src), fr))
                fr_significance = 1

            src_full = relative_to_absolute(src, reference_url)
            frames.append({"link": src_full, "type": "frame", "significance": fr_significance})

            significance += fr_significance

        # path
        parent = frset.parent.name

        framesets.append({"type": "frameset", "frames": frames, "parent": parent, "significance": significance})

    all_frames = framesets + iframes

    # add path
    links = [(pair[0], get_path(pair[1]), pair[1].name) for pair in links]

    # SM media tags
    # Open Graph
    has_open_graph = False
    metas_prop = soup.findAll("meta", property=True)
    for elem in metas_prop:
        if elem.attrs["property"].startswith("og:"):
            has_open_graph = True
            break

    # Twitter card
    has_twitter_card = False
    metas_name = soup.findAll("meta", attrs={"name": re.compile(r".*")})
    for elem in metas_name:
        if elem.attrs["name"].startswith("twitter:"):
            has_twitter_card = True
            break

    # Schema Tag
    has_schema_tag = False
    metas_item = soup.findAll("meta", itemprop=True)
    if len(metas_item) > 0:
        has_schema_tag = True

    sm_meta_tags = {"has_open_graph": has_open_graph, "has_twitter_card": has_twitter_card,
                    "has_schema_tag": has_schema_tag}

    return links, all_frames, window_locs, meta_refresh, n_lines_html, sm_meta_tags


def dev_html_complexity(soup):
    """Computes a measure of HTML complexity: the total number of tags"""
    all_tags = [e for e in soup.find_all()]
    all_tags = [e.name for e in all_tags if not isinstance(e, Comment)]
    all_tags = [e for e in all_tags if e not in DICO_TRIVIAL_TAGS]
    n_tags = len(all_tags)

    dico_resu = {"tag_quantity": n_tags}
    return dico_resu
