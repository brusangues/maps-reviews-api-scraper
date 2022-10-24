import requests
from lxml import etree, html
from bs4 import BeautifulSoup, Tag, NavigableString, Comment
import regex as re
from typing import Union

hl = "pt-BR"


def get_text(feature_id: str, token: str = "") -> str:
    query = (
        "https://www.google.com/async/reviewDialog?"
        f"hl={hl}&"
        f"async="
        f"feature_id:{feature_id},"
        f"sort_by:,"
        f"next_page_token:{token},"
        f"associated_topic:,"
        f"_fmt:pc"
    )
    response = requests.get(query)
    text = response.content.decode(encoding="unicode_escape")
    return text


def cut_text(text: str) -> str:
    idx_first_div = text.find("<div ")
    m = re.search("</div>", text, flags=re.REVERSE)
    idx_last_div = m.span()[1]
    text = text[idx_first_div:idx_last_div]
    return "<html><body>" + text + "</body></html>"


def node_to_xpath(node):
    node_type = {
        Tag: getattr(node, "name"),
        Comment: "comment()",
        NavigableString: "text()",
    }
    same_type_siblings = list(
        node.parent.find_all(
            lambda x: getattr(node, "name", True) == getattr(x, "name", False),
            recursive=False,
        )
    )
    if len(same_type_siblings) <= 1:
        return node_type[type(node)]
    pos = same_type_siblings.index(node) + 1
    return f"{node_type[type(node)]}[{pos}]"


def get_node_xpath(node: Union[Tag, Comment]):
    xpath = "/"
    elements = [f"{node_to_xpath(node)}"]
    for p in node.parents:
        if p.name == "[document]":
            break
        elements.insert(0, node_to_xpath(p))

    xpath = "/" + xpath.join(elements)
    return xpath
