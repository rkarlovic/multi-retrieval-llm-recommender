# -*- coding: utf-8 -*-
"""
String manipulation utilities.

Utilities for common string normalization tasks.

Low-level exceptions are intentionally allowed to propagate so callers can
decide how to handle them.
"""

from __future__ import annotations
from bs4 import BeautifulSoup, Comment
import re

from typing import (
    Annotated, 
    List, 
    Pattern, 
    Sequence, 
    Union
)


__all__ = ["ParsedHTMLScrape", "parse_html"]


# Precompile regexes for clarity and performance
_MULTI_NEWLINE_RE: Pattern[str] = re.compile(r"\n{2,}")
_HORZ_WS_RE: Pattern[str] = re.compile(r"[^\S\n]+")  # horizontal whitespace (spaces/tabs), not newlines
_NEWLINE_LEADING_WS_RE: Pattern[str] = re.compile(r"\n[ \t]+")
_NEWLINE_TRAILING_WS_RE: Pattern[str] = re.compile(r"[ \t]+\n")


class ParsedHTMLScrape:
    """
    Represents the result of parsing HTML content.
    
    Attributes:
        url: The URL of the HTML content.
        title: The title of the HTML document.
        text: The extracted plain text from the HTML.
    """
    def __init__(self, url: str, title: str, text: str) -> None:
        self.url = url
        self.title = title
        self.text = text


def _normalize_newlines(text: str) -> Annotated[str, "raises ValueError", "raises Exception"]:
    """
    Normalize newlines in `text` by:
      - Normalizing CRLF/CR to LF.
      - Collapsing horizontal whitespace (spaces/tabs) into single spaces while
        preserving newlines.
      - Removing spaces/tabs immediately following or preceding newlines.
      - Collapsing two or more consecutive newlines into a single newline.
      - Stripping leading/trailing whitespace from the entire string.

    Args:
        text: The input string to normalize.

    Returns:
        The normalized string.

    Raises:
        TypeError: If `text` is not a string.
        ValueError: If `text` is an empty string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if text == "":
        raise ValueError("text must not be empty")

    # 1) Normalize line endings to LF so subsequent regexes are predictable.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Collapse horizontal whitespace (spaces/tabs) but preserve newlines.
    #    Using [^\S\n]+ matches whitespace except newline.
    text = _HORZ_WS_RE.sub(" ", text)

    # 3) Remove spaces/tabs directly after or before newlines.
    text = _NEWLINE_LEADING_WS_RE.sub("\n", text)
    text = _NEWLINE_TRAILING_WS_RE.sub("\n", text)

    # 4) Collapse multiple consecutive newlines into a single newline.
    text = _MULTI_NEWLINE_RE.sub("\n", text)

    # Final trim of leading/trailing whitespace/newlines.
    return text.strip()


def parse_html(
        url: str,
        html: str,
        keep_links_as_markup: bool = True,
        ) -> Annotated[ParsedHTMLScrape, "raises ValueError"]:
    """
    Parse HTML content from `html` and extract meaningful text.

    Args:
        urL: The URL of the HTML content (for reference).
        html: The input HTML string to parse.
        keep_links_as_markup: Whether to preserve links in markdown format.

    Returns:
        The extracted plain text string and the title.

    Raises:
        TypeError: If `html` is not a string or a sequence of strings.
    """

    if not isinstance(html, str):
        raise TypeError("html must be a str")
    
    if html == "":
        return ParsedHTMLScrape(url, title="", text="")

    soup = BeautifulSoup(html, "html5lib")

    # remove script and style elements
    for script in soup(["script", "style", "noscript"]):
        script.decompose()

    # Remove meta, link, and other non-content elements
    for element in soup(["meta", "link", "svg", "path"]):
        element.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # convert links to markdown format if requested
    if keep_links_as_markup:
        for a in soup.find_all("a"):
            href = a.get("href")                                                    # type: ignore
            if href:
                # Use link text as the label; keep it safe by stripping.
                label = a.get_text().strip()
                a.replace_with(f"[{label}]({href})")                                # type: ignore

    # Get text
    text: str = soup.get_text()
    title: str = soup.title.string if soup.title and soup.title.string else ""

    # Remove any JavaScript artifacts that leaked through
    text = re.sub(r'css:\s*"[^"]*".*?(?=\n[A-Z\[]|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'js:\s*"[^"]*".*?(?=\n[A-Z\[]|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?:cacheStatusHP|reqObjData|pageType):[^}]*}', '', text, flags=re.DOTALL)
    
    # Extract text and normalize newlines/whitespace consistently.
    return ParsedHTMLScrape(url, title=title, text=_normalize_newlines(text))

