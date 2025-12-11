# -*- coding: utf-8 -*-
"""
This file contains class for scraping web page.

Implemented classes:
    - CustomWebLoader

All functions propagate exceptions to the caller.

"""

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

import re
from typing import List, Optional, Sequence, Union
from urllib.parse import urljoin

from utils.string import ParsedHTMLScrape, parse_html


__all__ = ["CustomWebLoader"]


class CustomWebLoader:
    """
    Asynchronous web loader using asyncio for true concurrent I/O operations.

    This implementation scrapes the web page and returns the content in
    the plain text format with links preserved in the markdown format.

    Mechanism:
        - Uses aiohttp for asynchronous HTTP requests. For larger number of webpages this approach scales better than
          using process executor since the process executor will be limited by the number of CPU cores.
             # Thread 1: [Request]----[Waiting for response (GIL released)]----[Parse]
             # Thread 2:     [Request]----[Waiting for response (GIL released)]----[Parse]
             # Thread 3:         [Request]----[Waiting for response (GIL released)]----[Parse]
             # Time:     ----→----→----→----→----→----→----→----→----→----→----→----→

        - Parses HTML content using utils.string.parse_html in a thread pool.

    """

    def __init__(
        self,
        max_concurrent_requests: int = 20,
        timeout: int = 10,
        rate_limit_delay: float = 0.1,
        user_agent: str | None = None,
    ) -> None:
        """
        Initialise the web loader.

        Args:
            max_concurrent_requests: Maximum number of concurrent requests.
            timeout: Timeout for each request in seconds.
            rate_limit_delay: Delay between requests to avoid overwhelming the server.
            user_agent: Custom User-Agent string for requests.
        """

        self._max_concurrent_requests = max_concurrent_requests
        self._timeout = ClientTimeout(total=timeout)
        self._rate_limit_delay = rate_limit_delay
        self._user_agent = user_agent or "Mozilla/5.0 (compatible; AsyncWebLoader/1.0)"
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Initialise the logger
        self._logger = logging.getLogger(__name__)

        self._logger.debug(
            "CustomWebLoader initialized with max_concurrent_requests=%d, timeout=%d, rate_limit_delay=%.2f",
            self._max_concurrent_requests,
            self._timeout,
            self._rate_limit_delay,
        )

    async def _fetch_single_url(
        self, session: ClientSession, url: str
    ) -> tuple[str, Optional[str], Optional[Exception]]:
        """
        Fetch a single URL asynchronously.

        Args:
            session: The aiohttp ClientSession to use for the request.
            url: The URL to fetch.

        Returns:
            A tuple containing the URL, the fetched content as a string (or None if an error occurred),
            and an exception (or None if the request was successful).
        """

        async with self._semaphore:
            try:
                # Apply rate limiting
                await asyncio.sleep(self._rate_limit_delay)

                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    self._logger.debug("Fetched URL: %s", url)
                    return url, html, None

            except asyncio.TimeoutError as e:
                self._logger.error(f"Timeout error for {url}: {str(e)}")
                return url, None, e
            except ClientError as e:
                self._logger.error(f"Client error for {url}: {str(e)}")
                return url, None, e
            except Exception as e:
                self._logger.error(f"Unexpected error for {url}: {str(e)}")
                return url, None, e

    async def _fetch_all_urls(self, urls: List[str]) -> List[tuple[str, str]]:
        """
        Fetch multiple URLs asynchronously.

        Args:
            urls: A list of URLs to fetch.

        Returns:
            A list of tuples containing the URL and the fetched content as a string.
        """

        connector = TCPConnector(limit=self._max_concurrent_requests)
        headers = {"User-Agent": self._user_agent}

        async with ClientSession(
            connector=connector, timeout=self._timeout, headers=headers
        ) as session:
            # prepare tasks
            tasks = [self._fetch_single_url(session, url) for url in urls]

            # execute tasks concurrently
            results = await asyncio.gather(*tasks)

            # process results
            html_contents: list[tuple[str, str]] = []
            for url, content, error in results:
                if error:
                    self._logger.warning("Failed to fetch %s: %s", url, str(error))
                    html_contents.append((url, ""))
                else:
                    html_contents.append((url, content or ""))

            return html_contents

    async def load_async(
        self, url: Union[str, Sequence[str]], keep_links_as_markup: bool = True
    ) -> Union[ParsedHTMLScrape, List[ParsedHTMLScrape]]:
        """
        Asynchronously load web page(s) from the given URL(s).

        Args:
            url: A single URL string or a sequence of URL strings to load.
            keep_links_as_markup: Whether to preserve links in markdown format.

        Returns:
            A single tuple of (text, title) if a single URL is provided,
            or a list of such tuples for multiple URLs.
        """

        if isinstance(url, str):
            urls = [url]
            single_url = True
        else:
            urls = list(url)
            single_url = False

        html_contents = await self._fetch_all_urls(urls)

        # Parse HTML content to extract meaningful text
        # We will use the ThreadPoolExecutor to avoid blocking the event loop

        async def parse_single(
            url: str, html: str, executor: ThreadPoolExecutor
        ) -> ParsedHTMLScrape:
            if html:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    executor, parse_html, url, html, keep_links_as_markup
                )
            else:
                return ParsedHTMLScrape(url=url, title="", text="")

        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = []
            for content in html_contents:
                (url, html) = content
                tasks.append(parse_single(url, html, executor))

            parsed_contents = await asyncio.gather(*tasks)

        return parsed_contents if not single_url else parsed_contents[0]

    def extract_links(
        self,
        base_url: str | None,
        parsed_content: ParsedHTMLScrape,
        allowed_list: List[str] | None = None,
    ) -> List[tuple[str, str]]:
        """
        Extract links from the parsed content.

        Args:
            base_url: The base URL to use for relative links.
            parsed_content: The parsed content to extract links from.

        Returns:
            A list of tuples (title, url) extracted from the parsed content.
        """

        result: List[tuple[str, str]] = []

        # Regular expression to match lines with links in format [name](url)
        link_pattern = r"\[(.*?)\]\((.*?)\)"

        lines: List[str] = parsed_content.text.split(sep="\n")
        for line in lines:
            if (
                not line
                or "[" not in line
                or "]" not in line
                or "(" not in line
                or ")" not in line
            ):
                continue

            match = re.match(pattern=link_pattern, string=line)
            if match:
                title: str = match.group(1)
                url: str = match.group(2)

                if title and url:
                    should_add = True
                    if allowed_list:
                        should_add = any(
                            allowed_url in url for allowed_url in allowed_list
                        )

                    if should_add:
                        if base_url and not (
                            url.startswith("http://") or url.startswith("https://")
                        ):
                            url = urljoin(base_url, url)

                        result.append((title, url))

        return result
