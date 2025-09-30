#!/usr/bin/env python3
"""
Enhanced web crawler for building text corpora from websites.

This crawler performs breadth-first traversal of a website, extracting
text content and saving it locally for indexing or analysis.
"""

import requests
import os
import re
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from collections import deque
from typing import Set, Optional, Dict, Tuple
from datetime import datetime
import hashlib
import json

# Configuration
DEFAULT_OUTPUT_DIR = "corpus"
DEFAULT_USER_AGENT = "mini-search-bot/1.0 (educational-purposes)"
DEFAULT_MAX_PAGES = 50
DEFAULT_DELAY = 1.0  # Seconds between requests
DEFAULT_TIMEOUT = 10  # Request timeout in seconds
MAX_RETRIES = 2
MAX_URL_LENGTH = 200

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebCrawler:
    """
    A polite web crawler that extracts text content from websites.
    """

    def __init__(self,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 user_agent: str = DEFAULT_USER_AGENT,
                 delay: float = DEFAULT_DELAY,
                 timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the web crawler.

        Args:
            output_dir: Directory to save extracted text files
            user_agent: User agent string for HTTP requests
            delay: Delay in seconds between requests (be polite!)
            timeout: Request timeout in seconds
        """
        self.output_dir = output_dir
        self.delay = delay
        self.timeout = timeout
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.url_to_file: Dict[str, str] = {}
        self.stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'total_bytes': 0,
            'start_time': None,
            'end_time': None
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup session with custom headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def slugify_url(self, url: str) -> str:
        """
        Convert URL to a safe filename.

        Args:
            url: URL to convert

        Returns:
            Safe filename string
        """
        # Parse URL and create a meaningful filename
        parsed = urlparse(url)

        # Use path and query for filename uniqueness
        path_part = parsed.path.strip('/').replace('/', '_')
        query_part = parsed.query[:50] if parsed.query else ''

        # Create base filename
        if path_part:
            base_name = path_part
        else:
            base_name = 'index'

        # Add query hash if present (for unique URLs with same path)
        if query_part:
            query_hash = hashlib.md5(parsed.query.encode()).hexdigest()[:8]
            base_name = f"{base_name}_{query_hash}"

        # Clean and limit filename
        safe_name = re.sub(r'[^0-9a-zA-Z_-]+', '_', base_name)
        safe_name = safe_name.strip('_')[:MAX_URL_LENGTH]

        return f"{safe_name}.txt"

    def extract_text(self, html: str, url: str) -> str:
        """
        Extract clean text from HTML content.

        Args:
            html: Raw HTML content
            url: Source URL (for metadata)

        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer',
                             'aside', 'iframe', 'noscript', 'svg']):
            element.decompose()

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, str) and '<!--' in text):
            comment.extract()

        # Extract text with proper spacing
        text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = ' '.join(text.split())

        # Add metadata header
        metadata = f"# Source: {url}\n# Crawled: {datetime.now().isoformat()}\n\n"

        return metadata + text

    def is_valid_url(self, url: str, domain: str) -> bool:
        """
        Check if URL is valid for crawling.

        Args:
            url: URL to validate
            domain: Allowed domain

        Returns:
            True if URL should be crawled
        """
        try:
            parsed = urlparse(url)

            # Check domain
            if parsed.netloc != domain:
                return False

            # Skip common non-HTML resources
            path = parsed.path.lower()
            skip_extensions = (
                '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip',
                '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
                '.exe', '.dmg', '.pkg', '.deb', '.rpm'
            )
            if any(path.endswith(ext) for ext in skip_extensions):
                return False

            # Skip common utility pages
            skip_patterns = [
                '/login', '/logout', '/signin', '/signup',
                '/download', '/print', '/share', '/email'
            ]
            if any(pattern in path for pattern in skip_patterns):
                return False

            return True

        except Exception:
            return False

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for consistency.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        # Remove fragment
        url = url.split('#')[0]

        # Remove trailing slashes
        url = url.rstrip('/')

        # Decode URL-encoded characters
        url = unquote(url)

        # Remove common tracking parameters
        tracking_params = ['utm_source', 'utm_medium', 'utm_campaign',
                           'utm_term', 'utm_content', 'fbclid', 'gclid']
        parsed = urlparse(url)
        if parsed.query:
            params = []
            for param in parsed.query.split('&'):
                if not any(tracking in param for tracking in tracking_params):
                    params.append(param)
            if params:
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{'&'.join(params)}"
            else:
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        return url

    def fetch_page(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Fetch a web page with retries.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (HTML content, final URL after redirects) or None
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    verify=True
                )

                # Check status code
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    return None

                # Check content type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    logger.debug(f"Skipping non-HTML content: {content_type}")
                    return None

                # Check content size (skip very large pages)
                if len(response.content) > 5 * 1024 * 1024:  # 5MB limit
                    logger.warning(f"Page too large: {len(response.content)} bytes")
                    return None

                self.stats['total_bytes'] += len(response.content)
                return response.text, response.url

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def discover_links(self, html: str, base_url: str, domain: str) -> Set[str]:
        """
        Extract and validate links from HTML.

        Args:
            html: HTML content
            base_url: Base URL for relative links
            domain: Allowed domain

        Returns:
            Set of valid URLs to crawl
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = set()

        for anchor in soup.find_all('a', href=True):
            href = anchor['href'].strip()

            # Skip empty or JavaScript links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue

            # Resolve relative URLs
            full_url = urljoin(base_url, href)

            # Normalize URL
            full_url = self.normalize_url(full_url)

            # Validate URL
            if self.is_valid_url(full_url, domain):
                links.add(full_url)

        return links

    def crawl(self,
              seed_url: str,
              max_pages: int = DEFAULT_MAX_PAGES,
              allowed_paths: Optional[list] = None) -> Dict:
        """
        Crawl website starting from seed URL.

        Args:
            seed_url: Starting URL
            max_pages: Maximum number of pages to crawl
            allowed_paths: Optional list of path prefixes to restrict crawling

        Returns:
            Dictionary of crawl statistics
        """
        logger.info(f"Starting crawl from {seed_url}")
        self.stats['start_time'] = datetime.now()

        # Parse seed URL
        parsed_seed = urlparse(seed_url)
        domain = parsed_seed.netloc

        # Initialize queue with seed URL
        queue = deque([self.normalize_url(seed_url)])

        while queue and len(self.visited_urls) < max_pages:
            url = queue.popleft()

            # Skip if already visited or failed
            if url in self.visited_urls or url in self.failed_urls:
                continue

            # Check allowed paths restriction
            if allowed_paths:
                parsed = urlparse(url)
                if not any(parsed.path.startswith(path) for path in allowed_paths):
                    continue

            logger.info(f"Crawling [{len(self.visited_urls) + 1}/{max_pages}]: {url}")

            # Fetch page
            result = self.fetch_page(url)
            if not result:
                self.failed_urls.add(url)
                self.stats['pages_failed'] += 1
                continue

            html, final_url = result

            # Extract and save text
            try:
                text = self.extract_text(html, final_url)
                filename = self.slugify_url(final_url)
                filepath = os.path.join(self.output_dir, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)

                self.visited_urls.add(url)
                self.url_to_file[final_url] = filename
                self.stats['pages_crawled'] += 1

                logger.info(f"  ✓ Saved {len(text)} chars to {filename}")

            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                self.failed_urls.add(url)
                self.stats['pages_failed'] += 1
                continue

            # Discover new links
            new_links = self.discover_links(html, final_url, domain)
            for link in new_links:
                if link not in self.visited_urls and link not in queue:
                    queue.append(link)

            logger.debug(f"  Found {len(new_links)} links, queue size: {len(queue)}")

            # Be polite - wait between requests
            time.sleep(self.delay)

        self.stats['end_time'] = datetime.now()

        # Save crawl metadata
        self.save_metadata()

        return self.stats

    def save_metadata(self):
        """Save crawl metadata and URL mappings."""
        metadata = {
            'stats': self.stats,
            'url_to_file': self.url_to_file,
            'visited_urls': list(self.visited_urls),
            'failed_urls': list(self.failed_urls)
        }

        # Convert datetime objects to strings
        if metadata['stats']['start_time']:
            metadata['stats']['start_time'] = metadata['stats']['start_time'].isoformat()
        if metadata['stats']['end_time']:
            metadata['stats']['end_time'] = metadata['stats']['end_time'].isoformat()

        metadata_file = os.path.join(self.output_dir, 'crawl_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved crawl metadata to {metadata_file}")

    def print_summary(self):
        """Print crawl summary statistics."""
        if not self.stats['start_time']:
            print("No crawl has been performed yet.")
            return

        duration = (self.stats['end_time'] - datetime.fromisoformat(self.stats['end_time'])).total_seconds()

        print("\n" + "=" * 50)
        print("CRAWL SUMMARY")
        print("=" * 50)
        print(f"Pages crawled:    {self.stats['pages_crawled']}")
        print(f"Pages failed:     {self.stats['pages_failed']}")
        print(f"Total data:       {self.stats['total_bytes'] / 1024:.2f} KB")
        print(f"Duration:         {duration:.1f} seconds")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Web crawler for building text corpora'
    )
    parser.add_argument('url', help='Starting URL to crawl')
    parser.add_argument(
        '-n', '--max-pages',
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f'Maximum pages to crawl (default: {DEFAULT_MAX_PAGES})'
    )
    parser.add_argument(
        '-o', '--output',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=DEFAULT_DELAY,
        help=f'Delay between requests in seconds (default: {DEFAULT_DELAY})'
    )
    parser.add_argument(
        '--paths',
        nargs='*',
        help='Allowed path prefixes (e.g., /blog /docs)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create crawler
    crawler = WebCrawler(
        output_dir=args.output,
        delay=args.delay
    )

    # Run crawl
    try:
        stats = crawler.crawl(
            seed_url=args.url,
            max_pages=args.max_pages,
            allowed_paths=args.paths
        )
        crawler.print_summary()

    except KeyboardInterrupt:
        logger.info("\nCrawl interrupted by user")
        crawler.stats['end_time'] = datetime.now()
        crawler.save_metadata()
        crawler.print_summary()
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        raise


if __name__ == "__main__":
    main()
