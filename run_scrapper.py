#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import json
from typing import List
from scraping import CustomWebLoader
from utils.string import ParsedHTMLScrape
from pathlib import Path


# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_pages_to_json(
    pages: list[ParsedHTMLScrape],
    output_path: str
) -> None:
    """
    Save scraped pages to a JSON file.
    """

    data = [
        {
            "url": page.url,
            "title": page.title,
            "text": page.text
        }
        for page in pages
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def example_rag_preparation():
    """
    Example 5: Complete RAG preparation workflow.
    This is the full workflow as described in the email.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete RAG Preparation Workflow")
    print("="*80 + "\n")
    
    loader = CustomWebLoader(
        max_concurrent_requests=10,
        timeout=15,
        rate_limit_delay=0.3
    )
    
    base_url = "https://www.losinj-hotels.com"
    
    try:
        # Step 1: Scrape the sitemap/homepage
        print("📥 Scraping sitemap...")
        sitemap_scrape: ParsedHTMLScrape = await loader.load_async(
            url=base_url,
            keep_links_as_markup=True
        )
        
        # Step 2: Extract relevant links
        print("🔗 Extracting links...")
        all_links: List[tuple[str, str]] = loader.extract_links(
            base_url=base_url,
            parsed_content=sitemap_scrape
        )
        
        for i, links in enumerate(all_links, 1):
            print(f"   Link {i}: {links}")
        print(f"   Found {len(all_links)} total links")
        
        # Step 3: Filter links for RAG (adjust patterns as needed)
        rag_patterns = [
            '/hotel',
            '/accommodation',
            '/restaurant',
            '/dining',
            '/rooms',
            '/suite',
            '/amenities',
            '/spa',
            '/wellness',
            '/holiday',
            '/destination',
            '/all-offers',
            '/best-rate-guarantee',
            '/terms-of-reservation',

        ]
        
        filtered_links: List[tuple[str, str]] = loader.extract_links(
            base_url=base_url,
            parsed_content=sitemap_scrape,
            allowed_list=rag_patterns
        )
        
        print(f"   Filtered to {len(filtered_links)} relevant links")
        
        # Step 4: Scrape all filtered pages
        print("📚 Scraping all pages...")
        urls_to_scrape = [url for title, url in filtered_links[:10]]  # Limit for demo
        
        scraped_pages: List[ParsedHTMLScrape] = await loader.load_async(
            url=urls_to_scrape,
            keep_links_as_markup=True
        )
        
        # Step 5: Process results for RAG
        print("\n✅ RAG Data Preparation Complete!\n")
        print(f"Total pages scraped: {len(scraped_pages)}")
        print(f"Total content size: {sum(len(p.text) for p in scraped_pages)} characters")
        
        print("\n📊 Content Summary:")
        print("-" * 80)
        for i, page in enumerate(scraped_pages, 1):
            print(f"{i}. {page.title}")
            print(f"   URL: {page.url}")
            print(f"   Size: {len(page.text)} chars")
            if page.text:
                print(f"   Preview: {page.text[:150]}...")
            print()
        
        # This is your RAG data ready to be processed
        output_file = "output/losinj_rag_pages.json"
        save_pages_to_json(scraped_pages, output_file)

        print(f"\n💾 Saved scraped pages to: {output_file}")

        # This is your RAG data ready to be processed
        return scraped_pages
        
    except Exception as e:
        logger.error(f"Error in RAG preparation: {e}")
        return []


async def main():
    """
    Main function to run all examples.
    """
    print("\n")
    print("="*80)
    print("WEB SCRAPING EXAMPLES")
    print("="*80)
    
    # Run examples (comment out the ones you don't want to run)
    
    # Example 1: Single page
    # await example_single_page_scrape()
    
    # Example 2: Sitemap scraping
    # await example_sitemap_scraping()
    
    # Example 3: Filtered extraction
    # await example_filtered_link_extraction()
    
    # Example 4: Bulk scraping
    # await example_bulk_scraping()
    
    # Example 5: Full RAG workflow
    await example_rag_preparation()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())