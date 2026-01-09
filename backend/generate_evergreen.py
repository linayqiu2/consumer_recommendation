#!/usr/bin/env python3
"""
CLI script to generate evergreen articles.

Usage:
    # Generate single article (topic is required)
    python generate_evergreen.py "Best mirrorless cameras for beginners" --topic Cameras

    # Generate multiple from file (file format: topic | query)
    python generate_evergreen.py --file queries.txt

    # List existing evergreen articles
    python generate_evergreen.py --list

    # Specify max videos to analyze
    python generate_evergreen.py "Best noise-cancelling headphones" --topic Audio --max-videos 8
"""

import argparse
import requests
import sys
from typing import Optional

DEFAULT_API_URL = "http://localhost:8000"

# Will be set by main() based on command line args
_api_url = DEFAULT_API_URL


def generate_article(query: str, topic: str, max_videos: int = 10) -> dict:
    """
    Generate an evergreen article from a query.

    Args:
        query: The product research query
        topic: The topic category (e.g., "EV", "Cameras", "Audio")
        max_videos: Maximum number of videos to analyze

    Returns:
        Response dict with article_id, title, content, etc.
    """
    print(f"\nGenerating article for: {query}")
    print(f"Topic: {topic}")
    print(f"Max videos: {max_videos}")
    print("-" * 50)

    try:
        response = requests.post(
            f"{_api_url}/api/articles/evergreen/generate",
            json={"query": query, "topic": topic, "max_videos": max_videos},
            timeout=600  # 10 minute timeout for long processing
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {_api_url}")
        print("Make sure the backend server is running.")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The article generation took too long.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: API returned error - {e.response.status_code}")
        try:
            detail = e.response.json().get("detail", str(e))
            print(f"Detail: {detail}")
        except:
            print(f"Response: {e.response.text[:500]}")
        sys.exit(1)


def list_evergreen_articles() -> None:
    """List all existing evergreen articles."""
    print("\nEvergreen Articles")
    print("=" * 60)

    try:
        response = requests.get(
            f"{_api_url}/api/articles/recent",
            params={"days": 365, "limit": 100, "article_type": "evergreen"}
        )
        response.raise_for_status()
        articles = response.json()

        if not articles:
            print("No evergreen articles found.")
            return

        for article in articles:
            article_id = article.get("id", "?")
            topic = article.get("topic", "?")
            title = article.get("article_title", "Untitled")[:45]
            created = article.get("created_at", "Unknown")[:10]
            print(f"ID: {article_id:4} | {topic:10} | {created} | {title}...")

        print(f"\nTotal: {len(articles)} evergreen articles")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {_api_url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: API returned error - {e.response.status_code}")
        sys.exit(1)


def generate_from_file(filepath: str, max_videos: int = 10) -> None:
    """
    Generate articles from a file of queries.

    File format: topic | query
    Example:
        # Comments start with #
        EV | Tesla Model Y Highland real-world range test
        Cameras | Recommend Hasselblad Cameras
    """
    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        sys.exit(1)

    if not lines:
        print("No queries found in file.")
        return

    # Parse lines into (topic, query) tuples
    entries = []
    for line in lines:
        if "|" not in line:
            print(f"Warning: Skipping invalid line (missing '|' separator): {line}")
            continue
        parts = line.split("|", 1)
        topic = parts[0].strip()
        query = parts[1].strip()
        if topic and query:
            entries.append((topic, query))
        else:
            print(f"Warning: Skipping invalid line (empty topic or query): {line}")

    if not entries:
        print("No valid entries found in file.")
        print("Expected format: topic | query")
        return

    print(f"\nFound {len(entries)} queries in {filepath}")
    print("=" * 60)

    results = []
    for i, (topic, query) in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] Processing: [{topic}] {query}")
        try:
            result = generate_article(query, topic, max_videos)
            results.append({
                "topic": topic,
                "query": query,
                "article_id": result["article_id"],
                "title": result["title"],
                "time": result["generation_time_seconds"]
            })
            print(f"  Generated: {result['title']} (ID: {result['article_id']})")
            print(f"  Time: {result['generation_time_seconds']:.1f}s")
        except SystemExit:
            print(f"  Failed to generate article for: {query}")
            results.append({
                "topic": topic,
                "query": query,
                "article_id": None,
                "title": "FAILED",
                "time": 0
            })

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    success = [r for r in results if r["article_id"]]
    failed = [r for r in results if not r["article_id"]]

    print(f"Successful: {len(success)}")
    for r in success:
        print(f"  ID {r['article_id']} [{r['topic']}]: {r['title'][:40]}... ({r['time']:.1f}s)")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  - [{r['topic']}] {r['query']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evergreen articles from product queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Best budget DSLR cameras" --topic Cameras
  %(prog)s "Sony WH-1000XM5 vs Bose QC Ultra" --topic Audio --max-videos 8
  %(prog)s --file queries.txt
  %(prog)s --list

File format for --file:
  # Comments start with #
  topic | query
  EV | Tesla Model Y Highland real-world range test
  Cameras | Recommend Hasselblad Cameras
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Query for article generation"
    )
    parser.add_argument(
        "--topic", "-t",
        help="Topic category for the article (e.g., 'EV', 'Cameras', 'Audio'). Required when using query."
    )
    parser.add_argument(
        "--file", "-f",
        help="File with queries (format: topic | query)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List existing evergreen articles"
    )
    parser.add_argument(
        "--max-videos", "-m",
        type=int,
        default=10,
        help="Maximum videos to analyze (default: 10)"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})"
    )

    args = parser.parse_args()

    # Update API URL if provided
    global _api_url
    _api_url = args.api_url

    if args.list:
        list_evergreen_articles()
    elif args.file:
        generate_from_file(args.file, args.max_videos)
    elif args.query:
        if not args.topic:
            print("Error: --topic is required when generating a single article.")
            print("Example: python generate_evergreen.py \"Best cameras\" --topic Cameras")
            sys.exit(1)
        result = generate_article(args.query, args.topic, args.max_videos)
        print(f"\nGenerated: {result['title']}")
        print(f"Article ID: {result['article_id']}")
        print(f"Topic: {args.topic}")
        print(f"Videos used: {len(result['videos'])}")
        print(f"Web sources: {len(result['web_sources'])}")
        print(f"Generation time: {result['generation_time_seconds']:.1f}s")
        print(f"\nContent preview ({len(result['content'])} chars):")
        print("-" * 50)
        print(result['content'][:500] + "...")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
