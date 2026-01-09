#!/usr/bin/env python3
"""
Test script for generate_articles_from_trending_content function.
Generates articles and prints them from the database.
"""

from main import generate_articles_from_trending_content
import database as db
import json


def test_generate_articles():
    print("=" * 60)
    print("TESTING: generate_articles_from_trending_content")
    print("=" * 60)

    # Run with minimal settings for faster testing
    # Uses predefined YouTube Topic IDs for precise discovery
    result = generate_articles_from_trending_content(
        topics=["Electronics","EV","Beauty","Cameras"],  # Uses YouTube Topic IDs
        timeframe="week",
        max_videos_per_topic=20,
        country="US",
        parallel_articles=2,
        parallel_topics=True,
        skip_synthesis=False,  # Skip for faster testing
        save_to_db=True
    )

    print("\n" + "=" * 60)
    print("GENERATION RESULTS")
    print("=" * 60)
    print(f"Batch Run ID: {result.get('batch_run_id')}")
    print(f"Total articles: {result['summary']['total_articles']}")
    print(f"Total errors: {result['summary']['total_errors']}")
    print(f"Total time: {result['summary']['total_time_seconds']:.2f}s")

    if result['errors']:
        print("\nErrors:")
        for err in result['errors']:
            print(f"  - {err.get('topic')}: {err.get('error')}")

    return result.get('batch_run_id')


def print_articles_from_db(batch_run_id=None):
    print("\n" + "=" * 60)
    print("ARTICLES FROM DATABASE")
    print("=" * 60)

    # Get recent articles
    articles = db.get_recent_articles(days=1, limit=10)

    if not articles:
        print("No articles found in database.")
        return

    print(f"Found {len(articles)} recent articles\n")

    for i, article in enumerate(articles, 1):
        print(f"\n{'─' * 60}")
        print(f"ARTICLE {i} (ID: {article['id']})")
        print(f"{'─' * 60}")
        print(f"Topic: {article['topic']}")
        print(f"Article Title: {article['article_title']}")
        print(f"Created: {article['created_at']}")
        print(f"\n--- CONTENT ---\n")

        # Print article content (truncated if too long)
        content = article.get('article_content', '')
        if len(content) > 2000:
            print(content[:2000])
            print(f"\n... [truncated, {len(content)} total characters]")
        else:
            print(content)

        # Get full article with videos
        full_article = db.get_article(article['id'])
        if full_article:
            videos = full_article.get('videos', [])
            web_sources = full_article.get('web_sources', [])
            print(f"\n--- SOURCES ---")
            print(f"Videos used: {len(videos)}")
            for v in videos[:3]:
                print(f"  - {v.get('title', 'Unknown')[:50]}... ({v.get('channel')})")
            if len(videos) > 3:
                print(f"  ... and {len(videos) - 3} more")

            print(f"Web sources: {len(web_sources)}")
            for w in web_sources[:3]:
                print(f"  - {w.get('title', 'Unknown')[:50]}...")


def print_db_statistics():
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    stats = db.get_statistics()
    print(f"Total articles: {stats['total_articles']}")
    print(f"Total videos referenced: {stats['total_videos']}")
    print(f"Total web sources: {stats['total_web_sources']}")
    print(f"Total batch runs: {stats['total_batch_runs']}")
    print(f"Articles in last 7 days: {stats['articles_last_7_days']}")

    if stats['articles_by_topic']:
        print("\nArticles by topic:")
        for topic, count in stats['articles_by_topic'].items():
            print(f"  - {topic}: {count}")


def print_batch_runs():
    print("\n" + "=" * 60)
    print("RECENT BATCH RUNS")
    print("=" * 60)

    runs = db.get_batch_runs(limit=5)

    if not runs:
        print("No batch runs found.")
        return

    for run in runs:
        print(f"\nBatch Run ID: {run['id']}")
        print(f"  Started: {run['started_at']}")
        print(f"  Completed: {run['completed_at']}")
        print(f"  Status: {run['status']}")
        print(f"  Articles: {run['total_articles']}, Errors: {run['total_errors']}")
        print(f"  Time: {run['total_time_seconds']:.2f}s" if run['total_time_seconds'] else "  Time: N/A")
        print(f"  Topics: {run['topics']}")


def print_topics():
    print("\n" + "=" * 60)
    print("RECENT TOPICS")
    print("=" * 60)

    topics = db.get_recent_topics(days=7, limit=20)

    if not topics:
        print("No topics found in database.")
        return

    print(f"Found {len(topics)} recent topics\n")

    for i, topic in enumerate(topics, 1):
        print(f"\n{'─' * 60}")
        print(f"TOPIC {i} (ID: {topic['id']})")
        print(f"{'─' * 60}")
        print(f"Name: {topic['topic_name']}")
        print(f"Timeframe: {topic['timeframe']}")
        print(f"Country: {topic['country']}")
        print(f"Created: {topic['created_at']}")

        topic_videos = topic.get('topic_videos', [])
        web_data = topic.get('web_data', [])

        print(f"\n--- TOPIC VIDEOS ({len(topic_videos)}) ---")
        for v in topic_videos[:3]:
            view_count = v.get('view_count', 0)
            print(f"  - [{view_count:,} views] {v.get('title', 'Unknown')[:50]}...")
        if len(topic_videos) > 3:
            print(f"  ... and {len(topic_videos) - 3} more")

        print(f"\n--- WEB DATA ({len(web_data)}) ---")
        for w in web_data[:3]:
            print(f"  - {w.get('title', 'Unknown')[:60]}...")
        if len(web_data) > 3:
            print(f"  ... and {len(web_data) - 3} more")

        if topic.get('description'):
            print(f"\n--- DESCRIPTION ---")
            print(f"{topic['description'][:300]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "generate":
            # Generate new articles
            batch_id = test_generate_articles()
            print_articles_from_db(batch_id)

        elif command == "list":
            # Just list articles from database
            print_articles_from_db()

        elif command == "stats":
            # Show database statistics
            print_db_statistics()

        elif command == "runs":
            # Show batch runs
            print_batch_runs()

        elif command == "topics":
            # Show topics with video IDs and web data
            print_topics()

        elif command == "all":
            # Show everything
            print_db_statistics()
            print_batch_runs()
            print_topics()
            print_articles_from_db()

        else:
            print(f"Unknown command: {command}")
            print("Usage: python test_article_generation.py [generate|list|stats|runs|topics|all]")
    else:
        # Default: generate and print
        print("Usage:")
        print("  python test_article_generation.py generate  - Generate new articles")
        print("  python test_article_generation.py list      - List articles from DB")
        print("  python test_article_generation.py stats     - Show DB statistics")
        print("  python test_article_generation.py runs      - Show batch runs")
        print("  python test_article_generation.py topics    - Show trending topics")
        print("  python test_article_generation.py all       - Show all DB info")
        print("\nRunning 'generate' by default...\n")

        batch_id = test_generate_articles()
        print_articles_from_db(batch_id)
        print_db_statistics()
