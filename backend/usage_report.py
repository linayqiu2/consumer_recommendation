#!/usr/bin/env python3
"""
Generate usage reports from tracking data.

Usage:
    python usage_report.py --days 7    # Last 7 days
    python usage_report.py --days 30   # Last 30 days (default)
    python usage_report.py --json      # Output as JSON
"""

import argparse
import json
import database as db


def print_report(days: int = 30, json_output: bool = False):
    """
    Generate and print a usage report.

    Args:
        days: Number of days to look back
        json_output: If True, output as JSON instead of formatted text
    """
    report = db.get_usage_report(days=days)

    if json_output:
        print(json.dumps(report, indent=2, default=str))
        return

    separator = "=" * 60

    print(f"\n{separator}")
    print(f"USAGE REPORT - Last {days} days")
    print(f"{separator}")

    # Totals
    totals = report.get('totals', {})
    total_cost = totals.get('total_cost', 0)
    total_input = totals.get('total_input_tokens', 0)
    total_output = totals.get('total_output_tokens', 0)
    total_requests = totals.get('total_requests', 0)

    print(f"\n## Summary")
    print(f"  Total Requests:      {total_requests:,}")
    print(f"  Total Input Tokens:  {total_input:,}")
    print(f"  Total Output Tokens: {total_output:,}")
    print(f"  Total Cost:          ${total_cost:.4f}")

    # Costs by model
    by_model = report.get('by_model', [])
    if by_model:
        print(f"\n## API Costs by Model")
        for m in by_model:
            model = m.get('model', 'unknown')
            cost = m.get('cost', 0)
            requests = m.get('requests', 0)
            input_tokens = m.get('input_tokens', 0)
            output_tokens = m.get('output_tokens', 0)
            print(f"  {model:20} ${cost:.4f} ({requests:,} requests, {input_tokens:,} in / {output_tokens:,} out)")

    # Costs by request type
    by_request_type = report.get('by_request_type', [])
    if by_request_type:
        print(f"\n## API Costs by Request Type")
        for rt in by_request_type:
            request_type = rt.get('request_type', 'unknown')
            cost = rt.get('cost', 0)
            requests = rt.get('requests', 0)
            print(f"  {request_type:20} ${cost:.4f} ({requests:,} requests)")

    # Top users by cost
    by_user = report.get('by_user', [])
    if by_user:
        print(f"\n## Top Users by Cost")
        for u in by_user[:10]:
            user_id = u.get('user_id')
            user_label = f"User {user_id}" if user_id else "Anonymous"
            cost = u.get('cost', 0)
            requests = u.get('requests', 0)
            print(f"  {user_label:20} ${cost:.4f} ({requests:,} requests)")

    # Event counts
    events = report.get('events', [])
    if events:
        print(f"\n## User Events")
        for e in events:
            event_type = e.get('event_type', 'unknown')
            count = e.get('count', 0)
            print(f"  {event_type:20} {count:,} events")

    print(f"\n{separator}\n")


def print_user_summary(user_id: int, days: int = 30):
    """Print usage summary for a specific user."""
    summary = db.get_user_usage_summary(user_id=user_id, days=days)

    print(f"\n## Usage Summary for User {user_id} (Last {days} days)")
    print(f"  Total Requests:      {summary.get('total_requests', 0):,}")
    print(f"  Total Input Tokens:  {summary.get('total_input_tokens', 0):,}")
    print(f"  Total Output Tokens: {summary.get('total_output_tokens', 0):,}")
    print(f"  Total Cost:          ${summary.get('total_cost', 0):.4f}")


def print_recent_events(event_type: str = None, days: int = 7, limit: int = 20):
    """Print recent user events."""
    events = db.get_user_events(event_type=event_type, days=days, limit=limit)

    event_filter = f" (type: {event_type})" if event_type else ""
    print(f"\n## Recent Events{event_filter} (Last {days} days, limit {limit})")

    if not events:
        print("  No events found.")
        return

    for e in events:
        event_type_str = e.get('event_type', 'unknown')
        created_at = e.get('created_at', '')
        event_data = e.get('event_data', {})

        # Format event data nicely
        data_summary = ""
        if event_type_str == 'article_view':
            data_summary = f"Article: {event_data.get('article_title', 'N/A')[:40]}"
        elif event_type_str == 'video_click':
            data_summary = f"Video: {event_data.get('video_title', 'N/A')[:40]}"
        elif event_type_str == 'query':
            data_summary = f"Query: {event_data.get('query', 'N/A')[:40]}"
        else:
            data_summary = json.dumps(event_data)[:60]

        print(f"  [{created_at}] {event_type_str:15} {data_summary}")


def main():
    parser = argparse.ArgumentParser(description="Generate usage reports from tracking data")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back (default: 30)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--user", type=int, help="Get summary for specific user ID")
    parser.add_argument("--events", action="store_true", help="Show recent events")
    parser.add_argument("--event-type", type=str, help="Filter events by type (article_view, video_click, query)")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of events (default: 20)")

    args = parser.parse_args()

    if args.user:
        print_user_summary(user_id=args.user, days=args.days)
    elif args.events:
        print_recent_events(event_type=args.event_type, days=args.days, limit=args.limit)
    else:
        print_report(days=args.days, json_output=args.json)


if __name__ == "__main__":
    main()
