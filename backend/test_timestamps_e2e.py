#!/usr/bin/env python3
"""
End-to-end test for timestamp links in chat responses.

This test verifies that:
1. Timestamps are correctly converted from [MM:SS] to URL parameters (&t=seconds)
2. Links point to actual YouTube URLs (not homepage # or /)
3. Channel names are clickable links
4. The markdown format renders correctly

Run with: python test_timestamps_e2e.py
"""

import sys
import re
import json
import requests
from typing import List, Tuple

# Configuration
API_URL = "http://localhost:8000"  # Change to deployed URL for production testing
TEST_QUERY = "best noise cancelling headphones"

def test_utility_functions():
    """Test the timestamp utility functions directly."""
    print("\n" + "="*60)
    print("TEST 1: Utility Functions")
    print("="*60)

    from main import parse_timestamp_to_seconds, build_timestamped_youtube_link, build_quote_markdown

    # Test parse_timestamp_to_seconds
    test_cases = [
        ("[01:23]", 83),
        ("01:23", 83),
        ("[11:21]", 681),
        ("[04:07]", 247),
        ("[00:00]", 0),
        ("[1:23:45]", 5025),
        ("invalid", 0),
        ("", 0),
    ]

    all_passed = True
    for ts, expected in test_cases:
        result = parse_timestamp_to_seconds(ts)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {status}: parse_timestamp_to_seconds('{ts}') = {result} (expected {expected})")

    # Test build_timestamped_youtube_link
    url = "https://youtube.com/watch?v=abc123"

    link1 = build_timestamped_youtube_link(url, "[04:07]")
    expected1 = "https://youtube.com/watch?v=abc123&t=247"
    status1 = "PASS" if link1 == expected1 else "FAIL"
    if status1 == "FAIL":
        all_passed = False
    print(f"  {status1}: build_timestamped_youtube_link with [04:07] = {link1}")

    link2 = build_timestamped_youtube_link(url, "[00:00]")
    status2 = "PASS" if link2 == url else "FAIL"
    if status2 == "FAIL":
        all_passed = False
    print(f"  {status2}: build_timestamped_youtube_link with [00:00] returns original URL")

    # Test build_quote_markdown
    md = build_quote_markdown("Great sound quality", "MKBHD", url, "[04:07]")
    expected_md = '*"Great sound quality"* — [MKBHD at [04:07]](https://youtube.com/watch?v=abc123&t=247)'
    status_md = "PASS" if md == expected_md else "FAIL"
    if status_md == "FAIL":
        all_passed = False
        print(f"  {status_md}: build_quote_markdown produces correct format")
        print(f"    Expected: {expected_md}")
        print(f"    Got: {md}")
    else:
        print(f"  {status_md}: build_quote_markdown produces correct format")
        print(f"    Result: {md}")

    return all_passed


def test_transform_function():
    """Test the transform_insights_for_answer_generation function."""
    print("\n" + "="*60)
    print("TEST 2: Transform Function")
    print("="*60)

    from main import transform_insights_for_answer_generation

    # Create test video insights
    video_insights = [{
        "channel": "TechReviewer",
        "url": "https://youtube.com/watch?v=test123",
        "products": [{
            "name": "Sony WH-1000XM5",
            "top_quotes": [
                {"text": "The noise cancellation is incredible", "timestamp": "[02:30]"},
                {"text": "Battery life exceeded expectations", "timestamp": "[08:45]"}
            ]
        }],
        "emotional_highlights": [
            {"moment": "Wow reaction to ANC", "timestamp": "[03:15]"}
        ]
    }]

    synthesis = {
        "products": [{
            "name": "Sony WH-1000XM5",
            "notable_quotes": [
                {"text": "Best in class comfort", "timestamp": "[05:20]", "channel": "AudioExpert", "url": "https://youtube.com/watch?v=other456"}
            ]
        }]
    }

    transformed_insights, transformed_synthesis = transform_insights_for_answer_generation(video_insights, synthesis)

    all_passed = True

    # Check video insights transformation
    quote1 = transformed_insights[0]["products"][0]["top_quotes"][0]

    # Check timestamp_seconds
    expected_seconds = 150  # 2:30 = 150 seconds
    status1 = "PASS" if quote1.get("timestamp_seconds") == expected_seconds else "FAIL"
    if status1 == "FAIL":
        all_passed = False
    print(f"  {status1}: timestamp_seconds = {quote1.get('timestamp_seconds')} (expected {expected_seconds})")

    # Check linked_url contains &t=
    linked_url = quote1.get("linked_url", "")
    status2 = "PASS" if "&t=150" in linked_url else "FAIL"
    if status2 == "FAIL":
        all_passed = False
    print(f"  {status2}: linked_url contains &t=150: {linked_url}")

    # Check markdown_link format - should include timestamp in link text
    md_link = quote1.get("markdown_link", "")
    status3 = "PASS" if "[TechReviewer at [02:30]](" in md_link and "&t=150)" in md_link else "FAIL"
    if status3 == "FAIL":
        all_passed = False
    print(f"  {status3}: markdown_link has correct format with timestamp")
    print(f"    Result: {md_link}")

    # Check synthesis transformation - should include timestamp in link text
    synth_quote = transformed_synthesis["products"][0]["notable_quotes"][0]
    synth_md = synth_quote.get("markdown_link", "")
    status4 = "PASS" if "[AudioExpert at [05:20]](" in synth_md and "&t=320)" in synth_md else "FAIL"  # 5:20 = 320
    if status4 == "FAIL":
        all_passed = False
    print(f"  {status4}: synthesis quote markdown_link correct with timestamp")
    print(f"    Result: {synth_md}")

    return all_passed


def extract_links_from_response(response_text: str) -> List[Tuple[str, str]]:
    """Extract all markdown links from response text."""
    # Pattern: [text](url)
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(pattern, response_text)


def validate_youtube_timestamp_links(links: List[Tuple[str, str]]) -> dict:
    """
    Validate that links are proper YouTube timestamp links.

    Returns dict with:
    - total_links: number of links found
    - youtube_links: number of YouTube links
    - links_with_timestamp: number with &t= parameter
    - homepage_links: number of bad homepage links (# or /)
    - issues: list of issues found
    """
    result = {
        "total_links": len(links),
        "youtube_links": 0,
        "links_with_timestamp": 0,
        "homepage_links": 0,
        "issues": []
    }

    for text, url in links:
        # Check for homepage links
        if url in ("#", "/", ""):
            result["homepage_links"] += 1
            result["issues"].append(f"Homepage link found: [{text}]({url})")
            continue

        # Check for YouTube links
        if "youtube.com" in url or "youtu.be" in url:
            result["youtube_links"] += 1

            # Check for timestamp parameter
            if "&t=" in url or "?t=" in url:
                result["links_with_timestamp"] += 1

                # Validate timestamp is a number
                t_match = re.search(r'[&?]t=(\d+)', url)
                if t_match:
                    timestamp = int(t_match.group(1))
                    if timestamp == 0:
                        result["issues"].append(f"Zero timestamp in link: [{text}]({url})")
                else:
                    result["issues"].append(f"Invalid timestamp format: [{text}]({url})")

    return result


def test_chat_endpoint():
    """Test the actual chat endpoint to verify timestamps in response."""
    print("\n" + "="*60)
    print("TEST 3: Chat Endpoint (End-to-End)")
    print("="*60)

    try:
        # First check if server is running
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"  SKIP: Server not healthy at {API_URL}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  SKIP: Cannot connect to server at {API_URL}")
        print(f"  To run this test, start the server with: uvicorn main:app --reload")
        return None

    print(f"  Server is running at {API_URL}")
    print(f"  Testing with query: '{TEST_QUERY}'")
    print("  This may take a minute...")

    # Make request to chat endpoint
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "query": TEST_QUERY,
                "max_videos": 3
            },
            timeout=120
        )

        if response.status_code != 200:
            print(f"  FAIL: Chat endpoint returned {response.status_code}")
            return False

        data = response.json()
        answer = data.get("answer", "")

        print(f"\n  Response length: {len(answer)} characters")

        # Extract and validate links
        links = extract_links_from_response(answer)
        print(f"  Total links found: {len(links)}")

        validation = validate_youtube_timestamp_links(links)

        print(f"\n  Link Analysis:")
        print(f"    YouTube links: {validation['youtube_links']}")
        print(f"    Links with timestamps: {validation['links_with_timestamp']}")
        print(f"    Homepage links (BAD): {validation['homepage_links']}")

        all_passed = True

        # Check for issues
        if validation["issues"]:
            print(f"\n  Issues found:")
            for issue in validation["issues"]:
                print(f"    - {issue}")
            all_passed = False

        # Validation criteria
        if validation["homepage_links"] > 0:
            print(f"  FAIL: Found {validation['homepage_links']} homepage links")
            all_passed = False
        else:
            print(f"  PASS: No homepage links found")

        if validation["youtube_links"] > 0 and validation["links_with_timestamp"] == 0:
            print(f"  FAIL: YouTube links found but none have timestamps")
            all_passed = False
        elif validation["youtube_links"] > 0:
            pct = (validation["links_with_timestamp"] / validation["youtube_links"]) * 100
            if pct >= 50:
                print(f"  PASS: {pct:.0f}% of YouTube links have timestamps")
            else:
                print(f"  WARN: Only {pct:.0f}% of YouTube links have timestamps")

        # Show sample links
        print(f"\n  Sample links from response:")
        for text, url in links[:5]:
            truncated_url = url[:80] + "..." if len(url) > 80 else url
            print(f"    [{text}]({truncated_url})")

        return all_passed

    except requests.exceptions.Timeout:
        print(f"  FAIL: Request timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"  FAIL: Error making request: {e}")
        return False


def test_streaming_endpoint():
    """Test the streaming chat endpoint."""
    print("\n" + "="*60)
    print("TEST 4: Streaming Endpoint (End-to-End)")
    print("="*60)

    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"  SKIP: Server not healthy at {API_URL}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  SKIP: Cannot connect to server at {API_URL}")
        return None

    print(f"  Testing streaming with query: '{TEST_QUERY}'")
    print("  This may take a minute...")

    try:
        response = requests.post(
            f"{API_URL}/chat/stream",
            json={
                "query": TEST_QUERY,
                "max_videos": 3
            },
            stream=True,
            timeout=180
        )

        if response.status_code != 200:
            print(f"  FAIL: Streaming endpoint returned {response.status_code}")
            return False

        # Collect the full answer from SSE events
        full_answer = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    try:
                        data = json.loads(line[5:])
                        if 'text' in data:
                            full_answer += data['text']
                    except json.JSONDecodeError:
                        pass

        print(f"\n  Response length: {len(full_answer)} characters")

        # Extract and validate links
        links = extract_links_from_response(full_answer)
        print(f"  Total links found: {len(links)}")

        validation = validate_youtube_timestamp_links(links)

        print(f"\n  Link Analysis:")
        print(f"    YouTube links: {validation['youtube_links']}")
        print(f"    Links with timestamps: {validation['links_with_timestamp']}")
        print(f"    Homepage links (BAD): {validation['homepage_links']}")

        all_passed = True

        if validation["homepage_links"] > 0:
            print(f"  FAIL: Found {validation['homepage_links']} homepage links")
            all_passed = False
        else:
            print(f"  PASS: No homepage links found")

        if validation["youtube_links"] > 0 and validation["links_with_timestamp"] == 0:
            print(f"  FAIL: YouTube links found but none have timestamps")
            all_passed = False
        elif validation["youtube_links"] > 0:
            pct = (validation["links_with_timestamp"] / validation["youtube_links"]) * 100
            print(f"  {'PASS' if pct >= 50 else 'WARN'}: {pct:.0f}% of YouTube links have timestamps")

        # Show sample links
        print(f"\n  Sample links from response:")
        for text, url in links[:5]:
            truncated_url = url[:80] + "..." if len(url) > 80 else url
            print(f"    [{text}]({truncated_url})")

        return all_passed

    except requests.exceptions.Timeout:
        print(f"  FAIL: Request timed out")
        return False
    except Exception as e:
        print(f"  FAIL: Error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TIMESTAMP END-TO-END TEST SUITE")
    print("="*60)

    results = {}

    # Test 1: Utility functions
    results["utility_functions"] = test_utility_functions()

    # Test 2: Transform function
    results["transform_function"] = test_transform_function()

    # Test 3: Chat endpoint (optional - requires server)
    results["chat_endpoint"] = test_chat_endpoint()

    # Test 4: Streaming endpoint (optional - requires server)
    # Skipping streaming test by default as it's similar to chat test
    # results["streaming_endpoint"] = test_streaming_endpoint()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {test_name}: {status}")

    # Exit with appropriate code
    failed = any(r is False for r in results.values())
    if failed:
        print("\nSome tests FAILED!")
        sys.exit(1)
    else:
        print("\nAll tests passed (or skipped)!")
        sys.exit(0)


if __name__ == "__main__":
    main()
