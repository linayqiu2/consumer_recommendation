from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Generator, Callable
import queue
import threading
import requests
import os
import traceback
import json
import time
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timedelta
from googleapiclient.discovery import build

# Import database functions
import database as db
# Import authentication utilities (optional - fails gracefully if deps not installed)
try:
    import auth
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    print("Warning: Auth dependencies not installed. Auth endpoints will be disabled.")

load_dotenv()

print(f"YOUTUBE_API_KEY loaded: {bool(os.getenv('YOUTUBE_API_KEY'))}")
print(f"OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"SUPADATA_API_KEY loaded: {bool(os.getenv('SUPADATA_API_KEY'))}")
print(f"TAVILY_API_KEY loaded: {bool(os.getenv('TAVILY_API_KEY'))}")

app = FastAPI(title="Consumer Recommendation API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# ================
# SYNTHESIS CACHE
# ================
# Cache synthesis results to avoid redundant LLM calls for similar video sets
# Key: hash of sorted product names from video insights
# Value: (synthesis_result, timestamp)
_synthesis_cache: dict[str, tuple[dict, float]] = {}
_synthesis_cache_lock = threading.Lock()
SYNTHESIS_CACHE_TTL_SECONDS = 3600  # 1 hour TTL


def _get_synthesis_cache_key(video_insights: List[dict]) -> str:
    """Generate a cache key from video insights based on product names and video IDs."""
    # Extract unique identifiers: video URLs + product names
    identifiers = []
    for v in video_insights:
        # Add video URL as identifier
        if v.get("video_url"):
            identifiers.append(v["video_url"])
        # Add product names
        for p in v.get("products", []):
            if p.get("name"):
                identifiers.append(p["name"].lower().strip())

    # Sort for consistency and create hash
    identifiers.sort()
    key_string = "|".join(identifiers)
    return hashlib.md5(key_string.encode()).hexdigest()


def _get_cached_synthesis(cache_key: str) -> Optional[dict]:
    """Get cached synthesis result if valid."""
    with _synthesis_cache_lock:
        if cache_key in _synthesis_cache:
            result, timestamp = _synthesis_cache[cache_key]
            if time.time() - timestamp < SYNTHESIS_CACHE_TTL_SECONDS:
                print(f"[Cache HIT] Synthesis cache hit for key {cache_key[:8]}...")
                return result
            else:
                # Expired, remove it
                del _synthesis_cache[cache_key]
                print(f"[Cache EXPIRED] Synthesis cache expired for key {cache_key[:8]}...")
    return None


def _set_cached_synthesis(cache_key: str, result: dict) -> None:
    """Store synthesis result in cache."""
    with _synthesis_cache_lock:
        _synthesis_cache[cache_key] = (result, time.time())
        print(f"[Cache SET] Stored synthesis for key {cache_key[:8]}...")

        # Clean up old entries if cache gets too large (keep max 100 entries)
        if len(_synthesis_cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(_synthesis_cache.keys(),
                                key=lambda k: _synthesis_cache[k][1])
            for old_key in sorted_keys[:20]:  # Remove oldest 20
                del _synthesis_cache[old_key]


# Country to language mapping (ISO 3166-1 alpha-2 to ISO 639-1)
COUNTRY_TO_LANGUAGE = {
    "US": "en",  # United States - English
    "GB": "en",  # United Kingdom - English
    "AU": "en",  # Australia - English
    "CA": "en",  # Canada - English (primarily)
    "IE": "en",  # Ireland - English
    "NZ": "en",  # New Zealand - English
    "JP": "ja",  # Japan - Japanese
    "DE": "de",  # Germany - German
    "AT": "de",  # Austria - German
    "CH": "de",  # Switzerland - German (primarily)
    "FR": "fr",  # France - French
    "BE": "fr",  # Belgium - French (primarily)
    "ES": "es",  # Spain - Spanish
    "MX": "es",  # Mexico - Spanish
    "AR": "es",  # Argentina - Spanish
    "IT": "it",  # Italy - Italian
    "PT": "pt",  # Portugal - Portuguese
    "BR": "pt",  # Brazil - Portuguese
    "KR": "ko",  # South Korea - Korean
    "CN": "zh",  # China - Chinese
    "TW": "zh",  # Taiwan - Chinese
    "HK": "zh",  # Hong Kong - Chinese
    "RU": "ru",  # Russia - Russian
    "NL": "nl",  # Netherlands - Dutch
    "PL": "pl",  # Poland - Polish
    "SE": "sv",  # Sweden - Swedish
    "NO": "no",  # Norway - Norwegian
    "DK": "da",  # Denmark - Danish
    "FI": "fi",  # Finland - Finnish
    "IN": "en",  # India - English (for tech content)
    "SG": "en",  # Singapore - English
    "PH": "en",  # Philippines - English
}


def get_language_for_country(country: str) -> str:
    """Get the language code for a country. Defaults to English."""
    return COUNTRY_TO_LANGUAGE.get(country.upper(), "en")


def parse_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration to seconds. E.g., PT15M30S -> 930, PT1H5M -> 3900"""
    if not duration_str:
        return 0
    import re
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def format_number(n: int) -> str:
    """Format large numbers with K/M suffix. E.g., 2500000 -> '2.5M', 180000 -> '180K'"""
    if n is None or n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ==================== Conversation Context Functions ====================

def summarize_conversation(history: list) -> str:
    """
    Summarize conversation history into key context points.
    Returns a compact summary (~100-200 tokens) instead of full history.
    Uses gpt-4o-mini for efficiency.
    """
    if not history:
        return ""

    # Build raw history for summarization (last 8 messages max)
    def get_msg_parts(msg):
        msg_role = msg.get('role') if isinstance(msg, dict) else msg.role
        msg_content = msg.get('content', '') if isinstance(msg, dict) else msg.content
        return msg_role, msg_content

    raw_history = "\n".join([
        f"{'User' if get_msg_parts(msg)[0] == 'user' else 'Assistant'}: {get_msg_parts(msg)[1][:500]}"
        for msg in history[-8:]
    ])

    system_prompt = """You are a conversation summarizer. Extract the essential context from the conversation that would be needed to understand follow-up questions.

Output a brief summary (2-4 bullet points) covering:
- The main topic/product being discussed
- Key products, models, or items mentioned by name
- Any specific requirements or preferences stated by the user
- The last question/answer pair

Keep it under 150 words. Focus only on facts that would help answer a follow-up question."""

    user_prompt = f"""Summarize this conversation for context:

{raw_history}

Summary:"""

    try:
        # Use fast/cheap model for summarization
        summary = call_gpt5(
            model="gpt-4o-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_type="summarization"
        )
        return summary
    except Exception as e:
        print(f"Error summarizing conversation: {e}")
        # Fall back to truncated raw history
        return format_raw_history(history[-4:])


def format_raw_history(history: list) -> str:
    """Format short history without summarization."""
    if not history:
        return ""
    parts = ["Previous conversation:"]
    for msg in history:
        # Handle both dict and object formats
        msg_role = msg.get('role') if isinstance(msg, dict) else msg.role
        msg_content = msg.get('content', '') if isinstance(msg, dict) else msg.content
        role = "User" if msg_role == "user" else "Assistant"
        content = msg_content[:300] + "..." if len(msg_content) > 300 else msg_content
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def build_conversation_context(history: list) -> str:
    """
    Build context from conversation history.
    - For short conversations (≤4 messages): use raw history
    - For longer conversations: summarize to save tokens
    """
    if not history:
        return ""

    # For short conversations (≤2 exchanges), use raw history
    if len(history) <= 4:
        return format_raw_history(history)

    # For longer conversations, summarize
    return summarize_conversation(history)


def get_last_assistant_message(history: list) -> str:
    """Get the full content of the last assistant message.

    Used when user references previous content (e.g., 'translate the above').
    Returns full content without truncation.
    """
    if not history:
        return ""

    # Find the last assistant message
    for msg in reversed(history):
        msg_role = msg.get('role') if isinstance(msg, dict) else getattr(msg, 'role', None)
        if msg_role == 'assistant':
            msg_content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
            return msg_content

    return ""


def needs_full_content(query: str) -> bool:
    """Check if the query needs full previous content (not truncated).

    Returns True for queries like 'translate the above', 'summarize that', etc.
    """
    query_lower = query.lower()
    reference_patterns = [
        'translate', 'summarize', 'rewrite', 'rephrase',
        'the above', 'that response', 'this response',
        '翻译', '总结', '改写'  # Chinese equivalents
    ]
    return any(pattern in query_lower for pattern in reference_patterns)


# YouTube Topic IDs for precise video discovery
# Reference: https://developers.google.com/youtube/v3/docs/search/list#topicId
# Values can be a single topic ID (str) or a list of topic IDs (list) for broader coverage
# Note: Only curated topic IDs are supported since Freebase deprecation (Feb 2017)
YOUTUBE_TOPIC_MAP = {
    "Electronics": ["/m/07c1v", "/m/019_rr"],  # Technology + Lifestyle (for gadget reviews)
    "EV": ["/m/07yv9", "/m/07c1v", "/m/012f08", "/m/0k4j"],  # Vehicles + Technology + Electric vehicles + Car
    "Beauty": ["/m/019_rr", "/m/0kt51"],       # Lifestyle + Health (for skincare)
    "Cameras": ["/m/07c1v", "/m/03glg"],       # Technology + Hobby (no Photography topic in curated list)
    "Audio": ["/m/04rlf", "/m/07c1v"],         # Music + Technology (for audio equipment)
    "Gaming": "/m/0bzvm2",                     # Gaming (parent)
    "Fashion": "/m/032tl",                     # Fashion
    "Travel": "/m/07bxq",                      # Tourism
    "Fitness": "/m/027x7n",                    # Fitness
    "Automotive": "/m/07yv9",                  # Vehicles
    "Hotels": "/m/07bxq",                      # Tourism (no Hotels topic)
    "Restaurants": "/m/02wbm",                 # Food
}

# Query suffix for review-focused search
YOUTUBE_REVIEW_QUERY_SUFFIX = "review OR unboxing OR hands-on"

def call_gpt5(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    user_id: Optional[int] = None,
    request_type: str = 'chat'
) -> str:
    """
    Helper function to call gpt-5-mini or gpt-5.1 using the new responses API.
    Includes retry logic for transient errors (timeouts, rate limits).
    Automatically logs API usage to the database.

    Args:
        model: "gpt-5-mini" or "gpt-5.1"
        system_prompt: The system instruction
        user_prompt: The user message
        max_retries: Maximum number of retry attempts (default 3)
        user_id: Optional user ID for usage tracking
        request_type: Type of request for tracking ('chat', 'article_generation', etc.)

    Returns:
        The model's text response
    """
    # Combine system and user prompts into a single input
    combined_input = f"{system_prompt}\n\n---\n\nUser: {user_prompt}"

    last_error = None
    for attempt in range(max_retries):
        try:
            response = openai_client.responses.create(
                model=model,
                input=combined_input,
            )

            # Log API usage if we have usage information
            try:
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = getattr(response.usage, 'input_tokens', 0) or 0
                    output_tokens = getattr(response.usage, 'output_tokens', 0) or 0
                    db.log_api_usage(
                        user_id=user_id,
                        request_type=request_type,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
            except Exception as tracking_error:
                # Don't fail the request if tracking fails
                print(f"Warning: Failed to log API usage: {tracking_error}")

            return response.output_text
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            # Check if it's a retryable error (timeout, rate limit, connection error)
            is_retryable = any(keyword in error_type.lower() or keyword in str(e).lower()
                              for keyword in ['timeout', 'rate', 'connection', 'network', 'temporary'])

            if is_retryable and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {error_type} - {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Non-retryable error or final attempt
                raise

    # Should not reach here, but just in case
    raise last_error


def call_gpt5_streaming(
    model: str,
    system_prompt: str,
    user_prompt: str,
    user_id: Optional[int] = None,
    request_type: str = 'chat'
) -> Generator[str, None, None]:
    """
    Streaming version of call_gpt5. Yields text chunks as they arrive.
    Uses OpenAI's responses API with stream=True.
    Logs API usage when the stream completes.

    Args:
        model: The model to use
        system_prompt: System instruction
        user_prompt: User message
        user_id: Optional user ID for usage tracking
        request_type: Type of request for tracking
    """
    combined_input = f"{system_prompt}\n\n---\n\nUser: {user_prompt}"
    usage_info = {'input_tokens': 0, 'output_tokens': 0}

    try:
        # Use streaming mode
        response = openai_client.responses.create(
            model=model,
            input=combined_input,
            stream=True,
        )

        # Iterate over the streaming response
        for event in response:
            # The responses API streams events with different types
            # We're looking for text delta events
            if hasattr(event, 'type'):
                if event.type == 'response.output_text.delta':
                    # This event contains a text chunk
                    if hasattr(event, 'delta'):
                        yield event.delta
                elif event.type == 'response.completed':
                    # Response finished - capture usage if available
                    if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                        usage = event.response.usage
                        usage_info['input_tokens'] = getattr(usage, 'input_tokens', 0) or 0
                        usage_info['output_tokens'] = getattr(usage, 'output_tokens', 0) or 0
                    break
            elif hasattr(event, 'delta'):
                # Fallback: directly yield delta if present
                yield event.delta
            elif hasattr(event, 'text'):
                yield event.text

        # Log usage after streaming completes
        try:
            if usage_info['input_tokens'] > 0 or usage_info['output_tokens'] > 0:
                db.log_api_usage(
                    user_id=user_id,
                    request_type=request_type,
                    model=model,
                    input_tokens=usage_info['input_tokens'],
                    output_tokens=usage_info['output_tokens']
                )
        except Exception as tracking_error:
            print(f"Warning: Failed to log streaming API usage: {tracking_error}")

    except Exception as e:
        print(f"Streaming error: {e}")
        traceback.print_exc()
        yield f"\n\n[Error during streaming: {str(e)}]"


# In-memory cache for video transcripts
transcript_cache: dict[str, Optional[str]] = {}

class ChatMessage(BaseModel):
    """A single message in conversation history."""
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    query: str
    max_videos: int = 12
    disable_search: bool = False
    conversation_history: List[ChatMessage] = []  # Previous messages for context

class VideoInfo(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str
    channel: str
    has_transcript: bool = False
    view_count: Optional[int] = None
    like_count: Optional[int] = None

class QueryResult(BaseModel):
    query: str
    videos: List[VideoInfo]

class WebResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0

class WebQueryResult(BaseModel):
    query: str
    results: List[WebResult]

class AnswerGenerationDebug(BaseModel):
    """Debug info for answer generation flow."""
    method_used: str  # "structured" or "fallback"
    fallback_reason: Optional[str] = None  # Why fallback was used
    video_insights: Optional[List[dict]] = None  # Per-video structured insights
    synthesis: Optional[dict] = None  # Cross-video synthesis

class RankedVideoInfo(BaseModel):
    """Info about a ranked video selected for analysis."""
    video_id: str
    title: str
    url: str
    thumbnail: str
    channel: str
    description: str = ""
    source_query: str = ""
    has_transcript: bool = False

class TimingInfo(BaseModel):
    """Timing information for each step of the pipeline."""
    total_seconds: float = 0.0
    classify_query_seconds: float = 0.0
    generate_queries_seconds: float = 0.0
    youtube_search_seconds: float = 0.0
    video_ranking_seconds: float = 0.0
    web_search_seconds: float = 0.0
    transcript_fetch_seconds: float = 0.0
    insights_extraction_seconds: float = 0.0
    synthesis_seconds: float = 0.0
    answer_generation_seconds: float = 0.0

class DebugInfo(BaseModel):
    generated_queries: List[str]
    query_results: List[QueryResult]
    total_videos_found: int
    videos_with_transcripts: int
    videos_analyzed: int
    # Ranked videos selected for transcript generation
    ranked_videos: List[RankedVideoInfo] = []
    # Web search debug
    web_queries: List[str] = []
    web_query_results: List[WebQueryResult] = []
    total_web_results: int = 0
    # Answer generation debug
    answer_generation: Optional[AnswerGenerationDebug] = None
    # Timing info
    timing: Optional[TimingInfo] = None

class ChatResponse(BaseModel):
    answer: str
    videos: List[VideoInfo]
    sources_summary: str
    debug: Optional[DebugInfo] = None

def search_youtube(
    query: str,
    max_results: int = 5,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    country: Optional[str] = None
) -> List[dict]:
    """Search YouTube for videos related to the query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        published_after: Filter videos published after this date (RFC 3339 format, e.g., '2024-01-01T00:00:00Z')
        published_before: Filter videos published before this date (RFC 3339 format, e.g., '2024-12-31T23:59:59Z')
        country: ISO 3166-1 alpha-2 country code for language filtering (e.g., "US", "JP")

    Returns:
        List of video dictionaries with metadata
    """
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY,
        "videoDuration": "medium",  # Filter for medium-length videos (4-20 min)
    }

    # Add date filters if specified
    if published_after:
        params["publishedAfter"] = published_after
    if published_before:
        params["publishedBefore"] = published_before
    # Add country/language filters if specified
    if country:
        params["regionCode"] = country
        language = get_language_for_country(country)
        params["relevanceLanguage"] = language

    resp = requests.get(search_url, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"YouTube API error: {resp.text}")

    data = resp.json()
    videos = []

    # Limit to max_results (YouTube API sometimes returns more)
    items = data.get("items", [])[:max_results]

    # Collect video IDs to fetch statistics in batch
    video_ids = [item["id"]["videoId"] for item in items]

    # Fetch video statistics (views, likes) and contentDetails (duration) in a single API call
    stats_map = {}
    if video_ids:
        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        stats_params = {
            "part": "statistics,contentDetails",  # Include contentDetails for duration
            "id": ",".join(video_ids),
            "key": YOUTUBE_API_KEY,
        }
        stats_resp = requests.get(videos_url, params=stats_params)
        if stats_resp.status_code == 200:
            stats_data = stats_resp.json()
            for video_item in stats_data.get("items", []):
                vid_id = video_item["id"]
                stats = video_item.get("statistics", {})
                view_count = stats.get("viewCount")
                like_count = stats.get("likeCount")
                # Parse duration from contentDetails
                content_details = video_item.get("contentDetails", {})
                duration_str = content_details.get("duration", "PT0S")
                duration_seconds = parse_duration(duration_str)
                stats_map[vid_id] = {
                    "view_count": int(view_count) if view_count else 0,
                    "like_count": int(like_count) if like_count else None,
                    "duration_seconds": duration_seconds,
                }

    for item in items:
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]
        stats = stats_map.get(video_id, {})
        videos.append({
            "video_id": video_id,
            "title": snippet["title"],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": snippet["thumbnails"]["medium"]["url"],
            "channel": snippet["channelTitle"],
            "description": snippet.get("description", ""),  # Include description for LLM ranking
            "view_count": stats.get("view_count"),
            "like_count": stats.get("like_count"),
            "duration_seconds": stats.get("duration_seconds", 0),
        })

    return videos


def rank_videos_heuristic(videos: List[dict], max_to_select: int = 10) -> List[dict]:
    """
    Heuristic-based video ranking (fallback method).
    Ranks by search position and ensures diversity across queries.
    """
    if len(videos) <= max_to_select:
        return videos

    scored_videos = []
    query_counts = {}

    for v in videos:
        query = v.get("source_query", "")
        position = v.get("query_position", 0)

        # Base score: inverse of position (first result = highest score)
        base_score = 10 - position

        # Diversity penalty: reduce score if we already have videos from this query
        query_counts[query] = query_counts.get(query, 0)
        diversity_penalty = query_counts[query] * 2

        final_score = base_score - diversity_penalty
        scored_videos.append((final_score, v))
        query_counts[query] += 1

    # Sort by score descending
    scored_videos.sort(key=lambda x: x[0], reverse=True)

    # Select top videos
    selected = [v for _, v in scored_videos[:max_to_select]]
    return selected


def rank_videos_with_llm(videos: List[dict], user_query: str, max_to_select: int = 10) -> List[dict]:
    """
    Use LLM to rank videos based on relevance to user query.
    Uses video title and description to determine which videos are most likely
    to contain useful information for the user's question.
    Falls back to heuristic ranking if LLM call fails.
    """
    if len(videos) <= max_to_select:
        return videos

    # Build a list of videos with their index for the LLM to rank
    video_list = []
    for i, v in enumerate(videos):
        video_list.append({
            "index": i,
            "title": v.get("title", ""),
            "channel": v.get("channel", ""),
            "description": v.get("description", "")[:300],  # Truncate long descriptions
        })

    system_prompt = """You are a video relevance ranking assistant. Given a user's query and a list of YouTube videos (with title, channel, and description), rank the videos by how likely they are to contain useful, in-depth information to answer the user's question.

Consider:
1. Direct relevance: Does the video appear to directly address the user's question?
2. Review quality signals: Does it look like an in-depth review vs a short unboxing or reaction?
3. Channel credibility: Is the channel name suggesting expertise (e.g., tech reviewer names, professional channels)?
4. Specificity: Does the video focus on the specific products/topics the user asked about?

Return ONLY a JSON array of video indices (integers) in order from most relevant to least relevant.
Example output: [3, 7, 1, 0, 5, 2, 4, 6]

Do not include any explanation or markdown, just the JSON array."""

    user_prompt = f"""User Query: {user_query}

Videos to rank:
{json.dumps(video_list, ensure_ascii=False, indent=2)}

Return the indices ranked from most relevant to least relevant as a JSON array:"""

    try:
        result = call_gpt5("gpt-5-mini", system_prompt, user_prompt)
        # Parse the JSON array of indices
        ranked_indices = json.loads(result.strip())

        # Validate and select videos
        selected = []
        seen_indices = set()
        for idx in ranked_indices:
            if isinstance(idx, int) and 0 <= idx < len(videos) and idx not in seen_indices:
                selected.append(videos[idx])
                seen_indices.add(idx)
                if len(selected) >= max_to_select:
                    break

        # If LLM didn't return enough valid indices, append remaining videos
        if len(selected) < max_to_select:
            for i, v in enumerate(videos):
                if i not in seen_indices:
                    selected.append(v)
                    if len(selected) >= max_to_select:
                        break

        print(f"LLM ranked videos. Top selected indices: {[videos.index(v) for v in selected[:5]]}...")
        return selected

    except Exception as e:
        print(f"LLM ranking failed: {e}. Falling back to heuristic ranking.")
        return rank_videos_heuristic(videos, max_to_select)


def plan_article_content(
    topic_name: str,
    videos: List[dict],
    web_data: List[dict],
    country: str = "US"
) -> dict:
    """
    Use LLM to determine the article idea/concept and select which videos and web data to use.

    Based on topic_name and the associated video titles, video descriptions, and web data,
    determines a trending and interesting article idea and which content should be used.

    Args:
        topic_name: The topic/category name (e.g., "electronics", "beauty")
        videos: List of video dicts with title, description, channel, etc.
        web_data: List of web source dicts with title, url, content, etc.
        country: ISO 3166-1 alpha-2 country code (e.g., "US", "UK", "IN")

    Returns:
        Dictionary containing:
        - article_idea: The generated article idea/concept
        - selected_video_indices: List of video indices to use
        - selected_web_indices: List of web data indices to use
        - content_strategy: Brief description of how to blend the content
    """
    # Build video summaries for LLM
    video_summaries = []
    for i, v in enumerate(videos):
        video_summaries.append({
            "index": i,
            "title": v.get("title", ""),
            "channel": v.get("channel", ""),
            "description": v.get("description", "")[:300]
        })

    # Build web data summaries for LLM
    web_summaries = []
    for i, w in enumerate(web_data):
        web_summaries.append({
            "index": i,
            "title": w.get("title", ""),
            "url": w.get("url", ""),
            "content_preview": w.get("content", "")[:200]
        })

    # Get country name for better LLM understanding
    country_names = {
        "US": "United States", "UK": "United Kingdom", "GB": "United Kingdom",
        "CA": "Canada", "AU": "Australia", "IN": "India", "DE": "Germany",
        "FR": "France", "JP": "Japan", "BR": "Brazil", "MX": "Mexico",
        "KR": "South Korea", "IT": "Italy", "ES": "Spain", "NL": "Netherlands"
    }
    country_name = country_names.get(country, country)

    system_prompt = f"""You are a content strategist for a consumer recommendation website. Today's date is {datetime.now().strftime('%B %d, %Y')}.

TARGET AUDIENCE: Readers in {country_name} ({country})

Given a topic and available video/web content, your job is to:
1. Create a compelling article idea/concept that would attract readers in {country_name}
2. Select the best videos and web sources to create a comprehensive, interesting article
3. Suggest how to blend the content together

IMPORTANT - Country-Specific Guidelines:
- The article MUST be relevant to consumers in {country_name}
- Focus on products, brands, and trends available or popular in {country_name}
- Do NOT create articles about products or trends specific to other countries (e.g., if targeting US, don't write about "Best EVs in India" or "Top UK-only deals")
- Pricing references should be appropriate for {country_name} market
- Consider local preferences, availability, and market conditions

The article idea should be:
- A clear concept or angle for the article (the actual title will be generated later)
- Specific and actionable (e.g., "Compare top noise-cancelling headphones with expert picks")
- Trending and timely
- Consumer-focused (helping people make purchasing decisions)
- Relevant to {country_name} consumers

When selecting content:
- Choose videos that provide different perspectives or cover different aspects
- Prefer content relevant to {country_name} market when available
- Select web sources that add credibility or additional information
- Aim for a mix that creates a well-rounded article
- Feel free to use multiple videos and web content, blending them together

Return ONLY valid JSON with this structure:
{{
    "article_idea": "Your article concept/idea here",
    "selected_video_indices": [0, 2, 5, ...],
    "selected_web_indices": [0, 1, ...],
    "content_strategy": "Brief description of how to blend the content"
}}"""

    user_prompt = f"""Topic: {topic_name}

Available Videos ({len(video_summaries)} total):
{json.dumps(video_summaries, ensure_ascii=False, indent=2)}

Available Web Sources ({len(web_summaries)} total):
{json.dumps(web_summaries, ensure_ascii=False, indent=2)}

Select videos and web sources to use. Create a trending article idea and content strategy."""

    try:
        result = call_gpt5("gpt-5-mini", system_prompt, user_prompt)

        # Clean up response
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]

        plan = json.loads(result.strip())

        # Validate and extract indices
        video_indices = plan.get("selected_video_indices", [])
        web_indices = plan.get("selected_web_indices", [])

        # Filter valid indices
        valid_video_indices = [i for i in video_indices if isinstance(i, int) and 0 <= i < len(videos)]
        valid_web_indices = [i for i in web_indices if isinstance(i, int) and 0 <= i < len(web_data)]

        return {
            "article_idea": plan.get("article_idea", topic_name),
            "selected_video_indices": valid_video_indices,
            "selected_web_indices": valid_web_indices,
            "content_strategy": plan.get("content_strategy", "")
        }

    except Exception as e:
        print(f"Article planning failed: {e}. Using defaults.")
        # Fallback: use all videos and all web data, with topic as idea
        return {
            "article_idea": topic_name,
            "selected_video_indices": list(range(len(videos))),
            "selected_web_indices": list(range(len(web_data))),
            "content_strategy": "Use all available content"
        }


def get_transcript(video_id: str) -> Optional[str]:
    """Get transcript for a YouTube video using Supadata API with caching.
    Returns raw transcript with timestamps in format: [MM:SS] text

    Uses a two-tier cache:
    1. In-memory cache (fastest, cleared on restart)
    2. Persistent SQLite cache (survives restarts)
    """
    # Check in-memory cache first (fastest)
    if video_id in transcript_cache:
        print(f"Cache hit (memory) for video {video_id}")
        return transcript_cache[video_id]

    # Check persistent DB cache
    cached = db.get_cached_transcript(video_id)
    if cached:
        # Populate in-memory cache for faster subsequent access
        transcript_cache[video_id] = cached
        print(f"Cache hit (DB) for video {video_id}")
        return cached

    print(f"Cache miss for video {video_id}, fetching from Supadata...")
    try:
        # Don't use text=true to get raw segments with timestamps
        response = requests.get(
            "https://api.supadata.ai/v1/youtube/transcript",
            params={"videoId": video_id},
            headers={"x-api-key": SUPADATA_API_KEY},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            transcript = None

            # Supadata returns segments with timestamps when text param is not set
            if isinstance(data, dict) and "content" in data:
                # If it returns segments in content field
                segments = data.get("content", [])
                if isinstance(segments, list):
                    transcript = format_transcript_with_timestamps(segments)
                else:
                    # Plain text returned
                    transcript = segments
            elif isinstance(data, list):
                # Direct list of segments
                transcript = format_transcript_with_timestamps(data)
            elif isinstance(data, dict) and "transcript" in data:
                transcript = data["transcript"]
            else:
                print(f"Unexpected Supadata response format for {video_id}: {data}")
                transcript_cache[video_id] = None
                return None

            if transcript:
                # Save to both caches
                transcript_cache[video_id] = transcript
                db.save_transcript_to_cache(video_id, transcript)
                return transcript
            else:
                transcript_cache[video_id] = None
                return None
        else:
            print(f"Supadata API error for {video_id}: {response.status_code} - {response.text}")
            transcript_cache[video_id] = None
            return None
    except Exception as e:
        print(f"Could not get transcript for {video_id}: {e}")
        transcript_cache[video_id] = None
        return None

def format_transcript_with_timestamps(segments: List[dict]) -> str:
    """Format transcript segments into readable text with timestamps."""
    lines = []
    for segment in segments:
        # Get timestamp in seconds and convert to MM:SS format
        start_time = segment.get("start", segment.get("offset", 0))/1000
        if isinstance(start_time, (int, float)):
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
        else:
            timestamp = "[00:00]"

        text = segment.get("text", "")
        if text:
            lines.append(f"{timestamp} {text}")

    return "\n".join(lines)


def parse_timestamp_to_seconds(timestamp: str) -> int:
    """
    Convert [MM:SS] or MM:SS format to total seconds.

    Args:
        timestamp: String like "[01:23]", "01:23", "[1:23:45]", or "1:23:45"

    Returns:
        Total seconds as integer, or 0 if parsing fails
    """
    if not timestamp:
        return 0
    # Remove brackets if present
    clean = timestamp.strip().strip('[]')

    # Match HH:MM:SS or MM:SS
    match = re.match(r'^(?:(\d+):)?(\d+):(\d+)$', clean)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    return 0


def build_timestamped_youtube_link(url: str, timestamp: str) -> str:
    """
    Build a YouTube URL with timestamp parameter.

    Args:
        url: YouTube video URL
        timestamp: Timestamp in [MM:SS] format

    Returns:
        URL with &t=SECONDS appended, or original URL if no valid timestamp
    """
    if not url:
        return url
    seconds = parse_timestamp_to_seconds(timestamp)
    if seconds > 0:
        # Handle URLs that may already have query params
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}t={seconds}"
    return url


def build_quote_markdown(quote_text: str, channel: str, url: str, timestamp: str) -> str:
    """
    Build a pre-formatted markdown quote with clickable timestamp link.

    Args:
        quote_text: The quote text
        channel: Channel name
        url: Video URL
        timestamp: Timestamp in [MM:SS] format

    Returns:
        Formatted markdown string like: *"quote"* — [Channel at [MM:SS]](url&t=seconds)
    """
    linked_url = build_timestamped_youtube_link(url, timestamp)
    # Escape any quotes in the text
    escaped_text = quote_text.replace('"', '\\"') if quote_text else ""
    # Clean up timestamp format - ensure it's [MM:SS]
    clean_ts = timestamp.strip() if timestamp else ""
    if clean_ts and not clean_ts.startswith('['):
        clean_ts = f'[{clean_ts}]'
    # Include timestamp in visible link text for clickability
    if clean_ts and clean_ts != '[00:00]':
        return f'*"{escaped_text}"* — [{channel} at {clean_ts}]({linked_url})'
    else:
        return f'*"{escaped_text}"* — [{channel}]({linked_url})'


def transform_insights_for_answer_generation(
    video_insights: List[dict],
    synthesis: Optional[dict]
) -> tuple[List[dict], Optional[dict]]:
    """
    Transform video insights and synthesis data to include pre-built markdown links.

    This removes the burden from the LLM of converting timestamps and building links.
    Each quote will have a 'markdown_link' field that can be used directly in the response.

    Args:
        video_insights: List of video insight dicts
        synthesis: Optional synthesis dict

    Returns:
        Tuple of (transformed_video_insights, transformed_synthesis)
    """
    import copy

    def transform_quote(quote: dict, channel: str, url: str) -> dict:
        """Transform a single quote to include pre-built markdown."""
        if not quote:
            return quote
        timestamp = quote.get('timestamp', '[00:00]')
        text = quote.get('text', '')

        return {
            **quote,
            'timestamp_seconds': parse_timestamp_to_seconds(timestamp),
            'linked_url': build_timestamped_youtube_link(url, timestamp),
            'markdown_link': build_quote_markdown(text, channel, url, timestamp)
        }

    # Transform video_insights
    transformed_insights = []
    for insight in video_insights:
        channel = insight.get('channel', '')
        url = insight.get('video_url') or insight.get('url', '')

        transformed_insight = copy.deepcopy(insight)

        # Transform top_quotes at insight level (if present)
        if 'top_quotes' in transformed_insight:
            transformed_insight['top_quotes'] = [
                transform_quote(q, channel, url)
                for q in transformed_insight.get('top_quotes', [])
            ]

        # Transform quotes in each product
        if 'products' in transformed_insight:
            for product in transformed_insight['products']:
                if 'top_quotes' in product:
                    product['top_quotes'] = [
                        transform_quote(q, channel, url)
                        for q in product.get('top_quotes', [])
                    ]

        # Transform emotional_highlights
        if 'emotional_highlights' in transformed_insight:
            transformed_insight['emotional_highlights'] = [
                {
                    **eh,
                    'timestamp_seconds': parse_timestamp_to_seconds(eh.get('timestamp', '[00:00]')),
                    'linked_url': build_timestamped_youtube_link(url, eh.get('timestamp', '[00:00]'))
                }
                for eh in transformed_insight.get('emotional_highlights', [])
            ]

        transformed_insights.append(transformed_insight)

    # Transform synthesis
    transformed_synthesis = None
    if synthesis:
        transformed_synthesis = copy.deepcopy(synthesis)

        # Transform notable_quotes in each product
        if 'products' in transformed_synthesis:
            for product in transformed_synthesis['products']:
                if 'notable_quotes' in product:
                    product['notable_quotes'] = [
                        transform_quote(
                            q,
                            q.get('channel', ''),
                            q.get('url', '')
                        )
                        for q in product.get('notable_quotes', [])
                    ]

    return transformed_insights, transformed_synthesis


def fetch_transcripts_parallel(
    videos: List[dict],
    max_transcripts: int = 10,
    max_workers: int = 5,
    country: str = "US",
    on_transcript_complete: Optional[Callable[[int, int, str, bool], None]] = None
) -> tuple[List[dict], List[dict]]:
    """
    Fetch transcripts for multiple videos in parallel.

    Args:
        videos: List of video dicts (from ranking)
        max_transcripts: Stop after getting this many valid transcripts
        max_workers: Maximum parallel workers
        country: Country code for language validation
        on_transcript_complete: Optional callback(fetched_count, valid_count, video_title, is_valid)
                               called each time a transcript is fetched

    Returns:
        Tuple of (videos_with_transcripts, videos_with_content)
        - videos_with_transcripts: All videos attempted with has_transcript flag
        - videos_with_content: Only videos with valid transcripts
    """
    videos_with_transcripts = []
    videos_with_content = []
    fetched_count = 0

    def fetch_for_video(video: dict) -> dict:
        """Fetch transcript for a single video and validate language."""
        transcript = get_transcript(video["video_id"])
        is_valid = transcript is not None and is_transcript_language_match(transcript, country)
        return {
            **video,
            "transcript": transcript if is_valid else None,
            "has_transcript": is_valid
        }

    # Process videos in batches to allow early termination
    # Fetch in parallel but check results as they complete
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(fetch_for_video, video): video
            for video in videos
        }

        # Process results as they complete
        for future in as_completed(future_to_video):
            original_video = future_to_video[future]
            try:
                result = future.result()
                videos_with_transcripts.append(result)
                fetched_count += 1

                if result["has_transcript"]:
                    videos_with_content.append(result)
                    print(f"  ✓ Got transcript for: {result['title'][:50]}...")

                    # Call progress callback
                    if on_transcript_complete:
                        on_transcript_complete(fetched_count, len(videos_with_content), result['title'][:40], True)

                    # Check if we have enough
                    if len(videos_with_content) >= max_transcripts:
                        print(f"Reached target of {max_transcripts} videos with transcripts")
                        # Cancel remaining futures (best effort)
                        for f in future_to_video:
                            f.cancel()
                        break
                elif result.get("transcript") is not None:
                    print(f"  ✗ Wrong language transcript: {result['title'][:50]}...")
                    if on_transcript_complete:
                        on_transcript_complete(fetched_count, len(videos_with_content), result['title'][:40], False)
                else:
                    print(f"  ✗ No transcript for: {result['title'][:50]}...")
                    if on_transcript_complete:
                        on_transcript_complete(fetched_count, len(videos_with_content), result['title'][:40], False)

            except Exception as e:
                print(f"  ✗ Error fetching transcript for {original_video.get('title', 'unknown')[:50]}: {e}")
                fetched_count += 1
                videos_with_transcripts.append({
                    **original_video,
                    "transcript": None,
                    "has_transcript": False
                })
                if on_transcript_complete:
                    on_transcript_complete(fetched_count, len(videos_with_content), original_video.get('title', 'unknown')[:40], False)

    return videos_with_transcripts, videos_with_content


# Country to expected script type mapping
# "latin" = ASCII-based (English, Spanish, French, German, etc.)
# "cjk" = Chinese, Japanese, Korean
# "cyrillic" = Russian, Ukrainian, etc.
# "arabic" = Arabic script
COUNTRY_SCRIPT_MAP = {
    # Latin script countries
    "US": "latin", "GB": "latin", "CA": "latin", "AU": "latin", "NZ": "latin",
    "IE": "latin", "ZA": "latin", "IN": "latin",  # India uses English widely
    "MX": "latin", "ES": "latin", "AR": "latin", "CO": "latin", "CL": "latin",
    "PE": "latin", "VE": "latin", "EC": "latin",  # Spanish-speaking
    "BR": "latin", "PT": "latin",  # Portuguese
    "FR": "latin", "BE": "latin", "CH": "latin",  # French
    "DE": "latin", "AT": "latin",  # German
    "IT": "latin", "NL": "latin", "SE": "latin", "NO": "latin", "DK": "latin",
    "FI": "latin", "PL": "latin", "CZ": "latin", "HU": "latin", "RO": "latin",
    "TR": "latin", "ID": "latin", "MY": "latin", "PH": "latin", "VN": "latin",
    # CJK countries
    "CN": "cjk", "TW": "cjk", "HK": "cjk",  # Chinese
    "JP": "cjk",  # Japanese
    "KR": "cjk",  # Korean
    # Cyrillic countries
    "RU": "cyrillic", "UA": "cyrillic", "BY": "cyrillic", "KZ": "cyrillic",
    # Arabic script countries
    "SA": "arabic", "AE": "arabic", "EG": "arabic", "MA": "arabic",
    "DZ": "arabic", "IQ": "arabic", "KW": "arabic", "QA": "arabic",
    # Thai
    "TH": "thai",
    # Hebrew
    "IL": "hebrew",
    # Greek
    "GR": "greek",
}


def is_transcript_language_match(transcript: str, country: str = "US", sample_size: int = 500, threshold: float = 0.7) -> bool:
    """
    Check if a transcript's language matches the expected language for a country.

    Uses character script analysis to determine if the transcript is in the expected
    language family for the given country code.

    Args:
        transcript: The transcript text to check
        country: ISO 3166-1 alpha-2 country code (default: "US")
        sample_size: Number of characters to sample from the beginning (default: 500)
        threshold: Minimum ratio of expected script characters (default: 0.7)

    Returns:
        True if the transcript appears to match the expected language, False otherwise
    """
    if not transcript:
        return False

    # Get expected script for country (default to latin if unknown)
    expected_script = COUNTRY_SCRIPT_MAP.get(country.upper() if country else "US", "latin")

    # Take a sample from the beginning of the transcript (skip timestamps)
    clean_text = re.sub(r'\[\d{2}:\d{2}\]', '', transcript)
    sample = clean_text[:sample_size]

    if len(sample) < 50:  # Too short to determine
        return True  # Give benefit of the doubt

    # Count letters by script type
    total_letters = sum(1 for c in sample if c.isalpha())
    if total_letters == 0:
        return True  # No letters to analyze

    if expected_script == "latin":
        # Latin script: ASCII letters (a-z, A-Z) plus extended Latin (accented chars)
        matching = sum(1 for c in sample if c.isalpha() and ord(c) < 384)  # Latin Extended-A ends at 0x17F
    elif expected_script == "cjk":
        # CJK: Chinese, Japanese, Korean characters
        matching = sum(1 for c in sample if c.isalpha() and (
            0x4E00 <= ord(c) <= 0x9FFF or  # CJK Unified Ideographs
            0x3040 <= ord(c) <= 0x309F or  # Hiragana
            0x30A0 <= ord(c) <= 0x30FF or  # Katakana
            0xAC00 <= ord(c) <= 0xD7AF     # Korean Hangul
        ))
    elif expected_script == "cyrillic":
        # Cyrillic script
        matching = sum(1 for c in sample if c.isalpha() and 0x0400 <= ord(c) <= 0x04FF)
    elif expected_script == "arabic":
        # Arabic script
        matching = sum(1 for c in sample if c.isalpha() and 0x0600 <= ord(c) <= 0x06FF)
    elif expected_script == "thai":
        # Thai script
        matching = sum(1 for c in sample if c.isalpha() and 0x0E00 <= ord(c) <= 0x0E7F)
    elif expected_script == "hebrew":
        # Hebrew script
        matching = sum(1 for c in sample if c.isalpha() and 0x0590 <= ord(c) <= 0x05FF)
    elif expected_script == "greek":
        # Greek script
        matching = sum(1 for c in sample if c.isalpha() and 0x0370 <= ord(c) <= 0x03FF)
    else:
        # Default to latin check
        matching = sum(1 for c in sample if c.isalpha() and ord(c) < 128)

    ratio = matching / total_letters
    return ratio >= threshold


# Backward compatibility alias
def is_english_transcript(transcript: str, sample_size: int = 500, threshold: float = 0.7) -> bool:
    """Check if transcript is in English (alias for is_transcript_language_match with US)."""
    return is_transcript_language_match(transcript, "US", sample_size, threshold)

def search_web(query: str, max_results: int = 5, country: Optional[str] = None) -> List[dict]:
    """Search the web using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        country: ISO 3166-1 alpha-2 country code for language filtering (e.g., "US", "JP")
    """
    # Domains to exclude (video platforms we already search separately)
    excluded_domains = ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]

    # Build request payload
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results + 5,  # Request more to account for filtered results
        "include_answer": False,
        "include_raw_content": False,
        "search_depth": "basic",
        "exclude_domains": excluded_domains
    }

    # Add language filter if country is specified
    # Tavily supports include_domains for domain-level filtering
    # For language, we can modify the query or use country-specific TLDs
    if country:
        language = get_language_for_country(country)
        # Tavily doesn't have direct language param, but we can hint in query
        # or filter results. For now, we'll rely on query language matching.
        # Some search APIs support 'search_lang' or similar
        pass

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("results", []):
                url = item.get("url", "")
                # Double-check exclusion (in case API doesn't fully filter)
                if any(domain in url.lower() for domain in excluded_domains):
                    continue
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0)
                })
                if len(results) >= max_results:
                    break
            return results
        else:
            print(f"Tavily API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Web search error: {e}")
        return []

def generate_web_search_queries(user_query: str) -> List[str]:
    """Generate optimized web search queries for text-based content."""
    system_prompt = f"You are a helpful assistant that generates web search queries. Today's date is {datetime.now().strftime('%B %d, %Y')}. Given a user question about product recommendations, generate 2 optimized search queries that would find helpful articles, reviews, and buying guides. Focus on queries that would find written reviews, comparison articles, and expert recommendations (not videos). Return only the queries, one per line, no numbering."

    result = call_gpt5("gpt-5-mini", system_prompt, user_query)
    queries = result.strip().split("\n")
    return [q.strip() for q in queries if q.strip()][:2]


def search_youtube_batch(queries: List[str], max_results_per_query: int = 3) -> tuple[List[dict], set]:
    """
    Search YouTube for multiple queries in parallel.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per query

    Returns:
        Tuple of (all_videos, seen_ids)
        - all_videos: Deduplicated list of video dicts
        - seen_ids: Set of video IDs seen
    """
    all_videos = []
    seen_ids = set()

    def search_single_query(query: str) -> tuple[str, List[dict]]:
        """Search for a single query and return results."""
        results = search_youtube(query, max_results=max_results_per_query)
        return query, results

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(search_single_query, q) for q in queries]
        for future in as_completed(futures):
            try:
                query, results = future.result()
                for v in results:
                    if v["video_id"] not in seen_ids:
                        seen_ids.add(v["video_id"])
                        v["source_query"] = query
                        all_videos.append(v)
            except Exception as e:
                print(f"Error in YouTube search: {e}")

    return all_videos, seen_ids


def search_web_batch(queries: List[str], max_results_per_query: int = 3) -> tuple[List[dict], set]:
    """
    Search web for multiple queries in parallel.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per query

    Returns:
        Tuple of (all_results, seen_urls)
        - all_results: Deduplicated list of result dicts
        - seen_urls: Set of URLs seen
    """
    all_results = []
    seen_urls = set()

    def search_single_query(query: str) -> tuple[str, List[dict]]:
        """Search for a single query and return results."""
        results = search_web(query, max_results=max_results_per_query)
        return query, results

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(search_single_query, q) for q in queries]
        for future in as_completed(futures):
            try:
                query, results = future.result()
                for r in results:
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        r["source_query"] = query
                        all_results.append(r)
            except Exception as e:
                print(f"Error in web search: {e}")

    return all_results, seen_urls


def run_youtube_and_web_search_parallel(
    youtube_queries: List[str],
    web_queries: List[str],
    youtube_max_per_query: int = 3,
    web_max_per_query: int = 3
) -> tuple[List[dict], List[dict]]:
    """
    Run YouTube and Web searches in parallel.

    Args:
        youtube_queries: Queries for YouTube search
        web_queries: Queries for web search
        youtube_max_per_query: Max YouTube results per query
        web_max_per_query: Max web results per query

    Returns:
        Tuple of (youtube_videos, web_results)
    """
    youtube_videos = []
    web_results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        youtube_future = executor.submit(
            search_youtube_batch, youtube_queries, youtube_max_per_query
        )
        web_future = executor.submit(
            search_web_batch, web_queries, web_max_per_query
        )

        # Get results
        try:
            youtube_videos, _ = youtube_future.result()
        except Exception as e:
            print(f"YouTube batch search failed: {e}")

        try:
            web_results, _ = web_future.result()
        except Exception as e:
            print(f"Web batch search failed: {e}")

    return youtube_videos, web_results


def classify_query(user_query: str, conversation_context: str = "") -> str:
    """Classify the query into CHAT, SEARCH, or PRODUCT category.

    Args:
        user_query: The current user query
        conversation_context: Optional context from previous conversation
    """
    # Fast keyword-based override for obvious PRODUCT queries
    # This prevents LLM misclassification of clear recommendation requests
    query_lower = user_query.lower()
    product_keywords = [
        "recommend", "recommendation", "best", "top", "compare", "comparison",
        "should i buy", "which one", "what should i get", "suggest",
        "review", "reviews", "worth it", "good for", "better than"
    ]
    if any(keyword in query_lower for keyword in product_keywords):
        print(f"Query: '{user_query[:60]}...' -> Classified as: PRODUCT (keyword match)")
        return "PRODUCT"

    system_prompt = f"""You are a query classifier. Today's date is {datetime.now().strftime('%B %d, %Y')}. Classify user messages into exactly one of these categories:

CHAT - Simple conversation that can be answered directly without external info (greetings, basic questions, opinions, things you already know)

SEARCH - Questions needing current/factual information from the web, but NOT recommendations (weather, news, events, general facts, "what is X", current info lookups, definitions)

PRODUCT - ANY recommendation request including:
- Product recommendations, reviews, buying advice, comparisons
- "best X", "which X should I buy", "recommend X"
- Restaurant recommendations, hotel recommendations, travel recommendations
- Service recommendations (gym, spa, etc.)
- ANY query asking for opinions, reviews, or recommendations about products, places, or services
- Real-life testing, user experiences, "what people say" about something
- Follow-up questions about products/services previously discussed

IMPORTANT:
- If the user says "recommend" or asks for "best" anything, classify as PRODUCT.
- If conversation context shows the user was discussing products/services and asks a follow-up question (even short ones like "what about battery life?"), classify as PRODUCT.

Reply with ONLY one word: CHAT, SEARCH, or PRODUCT"""

    # Build user prompt with context if available
    if conversation_context:
        user_prompt = f"""{conversation_context}

Current query: {user_query}"""
    else:
        user_prompt = user_query

    result = call_gpt5("gpt-5-mini", system_prompt, user_prompt).strip().upper()
    if "PRODUCT" in result:
        classification = "PRODUCT"
    elif "SEARCH" in result:
        classification = "SEARCH"
    else:
        classification = "CHAT"
    print(f"Query: '{user_query[:60]}...' -> Classified as: {classification}")
    return classification

def generate_chat_response(user_query: str, conversation_context: str = "", full_last_message: str = "") -> str:
    """Generate a regular conversational response.

    Args:
        user_query: The current user query
        conversation_context: Summarized/truncated context for understanding
        full_last_message: Full content of last assistant message (for translate/summarize)
    """
    system_prompt = "You are a friendly product advisor assistant. If the user wants to translate, summarize, or otherwise operate on previous content, use the 'Previous content to operate on' provided. Otherwise, respond helpfully and let them know you can help with product recommendations, reviews, and buying advice. Keep responses brief and friendly."

    if full_last_message:
        # User wants to operate on previous content
        user_prompt = f"""Previous content to operate on:
{full_last_message}

User request: {user_query}"""
    elif conversation_context:
        user_prompt = f"""Context: {conversation_context}

User request: {user_query}"""
    else:
        user_prompt = user_query

    return call_gpt5("gpt-5-mini", system_prompt, user_prompt)

def generate_search_response(user_query: str, conversation_context: str = "") -> tuple[str, List[dict]]:
    """Handle SEARCH queries - quick web lookup for factual/current info.

    Args:
        user_query: The current user query
        conversation_context: Optional context from previous conversation for follow-up questions
    """
    # If there's conversation context, incorporate it into the search query
    search_query = user_query
    if conversation_context:
        # Extract key topics from context to improve search
        search_query = f"{user_query} (context: {conversation_context[:200]})"

    # Do a quick web search
    web_results = search_web(search_query, max_results=5)

    context_section = ""
    if conversation_context:
        context_section = f"\n\nPrevious conversation context:\n{conversation_context}\n\nThe user is asking a follow-up question. Use the context to understand what they're referring to."

    if not web_results:
        # Fallback to just answering without search results
        system_prompt = f"You are a helpful assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}. Answer the user's question. If you don't have current information, acknowledge that.{context_section}"
        result = call_gpt5("gpt-5-mini", system_prompt, user_query)
        return result, []

    # Build context from web results
    web_context = "\n\n".join([
        f"Source: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in web_results
    ])

    system_prompt = f"""You are a helpful assistant. Today's date is {datetime.now().strftime('%B %d, %Y')}.
Answer the user's question using the provided web search results.
Be concise and informative. Cite sources when relevant.
If the search results don't fully answer the question, acknowledge that.{context_section}"""

    user_prompt = f"Question: {user_query}\n\nWeb Search Results:\n{web_context}"
    result = call_gpt5("gpt-5-mini", system_prompt, user_prompt)
    return result, web_results

def generate_search_queries(user_query: str, conversation_context: str = "") -> List[str]:
    """Generate optimized search queries from user input.

    Args:
        user_query: The current user query
        conversation_context: Optional context from previous conversation for follow-up questions
    """

    system_prompt = f""""You are an expert YouTube search-query generator for consumer research.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

Given a user query, your task is to:
1) Identify the query type:
   - CATEGORY (general inquiry about a class of items)
   - SINGLE_ENTITY (asking about one specific item)
   - USER_LIST_COMPARISON (user provides a list of multiple specific items to compare)
   - CONSTRAINT_QUERY (preferences or conditions, e.g., "must be lightweight", "needs long battery life", "must have private onsen")
   - FOLLOW_UP (a follow-up question about previously discussed items - use conversation context to expand)

2) Generate **2–6 HIGH-QUALITY YouTube search queries** that are likely to return useful review or comparison videos.
Your goal is to surface videos with deep insights, strong opinions, and meaningful evaluations.

====================
 RULES BY QUERY TYPE
====================

=== FOLLOW_UP (with conversation context) ===
- If conversation context is provided, use it to understand what products/items were previously discussed.
- Expand ambiguous references (e.g., "it", "the first one", "which is better") into specific product names.
- Generate queries that combine the previous topic with the new question.
- Example: If context mentions "Sony A7IV camera" and query is "what about low-light?", generate:
  "Sony A7IV low light performance review"

=== CATEGORY ===
- Produce broad category-level queries.
- You may include:
  - 1 direct query,
  - 1 comparison-style query,
  - 1 niche or problem-based query.
- Keep each query focused on one clear idea.

=== SINGLE_ENTITY ===
- Generate 2–3 queries focusing specifically on that item.
- Keep queries clean, concise, and not overloaded with extra conditions.

=== USER_LIST_COMPARISON ===
(The user provides 3+ named items; a single video rarely covers all of them.)

Follow this strategy:

A) Generate **one query per item**, each optimized for discovering a good review or real-world test video.

B) Optionally generate **1–2 cluster-level queries** IF helpful AND within the total 2–6 query limit
   (e.g., a broad category + "review", "comparison").

C) Optionally generate **1–2 theme-based queries**, derived from the user's evaluation criteria
   (e.g., "durability comparison", "comfort real-world experience").
   Include these only if they fit within the total limit and are likely to retrieve results.

D) Optionally generate **1–2 pairwise comparison queries**, but ONLY if:
   - those two items are commonly compared, or
   - the user's list contains items that naturally pair.
   Include these only if they fit within the 2–6 total query limit.

=== CONSTRAINT_QUERY ===
- Generate queries centered on the constraints or factors the user cares about.
- Combine the relevant category/domain and the constraint in a short, natural way

=====================
 GENERAL INSTRUCTIONS
=====================

- Keep queries concise (around 3–8 words/tokens).
- Do NOT include commentary or numbering.
- Use either English or the user's language, but avoid mixing both heavily in the same query.
- Avoid over-specificity:
  - Do NOT list more than two item names in any query.
  - Do NOT encode all constraints into one query.
- If item names are long, abbreviate intelligently (brand + key identifier only).
- Each query should represent a single, clear search intent.
- FINAL OUTPUT LIMITS:
  - If user language is English → return **2–6** queries.
  - If user language is NOT English → return **Pure English + original language versions**, **no more than 12 total**.
- Output ONLY the queries, one per line, with no extra text."""

    # Build user prompt with context if available
    if conversation_context:
        user_prompt = f"""{conversation_context}

Current query: {user_query}

Generate search queries that account for the conversation context above. Expand any ambiguous references to specific products/items mentioned in the context."""
    else:
        user_prompt = user_query

    result = call_gpt5("gpt-5-mini", system_prompt, user_prompt)
    queries = result.strip().split("\n")
    return [q.strip() for q in queries if q.strip()][:6]

# =========================
# NEW: STRUCTURED EXTRACTION
# =========================

def extract_video_insights(video: dict) -> Optional[dict]:
    """
    Extract structured insights from a single video's transcript.

    Supports both:
    - Single-product review videos
    - Multi-product / comparison videos

    Output includes a 'products' list so each product has its own pros/cons/quotes.
    """
    transcript = video.get("transcript") or ""
    if not transcript:
        return None

    transcript_excerpt = transcript[:8000]

    system_prompt = (
        "You are a YouTube product review analyst. "
        "You receive a product review or comparison video transcript that already includes [MM:SS] timestamps. "
        "Your job is to extract rich, structured insights that capture what the reviewer really thinks "
        "about EACH product discussed, including memorable quotes and key emotional moments."
    )

    # Define JSON template separately (not in f-string) to avoid escaping issues
    json_template = """{
  "channel": "CHANNEL_NAME",
  "url": "VIDEO_URL",
  "video_title": "VIDEO_TITLE",
  "video_type": "single_product | multi_product | unclear",
  "products": [
    {
      "name": "exact product name as used in the video",
      "role": "main | alternative | mentioned",
      "overall_sentiment": "positive | negative | mixed | neutral",
      "top_quotes": [
        {"text": "A memorable, opinion-rich quote that expresses a CLEAR sentiment about a specific product or feature. It should be long enough to convey meaning (8–25 words), not a generic remark.", "timestamp": "[MM:SS]"},
        {"text": "Another meaningful quote, ideally expressing a different type of opinion (praise, criticism, or a key insight). Avoid generic filler phrases.", "timestamp": "[MM:SS]"},
        {"text": "If available, include a third quote with specificity (e.g., mentions of specific features, performance, value, or build quality).", "timestamp": "[MM:SS]"}
      ],
      "pros": [
        "short phrase describing a concrete advantage of THIS product"
      ],
      "cons": [
        "short phrase describing a concrete drawback of THIS product"
      ],
      "use_case_recommendations": [
        "who or what use case this specific product is good for"
      ],
      "price_mentioned": "price or price range for this product if mentioned, else empty string",
      "summary": "2–3 sentence summary of what the reviewer thinks about this specific product"
    }
  ],
  "general_advice": [
    "category-level buying tips not tied to a single product (e.g., 'spend more on lenses than bodies')"
  ],
  "tone": "overall tone of the video, e.g. enthusiastic, balanced, disappointed",
  "emotional_highlights": [
    {"moment": "what happens or is said at a strong emotional point", "timestamp": "[MM:SS]"}
  ]
}"""

    user_prompt = f"""
Video title: {video.get('title')}
Channel: {video.get('channel')}
URL: {video.get('url')}

Transcript with timestamps:
{transcript_excerpt}

This video might be:
- A single-product review (one main product)
- A comparison / multi-product video (two or more products meaningfully compared)
- Or unclear

Identify each distinct product that is actually evaluated (not just name-dropped).

Return ONLY valid JSON (no markdown, no commentary) with EXACTLY this structure:

{json_template}

Rules:
- If it's a single-product review, there will typically be ONE product with role 'main'.
- If it's a comparison, include ALL products that are meaningfully evaluated, with roles like 'main' or 'alternative'.
- Include at least 2 quotes total across products if possible (up to ~6), spread across the most important products.
- Prefer short, punchy quotes that clearly express strong opinions (love/hate/surprise).
- Use timestamps from the transcript when possible; if unclear, approximate with the closest [MM:SS] you see.
"""

    raw = call_gpt5("gpt-5-mini", system_prompt, user_prompt)
    try:
        return json.loads(raw.strip())
    except Exception as e:
        print(f"Failed to parse video insights JSON for {video.get('url')}: {e}\nRaw: {raw}")
        return None


def extract_insights_for_videos(
    videos_with_content: List[dict],
    max_workers: int = 8,
    on_video_complete: Optional[Callable[[int, int, str], None]] = None,
    on_insight_ready: Optional[Callable[[dict], None]] = None
) -> List[dict]:
    """
    Run structured extraction for all videos with transcripts IN PARALLEL.
    This significantly speeds up the process since each extraction is an independent LLM call.

    Args:
        videos_with_content: List of video dicts with transcripts
        max_workers: Max parallel threads
        on_video_complete: Optional callback(completed_count, total_count, video_title)
                          called each time a video finishes processing
        on_insight_ready: Optional callback(insight_dict) called immediately when
                         each insight is extracted (for early synthesis)
    """
    if not videos_with_content:
        return []

    insights = []
    total_videos = len(videos_with_content)
    completed_count = 0
    print(f"Extracting insights from {total_videos} videos in parallel (max {max_workers} workers)...")

    def extract_single(video):
        """Helper to extract insights from a single video."""
        try:
            vi = extract_video_insights(video)
            return vi
        except Exception as e:
            print(f"Error extracting insights for video {video.get('url')}: {e}")
            return None

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        future_to_video = {executor.submit(extract_single, v): v for v in videos_with_content}

        # Collect results as they complete
        for future in as_completed(future_to_video):
            video = future_to_video[future]
            try:
                result = future.result()
                completed_count += 1
                video_title = video.get('title', 'Unknown')[:40]

                if result:
                    insights.append(result)
                    print(f"  ✓ Extracted insights for: {video_title}...")
                    # Notify early synthesis callback if provided
                    if on_insight_ready:
                        on_insight_ready(result)
                else:
                    print(f"  ○ No insights from: {video_title}...")

                # Call the progress callback if provided
                if on_video_complete:
                    on_video_complete(completed_count, total_videos, video_title)

            except Exception as e:
                completed_count += 1
                print(f"  ✗ Failed for {video.get('title', 'Unknown')[:40]}: {e}")
                if on_video_complete:
                    on_video_complete(completed_count, total_videos, video.get('title', 'Unknown')[:40])

    print(f"Successfully extracted insights from {len(insights)}/{total_videos} videos")
    return insights

def synthesize_video_insights(video_insights: List[dict]) -> Optional[dict]:
    """
    Synthesize multiple video insights into aggregated JSON.

    Handles both:
    - Multiple videos about the same single product
    - Mixed set of videos, some single-product, some multi-product / comparison

    The synthesis is per-product, plus some comparison-level insights.

    Uses caching to avoid redundant LLM calls for the same video sets.
    """
    if not video_insights:
        return None

    # Check cache first
    cache_key = _get_synthesis_cache_key(video_insights)
    cached_result = _get_cached_synthesis(cache_key)
    if cached_result is not None:
        return cached_result

    system_prompt = (
        "You are a multi-video synthesis engine for product reviews. "
        "You receive structured JSON insights from several YouTube videos about a product category. "
        "Each video may cover one or multiple products. "
        "Your job is to aggregate them into a structured summary organized PER PRODUCT, plus key comparisons."
    )

    # Define JSON template separately (not in f-string) to avoid escaping issues
    synthesis_json_template = """{
  "products": [
    {
      "name": "normalized product name",
      "also_called": ["alternative spellings or abbreviations if any"],
      "overall_sentiment": "positive | negative | mixed | neutral",
      "consensus_pros": [
        "pro that multiple reviewers mention for this product"
      ],
      "consensus_cons": [
        "con that multiple reviewers mention for this product"
      ],
      "notable_quotes": [
        {
          "text": "direct quote about this product",
          "timestamp": "[MM:SS]",
          "channel": "Channel Name",
          "url": "video URL"
        }
      ],
      "best_for": [
        "who or what use case this specific product is best for"
      ],
      "price_range_rough": "rough price or price range based on all videos, if mentioned",
      "summary": "2–4 sentence summary of how this product is perceived overall across videos"
    }
  ],
  "comparisons": [
    {
      "between": ["Product Name A", "Product Name B"],
      "summary": "short summary of how reviewers compare A vs B (who should pick which)",
      "who_should_choose_which": "concrete guidance like: 'pick A if you care about X, B if you care about Y'"
    }
  ],
  "cross_video_general_advice": [
    "buying tips that apply across products, based on 'general_advice' and patterns across videos"
  ],
  "tone_overall": "overall tone of reviewers across the entire set (e.g., 'enthusiastic but cautious about budget trade-offs')",
  "unique_video_only_insights": [
    "insights that are more likely to come from video than from spec sheets or text (e.g., 'feels cheaper in hand', 'noticeable noise during operation')"
  ]
}"""

    user_prompt = f"""
Here are structured insights from multiple videos (valid JSON objects):

{json.dumps(video_insights, ensure_ascii=False, indent=2)}

Interpretation notes:
- Each video has a 'products' array.
- Each product has: name, role, overall_sentiment, pros, cons, use cases, price_mentioned, summary, top_quotes.
- Some videos are single-product (one main product).
- Some are multi-product comparisons (multiple products with roles 'main' or 'alternative').

Your job:
- Group together products that clearly refer to the same thing (e.g., same product with different capitalizations or abbreviations).
- Summarize consensus pros/cons per product across videos.
- Capture how reviewers compare products (A vs B, B vs C, etc.).
- Capture category-level advice that applies across products.

Return ONLY valid JSON (no markdown) with this exact structure:

{synthesis_json_template}
"""

    raw = call_gpt5("gpt-5-mini", system_prompt, user_prompt)
    try:
        result = json.loads(raw.strip())
        # Cache the successful result
        _set_cached_synthesis(cache_key, result)
        return result
    except Exception as e:
        print(f"Failed to parse synthesis JSON: {e}\nRaw: {raw[:500]}...")
        return None


# ================
# ANSWER GENERATION
# ================

def generate_answer_fallback(user_query: str, videos_with_transcripts: List[dict], web_results: List[dict] = None) -> str:
    """
    Old behavior: directly stuff transcripts and web content into a single prompt.
    Used as a fallback if structured extraction/synthesis fails.
    """
    video_context_parts = []
    for i, v in enumerate(videos_with_transcripts):
        if v.get("transcript"):
            transcript = v["transcript"][:3000]
            video_context_parts.append(f"""
Video {i+1}: "{v['title']}" by {v['channel']}
URL: {v['url']}
Transcript excerpt:
{transcript}
---""")

    video_context = "\n".join(video_context_parts)

    web_context_parts = []
    if web_results:
        for i, w in enumerate(web_results):
            content = w.get("content", "")[:2000]
            web_context_parts.append(f"""
Article {i+1}: "{w['title']}"
URL: {w['url']}
Content:
{content}
---""")

    web_context = "\n".join(web_context_parts)

    system_prompt = """You are an expert consumer product advisor. Your job is to provide comprehensive, helpful answers about product recommendations by synthesizing insights from both YouTube review videos and web articles.

Guidelines:
1. Synthesize information from both video AND web sources to give a balanced view
2. Highlight key pros and cons mentioned by reviewers and articles
3. Include specific product recommendations when relevant
4. Mention price ranges if discussed
5. Note any consensus or disagreements among different sources
6. Be concise but thorough
7. Format your response with clear sections using markdown

IMPORTANT - Making responses authentic and trustworthy:
8. Include 2-3 compelling DIRECT QUOTES from video transcripts to add personality and credibility. Format them as: *"quote here"* — Channel Name
9. Add INLINE CITATIONS using markdown links with friendly source names:
   - For videos: [Channel Name](video_url) - e.g., "According to [Channel Name](https://youtube.com/watch?v=xxx)..."
   - For articles: [Publication Name](article_url) - e.g., "[Wirecutter](https://nytimes.com/wirecutter/...) recommends..."
   - Extract the publication name from the URL domain or article title (e.g., rtings.com → RTINGS, tomsguide.com → Tom's Guide)
10. Make it feel like "I watched these reviews for you" rather than "I processed this data"
11. Use the actual URLs provided in the sources - do NOT make up or modify URLs

Always be honest if the sources don't fully answer the question."""

    user_message = f"""User Question: {user_query}

=== YOUTUBE VIDEO TRANSCRIPTS ===
{video_context if video_context else "No video transcripts available."}

=== WEB ARTICLES ===
{web_context if web_context else "No web articles available."}

Please provide a comprehensive answer to the user's question based on these sources."""

    return call_gpt5("gpt-5.1", system_prompt, user_message)

def generate_answer(user_query: str, videos_with_transcripts: List[dict], web_results: List[dict] = None, conversation_context: str = "") -> tuple[str, AnswerGenerationDebug]:
    """
    NEW behavior:
    1. Extract structured insights from each video's transcript (with timestamps).
    2. Synthesize cross-video insights.
    3. Use structured video insights + synthesis + web results to generate a differentiated answer.
    If anything fails badly, fall back to generate_answer_fallback.

    Args:
        user_query: The current user query
        videos_with_transcripts: List of video dicts with transcripts
        web_results: Optional list of web search results
        conversation_context: Optional context from previous conversation for follow-up questions

    Returns: (answer_text, AnswerGenerationDebug)
    """
    try:
        # Filter to videos that actually have transcripts
        videos_with_content = [v for v in videos_with_transcripts if v.get("transcript")]

        # 1) Structured extraction per video
        video_insights = extract_insights_for_videos(videos_with_content)
        if not video_insights:
            print("No structured video insights extracted; returning error (no fallback).")
            debug = AnswerGenerationDebug(
                method_used="error",
                fallback_reason="no_video_insights",
                video_insights=None,
                synthesis=None
            )
            return "Could not extract insights from video transcripts. Please try again.", debug

        # 2) Synthesis across videos (optional - continue without it if it fails)
        synthesis = synthesize_video_insights(video_insights)
        if synthesis is None:
            print("Synthesis failed; continuing with video insights only (no fallback).")

        # 2.5) Transform data to include pre-built markdown links for quotes
        # This ensures timestamps are correctly converted to YouTube URL parameters
        transformed_insights, transformed_synthesis = transform_insights_for_answer_generation(
            video_insights, synthesis
        )

        # 3) Build compact web context (titles + brief snippets)
        web_summaries = []
        if web_results:
            for w in web_results[:5]:
                content = (w.get("content") or "")[:400]
                web_summaries.append({
                    "title": w.get("title"),
                    "url": w.get("url"),
                    "snippet": content
                })

        # 4) Build conversation context section if available
        conversation_section = ""
        if conversation_context:
            conversation_section = f"""
## Previous Conversation Context
{conversation_context}

The user is asking a follow-up question. When responding:
- Reference previous discussion naturally if relevant
- Don't repeat information already given in the prior conversation
- Build on prior recommendations if applicable
- Acknowledge when answering follow-up questions (e.g., "Regarding your question about X...")
"""

        system_prompt = f"""You are an expert consumer product advisor who has already watched several YouTube review videos and read some web articles for the user.
{conversation_section}

You have THREE key inputs:
1) Structured insights from individual YouTube videos (including per-product details and quotes with timestamps).
2) An aggregated synthesis across all videos, organized PER PRODUCT and including COMPARISONS (which product is better for whom).
3) Brief summaries of web articles.

Some videos are single-product reviews; others are multi-product comparisons. The synthesis JSON already merges products across videos and includes a 'products' list and a 'comparisons' list.

Your job is to produce a uniquely rich answer that feels like:
"I watched these videos and read these articles for you."

CRITICAL: The user is asking a concrete question about product recommendations.
You MUST take a stance and give direct, actionable recommendations, not just summarize.

STRICT GUIDELINES:

1. Start with a decision-first section:
   "### My top picks (short version)"
   - Use the synthesized 'products' list to pick 2–4 specific products.
   - For each, add 1–2 bullets:
     - Who they are best for
     - The strongest reason based on the videos.
   - If budget is implied, mention approximate price tiers (e.g., under $500, around $800). Be approximate, not fake-precise.

2. If the synthesis includes 'comparisons':
   - Weave that into your explanation, especially if it's clearly A vs B.
   - Use patterns like: "If you care more about X, pick A; if you care more about Y, pick B."

3. Then add:
   "### What reviewers really think"
   - Use consensus_pros/cons per product from the synthesis.
   - Use controversial_points if present, or infer disagreements from mixed sentiments.
   - Use the phrasing "Across multiple reviewers..." and "Only one or two reviewers said..." where relevant.

4. Quotes:
   - Use 4–6 memorable DIRECT QUOTES from video_insights if available.
   - Spread them across multiple channels (do NOT only quote one channel).
   - When quoting about a specific product, make that product explicitly clear.
   - CRITICAL: Look for the 'markdown_link' field in each quote object within video_insights.products[].top_quotes[]
   - The markdown_link field contains a READY-TO-USE formatted string like: *"quote text"* — [Channel](url&t=seconds)
   - COPY THE markdown_link VALUE EXACTLY as-is into your response
   - DO NOT construct your own links - use the pre-built markdown_link field

5. Sources:
   - Make clear when insights come from video vs from web:
     - "According to [Channel Name](video_url)..."
     - "Based on [Publication Name](article_url)..."
   - Derive Publication Name from the domain or title.
   - Do NOT invent URLs. Only use those in the JSON payload.

6. Perspective and structure:
   Add small subsections (if relevant):
   - "### Reviewer personas" (who is a pixel-peeper, who is budget-focused, who is a vlogger, etc.)
   - "### When each option is a great choice" (map different user types to different products using 'best_for' and 'use_case_recommendations')
   - "### What to watch out for" (use consensus_cons + per-product cons)
   - "### Video-only insights you won’t see in spec sheets" (use emotional_highlights + unique_video_only_insights)

7. End with:
   "### What I’d pick if I were you"
   - Give 1–2 concrete recommendations based on the user’s question.
   - Map different user priorities to different picks (e.g., “If you care most about portability, pick X; if you care about better low-light, pick Y.”)
   - Be decisive but honest about trade-offs.

8. General rules:
   - Prefer products that appear most often and most positively across the video_insights.
   - Avoid hallucinating products that never appear.
   - Be concise but specific; avoid generic phrasing like "overall it's a good product" without details.
   - If the sources don’t fully answer something the user asked, say so explicitly and suggest what else they should look at.
"""

        user_payload = {
            "user_question": user_query,
            "video_insights": transformed_insights,
            "multi_video_synthesis": transformed_synthesis,
            "web_articles": web_summaries,
        }

        user_message = (
            "Here is the structured data you have access to. "
            "Use it to answer the user's question.\n\n"
            f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
            "REMINDER: When including quotes, use the 'markdown_link' field from each quote object. "
            "It contains a ready-to-use clickable link format."
        )

        answer = call_gpt5("gpt-5.1", system_prompt, user_message)
        debug = AnswerGenerationDebug(
            method_used="structured" if synthesis else "structured_no_synthesis",
            fallback_reason="synthesis_failed" if not synthesis else None,
            video_insights=video_insights,
            synthesis=synthesis
        )
        return answer, debug

    except Exception as e:
        print(f"Error in generate_answer structured flow: {e}")
        traceback.print_exc()
        # Don't fall back - return error info so we can debug
        debug = AnswerGenerationDebug(
            method_used="error",
            fallback_reason=f"exception: {str(e)}",
            video_insights=None,
            synthesis=None
        )
        return f"Error generating answer: {str(e)}", debug

# ==================== Authentication Endpoints ====================

if AUTH_AVAILABLE:
    class RegisterRequest(BaseModel):
        email: EmailStr
        password: str

    class LoginRequest(BaseModel):
        email: EmailStr
        password: str

    class AuthResponse(BaseModel):
        token: str
        user: dict

    class OAuthRequest(BaseModel):
        provider: str  # 'google' or 'github'
        access_token: str  # OAuth access token from provider

    @app.post("/auth/register", response_model=AuthResponse)
    async def register(request: RegisterRequest):
        """Register a new user with email and password."""
        # Check if user already exists
        existing_user = db.get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Validate password
        if len(request.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

        # Hash password and create user
        password_hash = auth.hash_password(request.password)
        user_id = db.create_user(email=request.email, password_hash=password_hash)

        # Create JWT token
        token = auth.create_access_token(data={"sub": str(user_id)})

        return AuthResponse(
            token=token,
            user={"id": user_id, "email": request.email}
        )

    @app.post("/auth/login", response_model=AuthResponse)
    async def login(request: LoginRequest):
        """Login with email and password."""
        # Find user
        user = db.get_user_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Check if user has a password (might be OAuth-only)
        if not user.get("password_hash"):
            raise HTTPException(status_code=401, detail="Please use OAuth to sign in")

        # Verify password
        if not auth.verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create JWT token
        token = auth.create_access_token(data={"sub": str(user["id"])})

        return AuthResponse(
            token=token,
            user={"id": user["id"], "email": user["email"]}
        )

    @app.post("/auth/oauth", response_model=AuthResponse)
    async def oauth_login(request: OAuthRequest):
        """Login or register via OAuth provider."""
        # Validate provider
        if request.provider not in ["google", "github"]:
            raise HTTPException(status_code=400, detail="Unsupported OAuth provider")

        # Get user info from OAuth provider
        try:
            if request.provider == "google":
                # Verify Google token and get user info
                response = requests.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {request.access_token}"}
                )
                if response.status_code != 200:
                    raise HTTPException(status_code=401, detail="Invalid Google token")
                user_info = response.json()
                oauth_id = user_info.get("id")
                email = user_info.get("email")
            elif request.provider == "github":
                # Verify GitHub token and get user info
                response = requests.get(
                    "https://api.github.com/user",
                    headers={"Authorization": f"Bearer {request.access_token}"}
                )
                if response.status_code != 200:
                    raise HTTPException(status_code=401, detail="Invalid GitHub token")
                user_info = response.json()
                oauth_id = str(user_info.get("id"))

                # GitHub might not return email in user endpoint, fetch separately
                email = user_info.get("email")
                if not email:
                    email_response = requests.get(
                        "https://api.github.com/user/emails",
                        headers={"Authorization": f"Bearer {request.access_token}"}
                    )
                    if email_response.status_code == 200:
                        emails = email_response.json()
                        primary_email = next((e for e in emails if e.get("primary")), None)
                        email = primary_email.get("email") if primary_email else None

                if not email:
                    raise HTTPException(status_code=400, detail="Could not get email from GitHub")

            if not email or not oauth_id:
                raise HTTPException(status_code=400, detail="Could not get user info from provider")

        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"OAuth provider error: {str(e)}")

        # Check if user exists by OAuth
        user = db.get_user_by_oauth(request.provider, oauth_id)

        if not user:
            # Check if user exists by email
            user = db.get_user_by_email(email)
            if user:
                # Link OAuth to existing account
                db.update_user_oauth(user["id"], request.provider, oauth_id)
            else:
                # Create new user
                user_id = db.create_user(
                    email=email,
                    oauth_provider=request.provider,
                    oauth_id=oauth_id
                )
                user = {"id": user_id, "email": email}

        # Create JWT token
        token = auth.create_access_token(data={"sub": str(user["id"])})

        return AuthResponse(
            token=token,
            user={"id": user["id"], "email": user["email"]}
        )

    @app.get("/auth/me")
    async def get_current_user(authorization: str = Header(None)):
        """Get current user info from JWT token."""
        if not authorization:
            raise HTTPException(status_code=401, detail="No authorization header")

        # Extract token from "Bearer <token>"
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header format")

        token = parts[1]
        user_id = auth.get_user_id_from_token(token)

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return {
            "id": user["id"],
            "email": user["email"],
            "created_at": user["created_at"]
        }

    # ==================== Conversation History Endpoints ====================

    class ConversationRequest(BaseModel):
        conversation_id: str
        title: str
        messages: list

    class ConversationUpdateRequest(BaseModel):
        title: Optional[str] = None
        messages: Optional[list] = None

    def get_user_from_auth(authorization: str) -> dict:
        """Helper to extract user from authorization header."""
        if not authorization:
            raise HTTPException(status_code=401, detail="No authorization header")

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header format")

        token = parts[1]
        user_id = auth.get_user_id_from_token(token)

        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    @app.post("/api/conversations")
    async def save_conversation_endpoint(
        request: ConversationRequest,
        authorization: str = Header(None)
    ):
        """Save or create a conversation."""
        user = get_user_from_auth(authorization)

        # Check if conversation exists
        existing = db.get_conversation(request.conversation_id)

        if existing:
            # Update existing conversation
            if existing["user_id"] != user["id"]:
                raise HTTPException(status_code=403, detail="Not authorized to update this conversation")

            db.update_conversation(
                conversation_id=request.conversation_id,
                title=request.title,
                messages=request.messages
            )
            return {"status": "updated", "conversation_id": request.conversation_id}
        else:
            # Create new conversation
            db.save_conversation(
                user_id=user["id"],
                conversation_id=request.conversation_id,
                title=request.title,
                messages=request.messages
            )
            return {"status": "created", "conversation_id": request.conversation_id}

    @app.get("/api/conversations")
    async def get_conversations_endpoint(
        authorization: str = Header(None),
        limit: int = 50,
        offset: int = 0
    ):
        """Get all conversations for the authenticated user."""
        user = get_user_from_auth(authorization)

        conversations = db.get_user_conversations(user["id"], limit=limit, offset=offset)
        return {"conversations": conversations}

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation_endpoint(
        conversation_id: str,
        authorization: str = Header(None)
    ):
        """Get a specific conversation with all messages."""
        user = get_user_from_auth(authorization)

        conversation = db.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized to view this conversation")

        return conversation

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation_endpoint(
        conversation_id: str,
        authorization: str = Header(None)
    ):
        """Delete a conversation."""
        user = get_user_from_auth(authorization)

        conversation = db.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation["user_id"] != user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized to delete this conversation")

        db.delete_conversation(conversation_id)
        return {"status": "deleted", "conversation_id": conversation_id}

# ==================== Main API Endpoints ====================

@app.get("/")
def read_root():
    return {"message": "Consumer Recommendation API", "status": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - searches YouTube, gets transcripts, generates answer."""

    try:
        # Start timing
        total_start = time.time()
        timing = TimingInfo()

        # Build conversation context from history
        history_for_context = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
        conversation_context = build_conversation_context(history_for_context)

        # If search is disabled, always use chat response (for debugging)
        if request.disable_search:
            full_last_msg = get_last_assistant_message(history_for_context) if needs_full_content(request.query) else ""
            chat_response = generate_chat_response(request.query, conversation_context, full_last_msg)
            return ChatResponse(
                answer=chat_response,
                videos=[],
                sources_summary="(Search disabled)",
                debug=None
            )

        # Step 0: Classify query into CHAT, SEARCH, or PRODUCT
        step_start = time.time()
        query_type = classify_query(request.query, conversation_context)
        timing.classify_query_seconds = time.time() - step_start
        print(f"Query: '{request.query}' -> Classified as: {query_type} ({timing.classify_query_seconds:.2f}s)")

        if query_type == "CHAT":
            # Handle as regular conversation - no search needed
            full_last_msg = get_last_assistant_message(history_for_context) if needs_full_content(request.query) else ""
            chat_response = generate_chat_response(request.query, conversation_context, full_last_msg)
            return ChatResponse(
                answer=chat_response,
                videos=[],
                sources_summary="",
                debug=None
            )

        if query_type == "SEARCH":
            # Handle as quick web search - factual/current info lookup
            answer, web_results = generate_search_response(request.query)
            sources_summary = f"Searched {len(web_results)} web sources" if web_results else ""
            return ChatResponse(
                answer=answer,
                videos=[],
                sources_summary=sources_summary,
                debug=None
            )

        # PRODUCT flow: Full product research with YouTube + Web
        # Step 1: Generate optimized search queries for both YouTube and Web
        step_start = time.time()
        search_queries = generate_search_queries(request.query)
        web_search_queries = generate_web_search_queries(request.query)
        timing.generate_queries_seconds = time.time() - step_start
        print(f"Generated queries in {timing.generate_queries_seconds:.2f}s")

        # Step 2: Search YouTube and Web in parallel
        # Start web search in background while we do YouTube search + ranking
        web_executor = ThreadPoolExecutor(max_workers=1)
        web_search_future = web_executor.submit(search_web_batch, web_search_queries, 3)
        web_search_start = time.time()

        # YouTube search - PARALLEL (multiple queries at once)
        step_start = time.time()
        num_queries = len(search_queries)
        videos_per_query = 3
        print(f"Generated {num_queries} YouTube queries, {videos_per_query} videos per query (parallel)")

        # Use parallel batch search for speed
        all_videos, seen_ids = search_youtube_batch(search_queries, max_results_per_query=videos_per_query)

        # Build query_results for debug info
        query_results = []
        for query in search_queries:
            query_videos = [
                VideoInfo(
                    video_id=v["video_id"],
                    title=v["title"],
                    url=v["url"],
                    thumbnail=v["thumbnail"],
                    channel=v["channel"],
                    has_transcript=False
                )
                for v in all_videos if v.get("source_query") == query
            ]
            query_results.append(QueryResult(query=query, videos=query_videos))

        total_videos_found = len(all_videos)
        timing.youtube_search_seconds = time.time() - step_start
        print(f"Total unique videos found: {total_videos_found} ({timing.youtube_search_seconds:.2f}s)")

        # Step 2a: Intelligently rank videos using LLM and select top 10 for transcript generation
        # LLM considers video title, description, and user query to rank by relevance
        # Falls back to heuristic ranking (position + diversity) if LLM fails
        step_start = time.time()
        ranked_videos = rank_videos_with_llm(all_videos, request.query, max_to_select=10)
        timing.video_ranking_seconds = time.time() - step_start
        print(f"Selected {len(ranked_videos)} videos for transcript generation (from {total_videos_found} candidates) ({timing.video_ranking_seconds:.2f}s)")

        # Step 2b: Wait for web search to complete (should be done or nearly done by now)
        try:
            all_web_results, seen_urls = web_search_future.result(timeout=30)
        except Exception as e:
            print(f"Web search error: {e}")
            all_web_results = []
            seen_urls = set()
        finally:
            web_executor.shutdown(wait=False)

        timing.web_search_seconds = time.time() - web_search_start

        # Build web_query_results for debug info
        web_query_results = []
        for query in web_search_queries:
            query_web_results = [
                WebResult(
                    title=r["title"],
                    url=r["url"],
                    content=r.get("content", "")[:500],  # Truncate for debug display
                    score=r.get("score", 0.0)
                )
                for r in all_web_results if r.get("source_query") == query
            ]
            web_query_results.append(WebQueryResult(query=query, results=query_web_results))

        total_web_results = len(all_web_results)
        print(f"Web search completed: {total_web_results} results ({timing.web_search_seconds:.2f}s)")

        # Step 3: Get transcripts for ranked videos (max 10 with transcripts) - PARALLEL
        step_start = time.time()
        max_videos_with_transcripts = 10
        print(f"Getting transcripts for up to {max_videos_with_transcripts} videos from {len(ranked_videos)} ranked candidates (parallel)...")

        # Use parallel fetching for speed improvement
        videos_with_transcripts, videos_with_content = fetch_transcripts_parallel(
            videos=ranked_videos,
            max_transcripts=max_videos_with_transcripts,
            max_workers=5,
            country="US"
        )

        timing.transcript_fetch_seconds = time.time() - step_start
        print(f"Videos with transcripts: {len(videos_with_content)} / {len(videos_with_transcripts)} attempted ({timing.transcript_fetch_seconds:.2f}s)")

        # Check if we have any content to work with
        has_video_content = len(videos_with_content) > 0
        has_web_content = len(all_web_results) > 0

        answer_gen_debug = None
        step_start = time.time()
        if not has_video_content and not has_web_content:
            answer = f"I couldn't find detailed content for your query. Here are some videos I found about: {request.query}"
            answer_gen_debug = AnswerGenerationDebug(
                method_used="none",
                fallback_reason="no_content_available",
                video_insights=None,
                synthesis=None
            )
        else:
            # Step 4: Generate comprehensive answer from both sources
            answer, answer_gen_debug = generate_answer(request.query, videos_with_content, all_web_results)
        timing.answer_generation_seconds = time.time() - step_start
        timing.total_seconds = time.time() - total_start
        print(f"Answer generated ({timing.answer_generation_seconds:.2f}s) | Total time: {timing.total_seconds:.2f}s")

        # Prepare video info for response (with transcript status)
        video_infos = [
            VideoInfo(
                video_id=v["video_id"],
                title=v["title"],
                url=v["url"],
                thumbnail=v["thumbnail"],
                channel=v["channel"],
                has_transcript=v.get("has_transcript", False),
                view_count=v.get("view_count"),
                like_count=v.get("like_count")
            )
            for v in videos_with_transcripts
        ]

        sources_summary = f"Analyzed {len(videos_with_content)} videos and {len(all_web_results)} web articles"

        # Build debug info
        # Build ranked video info for debug panel
        ranked_video_infos = [
            RankedVideoInfo(
                video_id=v["video_id"],
                title=v["title"],
                url=v["url"],
                thumbnail=v["thumbnail"],
                channel=v["channel"],
                description=v.get("description", "")[:200],  # Truncate for display
                source_query=v.get("source_query", ""),
                has_transcript=v.get("has_transcript", False)
            )
            for v in videos_with_transcripts
        ]

        debug_info = DebugInfo(
            generated_queries=search_queries,
            query_results=query_results,
            total_videos_found=total_videos_found,
            videos_with_transcripts=len(videos_with_content),
            videos_analyzed=len(videos_with_content),
            ranked_videos=ranked_video_infos,
            web_queries=web_search_queries,
            web_query_results=web_query_results,
            total_web_results=total_web_results,
            answer_generation=answer_gen_debug,
            timing=timing
        )

        return ChatResponse(
            answer=answer,
            videos=video_infos,
            sources_summary=sources_summary,
            debug=debug_info
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming version of chat endpoint using Server-Sent Events (SSE).

    Event types:
    - progress: {"step": "step_name", "message": "description"}
    - answer_chunk: {"text": "chunk of answer text"}
    - metadata: {"videos": [...], "sources_summary": "...", "debug": {...}}
    - error: {"message": "error description"}
    - done: {}
    """

    def generate_sse():
        try:
            total_start = time.time()
            timing = TimingInfo()

            # Build conversation context from history for follow-up questions
            history_for_context = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
            conversation_context = build_conversation_context(history_for_context)

            # Debug: log conversation context
            print(f"Conversation history: {len(request.conversation_history)} messages")
            if conversation_context:
                print(f"Built context ({len(conversation_context)} chars): {conversation_context[:200]}...")

            # Helper to send SSE event
            def send_event(event_type: str, data: dict):
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            # Helper to send keepalive ping (prevents Railway proxy timeout)
            def send_keepalive():
                return ": keepalive\n\n"

            # Define progress steps for PRODUCT flow (credibility display)
            PROGRESS_STEPS = [
                ("classify", "Analyzing query"),
                ("generate_queries", "Generating search queries"),
                ("youtube_search", "Searching YouTube"),
                ("ranking", "Ranking videos"),
                ("web_search", "Searching web"),
                ("transcripts", "Fetching transcripts"),
                ("insights", "Analyzing videos"),
                ("synthesis", "Synthesizing insights"),
                ("answer", "Writing response"),
            ]

            def send_progress(step_id: str, message: str, detail: str = None):
                """Send enhanced progress event with step index for multi-step UI."""
                step_index = next((i for i, (id, _) in enumerate(PROGRESS_STEPS) if id == step_id), -1)
                return send_event("progress", {
                    "step": step_id,
                    "step_index": step_index,
                    "total_steps": len(PROGRESS_STEPS),
                    "message": message,
                    "detail": detail
                })

            # If search is disabled, stream a simple response
            if request.disable_search:
                yield send_event("progress", {"step": "chat", "message": "Generating response..."})
                full_last_msg = get_last_assistant_message(history_for_context) if needs_full_content(request.query) else ""
                chat_response = generate_chat_response(request.query, conversation_context, full_last_msg)
                yield send_event("answer_chunk", {"text": chat_response})
                yield send_event("done", {})
                return

            # Step 0: Classify query
            yield send_progress("classify", "Analyzing your query")
            step_start = time.time()
            query_type = classify_query(request.query, conversation_context)
            timing.classify_query_seconds = time.time() - step_start
            yield send_progress("classify", "Analyzing your query", f"→ {query_type.capitalize()} research")

            if query_type == "CHAT":
                yield send_event("progress", {"step": "chat", "message": "Generating conversational response..."})
                full_last_msg = get_last_assistant_message(history_for_context) if needs_full_content(request.query) else ""
                chat_response = generate_chat_response(request.query, conversation_context, full_last_msg)
                yield send_event("answer_chunk", {"text": chat_response})
                yield send_event("done", {})
                return

            if query_type == "SEARCH":
                yield send_event("progress", {"step": "search", "message": "Handling search query..."})
                answer, web_results = generate_search_response(request.query, conversation_context)
                sources_summary = f"Searched {len(web_results)} web sources" if web_results else ""
                yield send_event("answer_chunk", {"text": answer})
                yield send_event("metadata", {"videos": [], "sources_summary": sources_summary})
                yield send_event("done", {})
                return

            # PRODUCT flow
            yield send_progress("generate_queries", "Preparing search strategy")
            step_start = time.time()
            search_queries = generate_search_queries(request.query, conversation_context)
            web_search_queries = generate_web_search_queries(request.query)
            timing.generate_queries_seconds = time.time() - step_start
            yield send_progress("generate_queries", "Preparing search strategy", "→ Ready")

            # Start web search in background while we do YouTube search + ranking
            # This overlaps web search with YouTube operations for speed
            web_executor = ThreadPoolExecutor(max_workers=1)
            web_search_future = web_executor.submit(search_web_batch, web_search_queries, 3)
            web_search_start = time.time()

            # YouTube search - PARALLEL (multiple queries at once)
            yield send_progress("youtube_search", "Searching YouTube")
            step_start = time.time()

            # Use parallel batch search for speed
            all_videos, seen_ids = search_youtube_batch(search_queries, max_results_per_query=3)

            # Build query_results for debug info
            query_results = []
            for query in search_queries:
                query_videos = [
                    VideoInfo(
                        video_id=v["video_id"],
                        title=v["title"],
                        url=v["url"],
                        thumbnail=v["thumbnail"],
                        channel=v["channel"]
                    )
                    for v in all_videos if v.get("source_query") == query
                ]
                query_results.append(QueryResult(query=query, videos=query_videos))

            total_videos_found = len(all_videos)
            timing.youtube_search_seconds = time.time() - step_start
            yield send_progress("youtube_search", "Searching YouTube", "→ Complete")

            # Rank videos
            yield send_progress("ranking", "Selecting most relevant videos")
            step_start = time.time()
            ranked_videos = rank_videos_with_llm(all_videos, request.query, max_to_select=request.max_videos)
            timing.video_ranking_seconds = time.time() - step_start
            # Calculate aggregated engagement stats for credibility display
            total_views = sum(v.get("view_count", 0) or 0 for v in ranked_videos)
            total_likes = sum(v.get("like_count", 0) or 0 for v in ranked_videos)
            total_duration_seconds = sum(v.get("duration_seconds", 0) or 0 for v in ranked_videos)
            total_duration_minutes = total_duration_seconds // 60
            yield send_progress("ranking", "Selecting most relevant videos", f"→ {format_number(total_views)} views, {format_number(total_likes)} likes")

            # Wait for web search to complete (should be done or nearly done by now)
            yield send_progress("web_search", "Searching web articles")
            try:
                all_web_results, seen_urls = web_search_future.result(timeout=30)
            except Exception as e:
                print(f"Web search error: {e}")
                all_web_results = []
                seen_urls = set()
            finally:
                web_executor.shutdown(wait=False)

            timing.web_search_seconds = time.time() - web_search_start

            # Build web_query_results for debug info
            web_query_results = []
            for query in web_search_queries:
                query_web_results = [
                    WebResult(
                        title=r["title"], url=r["url"], content=r.get("content", ""), score=r.get("score", 0)
                    )
                    for r in all_web_results if r.get("source_query") == query
                ]
                web_query_results.append(WebQueryResult(query=query, results=query_web_results))

            total_web_results = len(all_web_results)
            yield send_progress("web_search", "Searching web articles", "→ Complete")

            # Fetch transcripts - PARALLEL with progress updates
            total_ranked = len(ranked_videos)
            yield send_progress("transcripts", "Fetching video transcripts", f"→ 0/{total_ranked} fetched")
            step_start = time.time()

            # Use a queue to get progress updates from parallel transcript fetching
            transcript_queue: queue.Queue = queue.Queue()

            def on_transcript_done(fetched: int, valid: int, title: str, is_valid: bool):
                transcript_queue.put((fetched, valid, title, is_valid))

            # Run transcript fetching in a separate thread so we can poll the queue
            transcript_result = [None, None]  # [videos_with_transcripts, videos_with_content]

            def run_transcript_fetch():
                result = fetch_transcripts_parallel(
                    videos=ranked_videos,
                    max_transcripts=10,
                    max_workers=5,
                    country="US",
                    on_transcript_complete=on_transcript_done
                )
                transcript_result[0], transcript_result[1] = result

            transcript_thread = threading.Thread(target=run_transcript_fetch)
            transcript_thread.start()

            # Poll the queue and yield progress updates
            last_keepalive = time.time()
            while transcript_thread.is_alive() or not transcript_queue.empty():
                try:
                    fetched, valid, title, is_valid = transcript_queue.get(timeout=0.5)
                    short_title = title[:30] + "..." if len(title) > 30 else title
                    last_keepalive = time.time()
                    status = "✓" if is_valid else "○"
                    yield send_progress("transcripts", "Fetching video transcripts", f"→ {valid} valid, {status} {short_title}")
                except queue.Empty:
                    # Send keepalive ping every 5 seconds to prevent proxy timeout
                    if time.time() - last_keepalive > 5:
                        yield send_keepalive()
                        last_keepalive = time.time()
                    continue

            transcript_thread.join()
            videos_with_transcripts = transcript_result[0] or []
            videos_with_content = transcript_result[1] or []

            timing.transcript_fetch_seconds = time.time() - step_start
            yield send_progress("transcripts", "Fetching video transcripts", f"→ {len(videos_with_content)} transcripts ready")

            has_video_content = len(videos_with_content) > 0
            has_web_content = len(all_web_results) > 0

            # Generate answer with streaming
            step_start = time.time()

            answer_gen_debug = None
            full_answer = ""

            if not has_video_content and not has_web_content:
                msg = f"I couldn't find detailed content for your query. Here are some videos I found about: {request.query}"
                yield send_event("answer_chunk", {"text": msg})
                full_answer = msg
                answer_gen_debug = AnswerGenerationDebug(
                    method_used="none",
                    fallback_reason="no_content_available",
                    video_insights=None,
                    synthesis=None
                )
            else:
                # Extract video insights (non-streaming part)
                videos_for_insights = [v for v in videos_with_content if v.get("transcript")]

                if videos_for_insights:
                    total_vids = len(videos_for_insights)
                    yield send_progress("insights", "Analyzing video content", f"→ 0/{total_vids} videos analyzed")
                    step_start = time.time()

                    # Use queues to get progress updates and completed insights from parallel extraction
                    progress_queue: queue.Queue = queue.Queue()
                    insights_queue: queue.Queue = queue.Queue()  # For early synthesis

                    def on_video_done(completed: int, total: int, title: str):
                        progress_queue.put((completed, total, title))

                    # Run extraction in a separate thread so we can poll the queue
                    extraction_result = [None]  # Use list to allow mutation from inner function

                    def run_extraction():
                        extraction_result[0] = extract_insights_for_videos(
                            videos_for_insights,
                            on_video_complete=on_video_done,
                            on_insight_ready=lambda insight: insights_queue.put(insight) if insight else None
                        )

                    extraction_thread = threading.Thread(target=run_extraction)
                    extraction_thread.start()

                    # OPTIMIZATION: Start synthesis early once we have 2+ insights
                    # This runs synthesis in parallel with remaining video extractions
                    early_insights: List[dict] = []
                    synthesis_result = [None]
                    synthesis_done = threading.Event()
                    synthesis_thread = None
                    synthesis_started = False
                    MIN_INSIGHTS_FOR_EARLY_SYNTHESIS = 2

                    synthesis_substeps = [
                        "Identifying common themes across reviews",
                        "Comparing product mentions and ratings",
                        "Finding consensus pros and cons",
                        "Detecting controversial opinions",
                        "Building product-by-product summary",
                    ]

                    def run_synthesis_with_insights(insights_to_synthesize: List[dict]):
                        synthesis_result[0] = synthesize_video_insights(insights_to_synthesize)
                        synthesis_done.set()

                    # Poll the queue and yield progress updates
                    last_keepalive = time.time()
                    while extraction_thread.is_alive() or not progress_queue.empty() or not insights_queue.empty():
                        # Collect any ready insights
                        try:
                            while True:
                                insight = insights_queue.get_nowait()
                                if insight:
                                    early_insights.append(insight)

                                    # Start synthesis early if we have enough insights and haven't started yet
                                    if (not synthesis_started and
                                        len(early_insights) >= MIN_INSIGHTS_FOR_EARLY_SYNTHESIS and
                                        total_vids >= MIN_INSIGHTS_FOR_EARLY_SYNTHESIS):
                                        synthesis_started = True
                                        # Start synthesis with current insights (will use cache if available)
                                        synthesis_thread = threading.Thread(
                                            target=run_synthesis_with_insights,
                                            args=(list(early_insights),)  # Copy current insights
                                        )
                                        synthesis_thread.start()
                                        print(f"[Early Synthesis] Started with {len(early_insights)}/{total_vids} videos")
                        except queue.Empty:
                            pass

                        # Check for progress updates
                        try:
                            completed, total, title = progress_queue.get(timeout=0.5)
                            # Truncate title for cleaner display
                            short_title = title[:35] + "..." if len(title) > 35 else title
                            yield send_progress("insights", "Analyzing video content", f"→ {completed}/{total}: {short_title}")
                            last_keepalive = time.time()
                        except queue.Empty:
                            # Send keepalive ping every 5 seconds to prevent proxy timeout
                            if time.time() - last_keepalive > 5:
                                yield send_keepalive()
                                last_keepalive = time.time()
                            continue

                    extraction_thread.join()
                    video_insights = extraction_result[0] or []

                    timing.insights_extraction_seconds = time.time() - step_start
                    yield send_progress("insights", "Analyzing video content", f"→ {len(video_insights)} videos analyzed")

                    # If we started early synthesis but got more videos after, we may need to re-synthesize
                    # But only if we got significantly more insights (50%+ more)
                    should_resynthesize = (
                        synthesis_started and
                        len(video_insights) > len(early_insights) * 1.5 and
                        len(video_insights) > MIN_INSIGHTS_FOR_EARLY_SYNTHESIS
                    )

                    if should_resynthesize:
                        print(f"[Re-synthesis] More videos completed ({len(video_insights)} vs {len(early_insights)}), re-synthesizing")
                        # Wait for early synthesis to finish first
                        if synthesis_thread and synthesis_thread.is_alive():
                            synthesis_thread.join()
                        # Start new synthesis with all insights
                        synthesis_done.clear()
                        synthesis_thread = threading.Thread(
                            target=run_synthesis_with_insights,
                            args=(video_insights,)
                        )
                        synthesis_thread.start()
                else:
                    video_insights = []
                    synthesis_result = [None]
                    synthesis_done = threading.Event()
                    synthesis_thread = None
                    synthesis_started = False
                    synthesis_substeps = []

                # Synthesize insights across videos with animated sub-steps
                if video_insights:
                    synthesis_step_start = time.time()

                    # If synthesis wasn't started early (single video or very few), start it now
                    if not synthesis_started:
                        synthesis_substeps = [
                            "Identifying common themes across reviews",
                            "Comparing product mentions and ratings",
                            "Finding consensus pros and cons",
                            "Detecting controversial opinions",
                            "Building product-by-product summary",
                        ]

                        def run_synthesis():
                            synthesis_result[0] = synthesize_video_insights(video_insights)
                            synthesis_done.set()

                        synthesis_thread = threading.Thread(target=run_synthesis)
                        synthesis_thread.start()

                    # Cycle through sub-steps while waiting for synthesis to complete
                    substep_idx = 0
                    while not synthesis_done.is_set():
                        if synthesis_substeps:
                            current_substep = synthesis_substeps[substep_idx % len(synthesis_substeps)]
                            yield send_progress("synthesis", "Cross-referencing reviewer opinions", f"→ {current_substep}...")
                            substep_idx += 1
                        else:
                            # Send keepalive if no substeps defined
                            yield send_keepalive()
                        # Wait up to 1.5 seconds before showing next sub-step
                        synthesis_done.wait(timeout=1.5)

                    if synthesis_thread:
                        synthesis_thread.join()
                    synthesis = synthesis_result[0]
                    timing.synthesis_seconds = time.time() - synthesis_step_start
                    yield send_progress("synthesis", "Cross-referencing reviewer opinions", "→ Complete")
                else:
                    synthesis = None

                # Build web summaries
                web_summaries = []
                for w in all_web_results[:5]:
                    content = (w.get("content") or "")[:400]
                    web_summaries.append({
                        "title": w.get("title"),
                        "url": w.get("url"),
                        "snippet": content
                    })

                # Transform data to include pre-built markdown links for quotes
                # This ensures timestamps are correctly converted to YouTube URL parameters
                transformed_insights, transformed_synthesis = transform_insights_for_answer_generation(
                    video_insights, synthesis
                )

                # Debug: Log first quote transformation to verify it's working
                if transformed_insights:
                    for ti in transformed_insights:
                        for prod in ti.get('products', []):
                            for quote in prod.get('top_quotes', []):
                                if 'markdown_link' in quote:
                                    print(f"[DEBUG] Transformed quote markdown_link: {quote['markdown_link'][:100]}...")
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break

                # Get the prompts for streaming
                system_prompt = """You are an expert consumer product advisor who has already watched several YouTube review videos and read some web articles for the user.

You have THREE key inputs:
1) Structured insights from individual YouTube videos (including per-product details and quotes with timestamps).
2) An aggregated synthesis across all videos, organized PER PRODUCT and including COMPARISONS (which product is better for whom).
3) Brief summaries of web articles.

Your job is to produce a uniquely rich answer that feels like:
"I watched these videos and read these articles for you."

CRITICAL: The user is asking a concrete question about product recommendations.
You MUST take a stance and give direct, actionable recommendations, not just summarize.

STRICT GUIDELINES:
1. Start with a decision-first section with your top picks
2. Weave in comparisons where relevant
3. Include consensus pros/cons per product
4. Use 4-6 memorable DIRECT QUOTES from videos:
   - CRITICAL: Look for the 'markdown_link' field in each quote object within video_insights.products[].top_quotes[]
   - The markdown_link field contains a READY-TO-USE formatted string like: *"quote text"* — [Channel](url&t=seconds)
   - COPY THE markdown_link VALUE EXACTLY as-is into your response
   - DO NOT construct your own links - use the pre-built markdown_link field
   - Spread quotes across multiple channels (do NOT only quote one channel).
5. Make clear when insights come from video vs web
6. End with concrete recommendations
7. Be concise but specific; avoid generic phrasing

CONVERSATION CONTEXT:
If conversation context is provided, use it to:
- Understand what products/topics were discussed previously
- Resolve pronouns and references (e.g., "it", "the first one", "that camera")
- Build on prior recommendations without repeating information already given
- Acknowledge when answering follow-up questions"""

                # Add conversation context to system prompt if available
                if conversation_context:
                    system_prompt += f"\n\n## Previous Conversation Context\n{conversation_context}"

                user_payload = {
                    "user_question": request.query,
                    "video_insights": transformed_insights,
                    "multi_video_synthesis": transformed_synthesis,
                    "web_articles": web_summaries,
                }

                user_message = (
                    "Here is the structured data you have access to. "
                    "Use it to answer the user's question.\n\n"
                    f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
                    "REMINDER: When including quotes, use the 'markdown_link' field from each quote object. "
                    "It contains a ready-to-use clickable link format."
                )

                # Stream the answer
                yield send_progress("answer", "Writing personalized recommendations")
                for chunk in call_gpt5_streaming("gpt-5.1", system_prompt, user_message):
                    if chunk:
                        full_answer += chunk
                        yield send_event("answer_chunk", {"text": chunk})

                answer_gen_debug = AnswerGenerationDebug(
                    method_used="structured" if synthesis else "structured_no_synthesis",
                    fallback_reason="synthesis_failed" if not synthesis else None,
                    video_insights=video_insights,
                    synthesis=synthesis
                )

            timing.answer_generation_seconds = time.time() - step_start
            timing.total_seconds = time.time() - total_start

            # Prepare video info for response
            video_infos = [
                {
                    "video_id": v["video_id"],
                    "title": v["title"],
                    "url": v["url"],
                    "thumbnail": v["thumbnail"],
                    "channel": v["channel"],
                    "has_transcript": v.get("has_transcript", False),
                    "view_count": v.get("view_count"),
                    "like_count": v.get("like_count")
                }
                for v in videos_with_transcripts
            ]

            # Build ranked video info
            ranked_video_infos = [
                {
                    "video_id": v["video_id"],
                    "title": v["title"],
                    "url": v["url"],
                    "thumbnail": v["thumbnail"],
                    "channel": v["channel"],
                    "description": v.get("description", ""),
                    "source_query": v.get("source_query", ""),
                    "has_transcript": v.get("has_transcript", False)
                }
                for v in ranked_videos
            ]

            sources_summary = f"Analyzed {len(videos_with_content)} videos and {len(all_web_results)} web articles"

            # Build debug info
            debug_info = {
                "generated_queries": search_queries,
                "query_results": [{"query": qr.query, "videos": [{"video_id": v.video_id, "title": v.title, "url": v.url, "thumbnail": v.thumbnail, "channel": v.channel} for v in qr.videos]} for qr in query_results],
                "total_videos_found": total_videos_found,
                "videos_with_transcripts": len(videos_with_content),
                "videos_analyzed": len(videos_with_content),
                "ranked_videos": ranked_video_infos,
                "web_queries": web_search_queries,
                "web_query_results": [{"query": wqr.query, "results": [{"title": r.title, "url": r.url, "content": r.content, "score": r.score} for r in wqr.results]} for wqr in web_query_results],
                "total_web_results": total_web_results,
                "answer_generation": {
                    "method_used": answer_gen_debug.method_used if answer_gen_debug else "none",
                    "fallback_reason": answer_gen_debug.fallback_reason if answer_gen_debug else None,
                    "video_insights": answer_gen_debug.video_insights if answer_gen_debug else None,
                    "synthesis": answer_gen_debug.synthesis if answer_gen_debug else None
                } if answer_gen_debug else None,
                "timing": {
                    "total_seconds": timing.total_seconds,
                    "classify_query_seconds": timing.classify_query_seconds,
                    "generate_queries_seconds": timing.generate_queries_seconds,
                    "youtube_search_seconds": timing.youtube_search_seconds,
                    "video_ranking_seconds": timing.video_ranking_seconds,
                    "web_search_seconds": timing.web_search_seconds,
                    "transcript_fetch_seconds": timing.transcript_fetch_seconds,
                    "insights_extraction_seconds": timing.insights_extraction_seconds,
                    "synthesis_seconds": timing.synthesis_seconds,
                    "answer_generation_seconds": timing.answer_generation_seconds
                }
            }

            # Send final metadata
            metadata_payload = {
                "videos": video_infos,
                "sources_summary": sources_summary,
                "debug": debug_info
            }
            print(f"Sending metadata event with {len(video_infos)} videos, payload size: {len(json.dumps(metadata_payload))} bytes")
            yield send_event("metadata", metadata_payload)

            yield send_event("done", {})
            print(f"Streaming completed | Total time: {timing.total_seconds:.2f}s")

        except Exception as e:
            print(f"Error in streaming chat: {e}")
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ================
# ARTICLE GENERATION (Internal use - for batch processing and database storage)
# ================

def generate_article_content(
    user_query: str,
    video_insights: List[dict],
    synthesis: Optional[dict],
    web_results: List[dict]
) -> str:
    """
    Generate an article from video insights and web data.
    Uses a specific prompt optimized for article generation.
    """
    # Build compact web context
    web_summaries = []
    if web_results:
        for w in web_results[:5]:
            content = (w.get("content") or "")[:400]
            web_summaries.append({
                "title": w.get("title"),
                "url": w.get("url"),
                "snippet": content
            })

    # Transform data to include pre-built markdown links for quotes
    transformed_insights, transformed_synthesis = transform_insights_for_answer_generation(
        video_insights, synthesis
    )

    system_prompt = """You are an expert consumer product critic who has already watched several YouTube review videos and read some web articles for the user.

You have THREE key inputs:
1) Structured insights from individual YouTube videos (including per-product details and quotes with timestamps).
2) An aggregated synthesis across all videos, organized PER PRODUCT and including COMPARISONS (which product is better for whom).
3) Brief summaries of web articles.

Your job is to produce an article that feels like:
"I watched these videos and read these webpages, and performed analysis on the data."

CRITICAL: The objective is writing an article extracted and analyzed from the data.
You MUST make the article unique, insightful and interesting, not just summarize.

STRICT GUIDELINES:
1. Start the article with a compelling title on the FIRST LINE, formatted as: # Title Here
2. Layout the overview of the article after the title
3. Include references. For video references, include timestamps in the article
4. Use 4-6 memorable DIRECT QUOTES from videos:
   - CRITICAL: Look for the 'markdown_link' field in each quote object within video_insights.products[].top_quotes[]
   - The markdown_link field contains a READY-TO-USE formatted string like: *"quote text"* — [Channel](url&t=seconds)
   - COPY THE markdown_link VALUE EXACTLY as-is into your response
   - DO NOT construct your own links - use the pre-built markdown_link field
   - Spread quotes across multiple channels (do NOT only quote one channel).
5. Make clear when insights come from video vs web
6. End with concrete interesting and insightful findings
7. Be concise but specific; avoid generic phrasing

OUTPUT FORMAT: The very first line MUST be the title starting with "# " (markdown h1). Example:
# Best Wireless Earbuds of 2025: What the Experts Actually Recommend

Then continue with the article content."""

    user_payload = {
        "user_question": user_query,
        "video_insights": transformed_insights,
        "multi_video_synthesis": transformed_synthesis,
        "web_articles": web_summaries,
    }

    user_message = (
        "Here is the structured data you have access to. "
        "Use it to write an engaging article.\n\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "REMINDER: When including quotes, use the 'markdown_link' field from each quote object. "
        "It contains a ready-to-use clickable link format."
    )

    return call_gpt5("gpt-5.1", system_prompt, user_message)


# ================
# TRENDING TOPICS DISCOVERY
# ================

def generate_trending_query(topic_name: str, country: str = "US") -> Optional[str]:
    """
    Use LLM to generate a trending/creative YouTube search query for a topic.

    Args:
        topic_name: The topic/category name (e.g., "electronics", "beauty")
        country: Country code to tailor the query

    Returns:
        A creative search query string for YouTube, or None if generation fails
    """
    system_prompt = f"""You are a YouTube search expert. Today's date is {datetime.now().strftime('%B %d, %Y')}.

Given a product category, generate ONE creative and trending YouTube search query that would find popular review/recommendation videos.

Guidelines:
- Focus on what's trending NOW in {country}
- Include specific product names, brands, or recent releases when relevant
- Use terms that YouTube reviewers commonly use
- Make it specific enough to find quality review content
- Keep it concise (under 10 words)

Return ONLY the search query, nothing else."""

    user_prompt = f"Category: {topic_name}"

    try:
        result = call_gpt5("gpt-5-mini", system_prompt, user_prompt)
        return result.strip().strip('"').strip("'")
    except Exception as e:
        print(f"  LLM query generation failed for {topic_name}: {e}")
        return None


def find_trending_content(
    topics: Optional[List[str]] = None,
    timeframe: Optional[str] = "week",
    max_videos_per_topic: int = 10,
    country: Optional[str] = None,
    save_to_db: bool = True,
    query: Optional[str] = None,
    video_duration: Optional[str] = "medium,long",
    use_llm_queries: bool = True
) -> dict:
    """
    Find trending videos across predefined topics using YouTube Topic IDs for precise discovery.

    This function iterates through the specified topics (or all predefined topics),
    uses youtube.search().list() with topicId for high-precision results,
    and fetches web data for each topic.

    Args:
        topics: List of topic names to search (e.g., ["electronics", "beauty"]).
                If None, uses all topics from YOUTUBE_TOPIC_MAP.
        timeframe: Time period to search - "today", "week", "month", "year", or None for all time
        max_videos_per_topic: Maximum number of videos to fetch per topic (default: 10)
        country: ISO 3166-1 alpha-2 country code (e.g., "US", "GB", "JP", "DE", "FR"). If None, uses YouTube's default.
        save_to_db: Whether to save topics to database (default: True)
        query: Custom search query for YouTube. If None, uses "{topic_name} {YOUTUBE_REVIEW_QUERY_SUFFIX}".
        video_duration: Filter by video duration - "short" (<4 min), "medium" (4-20 min), "long" (>20 min),
                        comma-separated for multiple (e.g., "medium,long"), or None for any. Default: "medium,long".
        use_llm_queries: If True, also use LLM-generated queries to find additional content (default: True).
                         Results from LLM queries are merged with default query results.

    Returns:
        Dictionary containing:
        - topics: List of topic results with videos and web_data
        - timeframe: The timeframe used
        - published_after: The date filter used
        - videos_analyzed: Total number of videos analyzed
        - debug: Additional debug info
    """
    # Use default topics if none specified
    if topics is None:
        topics = ["Electronics", "EV", "Beauty", "Cameras"]

    print(f"\n=== Finding Trending Topics ===")
    print(f"Topics: {topics}")
    print(f"Timeframe: {timeframe}, Country: {country or 'default'}")

    # Calculate date range based on timeframe
    now = datetime.utcnow()
    if timeframe is None:
        published_after = None
    elif timeframe == "today":
        published_after = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif timeframe == "week":
        published_after = (now - timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif timeframe == "month":
        published_after = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif timeframe == "year":
        published_after = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # Default to week
        published_after = (now - timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        enriched_topics = []
        total_videos = 0

        for topic_name in topics:
            topic_id_config = YOUTUBE_TOPIC_MAP.get(topic_name)
            if not topic_id_config:
                print(f"  Warning: No Topic ID found for '{topic_name}', skipping...")
                continue

            # Normalize topic_ids to a list (supports both single ID and list of IDs)
            topic_ids = topic_id_config if isinstance(topic_id_config, list) else [topic_id_config]
            print(f"  Fetching videos for '{topic_name}' (Topic IDs: {topic_ids})...")

            # Track LLM query used for this topic
            llm_query_used = None

            # Use youtube.search().list() with topicId for precise results
            try:
                # Use custom query if provided, otherwise default to topic_name + suffix
                effective_query = query if query is not None else f"{topic_name} {YOUTUBE_REVIEW_QUERY_SUFFIX}"

                # Handle video_duration - may be comma-separated for multiple durations
                topic_videos = []
                seen_video_ids = set()

                # Calculate videos per topic ID
                videos_per_topic_id = max(1, max_videos_per_topic // len(topic_ids))

                # Search across all topic IDs for this topic
                for topic_id in topic_ids:
                    # Build base request params for this topic ID
                    base_search_params = {
                        "part": "snippet",
                        "topicId": topic_id,
                        "q": effective_query,
                        "type": "video",
                        "order": "viewCount",
                    }
                    if published_after is not None:
                        base_search_params["publishedAfter"] = published_after

                    if video_duration is not None and "," in video_duration:
                        # Multiple durations - make separate API calls and combine results
                        durations = [d.strip() for d in video_duration.split(",")]
                        videos_per_duration = max(1, videos_per_topic_id // len(durations))

                        for duration in durations:
                            search_params = base_search_params.copy()
                            search_params["maxResults"] = videos_per_duration
                            search_params["videoDuration"] = duration

                            request = youtube.search().list(**search_params)
                            response = request.execute()

                            for item in response.get('items', []):
                                video_id = item['id']['videoId']
                                if video_id not in seen_video_ids:
                                    seen_video_ids.add(video_id)
                                    snippet = item['snippet']
                                    topic_videos.append({
                                        "video_id": video_id,
                                        "title": snippet['title'],
                                        "url": f"https://www.youtube.com/watch?v={video_id}",
                                        "thumbnail": snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
                                        "channel": snippet['channelTitle'],
                                        "description": snippet.get('description', ''),
                                        "published_at": snippet.get('publishedAt', '')
                                    })
                    else:
                        # Single duration or None
                        search_params = base_search_params.copy()
                        search_params["maxResults"] = videos_per_topic_id
                        if video_duration is not None:
                            search_params["videoDuration"] = video_duration

                        request = youtube.search().list(**search_params)
                        response = request.execute()

                        for item in response.get('items', []):
                            snippet = item['snippet']
                            video_id = item['id']['videoId']
                            if video_id not in seen_video_ids:
                                seen_video_ids.add(video_id)
                                topic_videos.append({
                                    "video_id": video_id,
                                    "title": snippet['title'],
                                    "url": f"https://www.youtube.com/watch?v={video_id}",
                                    "thumbnail": snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
                                    "channel": snippet['channelTitle'],
                                    "description": snippet.get('description', ''),
                                    "published_at": snippet.get('publishedAt', '')
                                })

                print(f"    Found {len(topic_videos)} videos from default query")

                # LLM-generated query search (additional content)
                llm_query_used = None
                if use_llm_queries and query is None:  # Only if not using custom query
                    print(f"    Generating LLM query for '{topic_name}'...")
                    llm_query = generate_trending_query(topic_name, country or "US")
                    if llm_query and llm_query.lower() != effective_query.lower():
                        llm_query_used = llm_query
                        print(f"    LLM query: '{llm_query}'")

                        # Search with LLM query across all topic IDs
                        llm_videos_added = 0
                        llm_videos_per_topic_id = max(1, (max_videos_per_topic // 2) // len(topic_ids))

                        for topic_id in topic_ids:
                            # Build LLM search params
                            llm_search_params = {
                                "part": "snippet",
                                "topicId": topic_id,
                                "q": llm_query,
                                "type": "video",
                                "order": "viewCount",
                            }
                            if published_after is not None:
                                llm_search_params["publishedAfter"] = published_after

                            if video_duration is not None and "," in video_duration:
                                # Multiple durations
                                durations = [d.strip() for d in video_duration.split(",")]
                                llm_videos_per_duration = max(1, llm_videos_per_topic_id // len(durations))

                                for duration in durations:
                                    llm_params = llm_search_params.copy()
                                    llm_params["maxResults"] = llm_videos_per_duration
                                    llm_params["videoDuration"] = duration

                                    try:
                                        llm_request = youtube.search().list(**llm_params)
                                        llm_response = llm_request.execute()

                                        for item in llm_response.get('items', []):
                                            video_id = item['id']['videoId']
                                            if video_id not in seen_video_ids:
                                                seen_video_ids.add(video_id)
                                                snippet = item['snippet']
                                                topic_videos.append({
                                                    "video_id": video_id,
                                                    "title": snippet['title'],
                                                    "url": f"https://www.youtube.com/watch?v={video_id}",
                                                    "thumbnail": snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
                                                    "channel": snippet['channelTitle'],
                                                    "description": snippet.get('description', ''),
                                                    "published_at": snippet.get('publishedAt', '')
                                                })
                                                llm_videos_added += 1
                                    except Exception as llm_err:
                                        print(f"    Warning: LLM query search failed for duration {duration}: {llm_err}")
                            else:
                                # Single duration or None
                                llm_params = llm_search_params.copy()
                                llm_params["maxResults"] = llm_videos_per_topic_id
                                if video_duration is not None:
                                    llm_params["videoDuration"] = video_duration

                                try:
                                    llm_request = youtube.search().list(**llm_params)
                                    llm_response = llm_request.execute()

                                    for item in llm_response.get('items', []):
                                        video_id = item['id']['videoId']
                                        if video_id not in seen_video_ids:
                                            seen_video_ids.add(video_id)
                                            snippet = item['snippet']
                                            topic_videos.append({
                                                "video_id": video_id,
                                                "title": snippet['title'],
                                                "url": f"https://www.youtube.com/watch?v={video_id}",
                                                "thumbnail": snippet['thumbnails']['medium']['url'] if 'medium' in snippet['thumbnails'] else snippet['thumbnails']['default']['url'],
                                                "channel": snippet['channelTitle'],
                                                "description": snippet.get('description', ''),
                                                "published_at": snippet.get('publishedAt', '')
                                            })
                                            llm_videos_added += 1
                                except Exception as llm_err:
                                    print(f"    Warning: LLM query search failed: {llm_err}")

                        print(f"    Found {llm_videos_added} additional videos from LLM query")
                    else:
                        print(f"    Skipping LLM query (same as default or generation failed)")

                # Fetch video statistics (view_count, like_count) in a batch request
                if topic_videos:
                    video_ids = [v["video_id"] for v in topic_videos]
                    try:
                        stats_request = youtube.videos().list(
                            part="statistics",
                            id=",".join(video_ids)
                        )
                        stats_response = stats_request.execute()

                        # Build a map of video_id -> statistics
                        stats_map = {}
                        for stats_item in stats_response.get("items", []):
                            vid_id = stats_item["id"]
                            stats = stats_item.get("statistics", {})
                            stats_map[vid_id] = {
                                "view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                                "like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else None
                            }

                        # Update topic_videos with statistics
                        for video in topic_videos:
                            vid_stats = stats_map.get(video["video_id"], {})
                            video["view_count"] = vid_stats.get("view_count", 0)
                            video["like_count"] = vid_stats.get("like_count")

                        print(f"    Fetched statistics for {len(stats_map)} videos")
                    except Exception as stats_err:
                        print(f"    Warning: Could not fetch video statistics: {stats_err}")
                        # Continue without statistics - they'll be null in DB

                total_videos += len(topic_videos)
                print(f"    Found {len(topic_videos)} videos for '{topic_name}'")

            except Exception as api_err:
                print(f"    Error fetching videos for '{topic_name}': {api_err}")
                topic_videos = []

            # Fetch web data for this topic
            print(f"    Fetching web data for topic: {topic_name}...")
            web_query = query if query is not None else f"{topic_name} {YOUTUBE_REVIEW_QUERY_SUFFIX}"
            web_results = search_web(web_query, max_results=5, country=country)
            web_data = [
                {
                    "title": w.get("title", ""),
                    "url": w.get("url", ""),
                    "content": w.get("content", ""),
                    "score": w.get("score", 0.0)
                }
                for w in web_results
            ]

            enriched_topic = {
                "topic_name": topic_name,
                "topic_id": topic_id,
                "video_ids": topic_videos,
                "web_data": web_data,
                "llm_query_used": llm_query_used
            }

            # Save to database if requested
            if save_to_db:
                try:
                    topic_db_id = db.save_topic(
                        topic_name=topic_name,
                        description=f"Trending {topic_name} content",
                        topic_videos=topic_videos,
                        web_data=web_data,
                        timeframe=timeframe,
                        country=country,
                        published_after=published_after
                    )
                    enriched_topic["db_id"] = topic_db_id
                    print(f"    💾 Saved topic to database: ID={topic_db_id}")
                except Exception as db_err:
                    print(f"    ⚠ Database save failed for topic '{topic_name}': {db_err}")

            enriched_topics.append(enriched_topic)

        return {
            "topics": enriched_topics,
            "timeframe": timeframe,
            "published_after": published_after,
            "videos_found": total_videos,
            "debug": {
                "topics_searched": topics,
                "topic_map_used": {t: YOUTUBE_TOPIC_MAP.get(t) for t in topics}
            }
        }

    except Exception as e:
        print(f"Error finding trending topics: {e}")
        traceback.print_exc()
        return {
            "topics": [],
            "timeframe": timeframe,
            "videos_found": 0,
            "error": str(e)
        }


# ================
# BATCH ARTICLE GENERATION FROM TRENDING TOPICS
# ================

def generate_articles_from_trending_content(
    topics: List[str] = None,
    timeframe: Optional[str] = "week",
    max_videos_per_topic: int = 10,
    country: str = "US",
    parallel_articles: int = 3,
    parallel_topics: bool = True,
    skip_synthesis: bool = False,
    use_cache: bool = False,
    cache_ttl_minutes: int = 60,
    save_to_db: bool = True
) -> dict:
    """
    Generate articles for trending topics using YouTube Topic IDs.

    For each topic:
    1. Find trending videos using find_trending_content with YouTube Topic IDs
    2. Generate an article for each topic
    3. Save to database (if save_to_db=True)

    Args:
        topics: List of topics to process (e.g., ["electronics", "beauty"]).
                If None, uses all topics from YOUTUBE_TOPIC_MAP.
        timeframe: Time period for trending topics - "today", "week", "month", "year", or None for all time (default: "week")
        max_videos_per_topic: Maximum videos to fetch per topic (default: 10)
        country: ISO 3166-1 alpha-2 country code (default: "US")
        parallel_articles: Number of articles to generate in parallel (default: 3)
        parallel_topics: Process topics in parallel (default: True)
        skip_synthesis: Skip cross-video synthesis for faster generation (default: False)
        use_cache: Use cached trending topics if available (default: True)
        cache_ttl_minutes: Cache time-to-live in minutes (default: 60)
        save_to_db: Save generated articles to database (default: True)

    Returns:
        Dictionary containing:
        - articles: List of generated article dicts with topic, article content, videos, etc.
        - summary: Summary statistics
        - errors: List of any errors encountered
        - batch_run_id: Database batch run ID (if save_to_db=True)
    """
    if topics is None:
        topics = list(YOUTUBE_TOPIC_MAP.keys())

    print(f"\n{'='*60}")
    print(f"BATCH ARTICLE GENERATION FROM TRENDING TOPICS")
    print(f"{'='*60}")
    print(f"Topics: {topics}")
    print(f"Timeframe: {timeframe}, Max Videos per Topic: {max_videos_per_topic}")
    print(f"Country: {country}")
    print(f"Optimizations: parallel_articles={parallel_articles}, parallel_topics={parallel_topics}, skip_synthesis={skip_synthesis}, use_cache={use_cache}")
    print(f"Save to database: {save_to_db}")

    # Calculate published_after based on timeframe
    now = datetime.utcnow()
    if timeframe is None:
        published_after = None
    elif timeframe == "today":
        published_after = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif timeframe == "week":
        published_after = (now - timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif timeframe == "month":
        published_after = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif timeframe == "year":
        published_after = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # Default to week
        published_after = (now - timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    total_start = time.time()
    all_articles = []
    errors = []
    batch_run_id = None

    # Generation parameters for database
    generation_params = {
        "timeframe": timeframe,
        "max_videos_per_topic": max_videos_per_topic,
        "country": country,
        "parallel_articles": parallel_articles,
        "parallel_topics": parallel_topics,
        "skip_synthesis": skip_synthesis,
        "published_after": published_after
    }

    # Start batch run in database
    if save_to_db:
        batch_run_id = db.start_batch_run(topics, generation_params)
        print(f"Database batch run started: ID={batch_run_id}")

    # Step 1: Find trending topics using YouTube Topic IDs (with caching)
    cache_key = f"trending_{','.join(topics)}_{timeframe}_{country}"
    cached_result = None

    if use_cache:
        cached_result = _get_cached_trending_content(cache_key, cache_ttl_minutes)

    if cached_result:
        trending_result = cached_result
        print(f"Using cached trending topics")
    else:
        trending_result = find_trending_content(
            topics=topics,
            timeframe=timeframe,
            max_videos_per_topic=max_videos_per_topic,
            country=country,
            save_to_db=save_to_db
        )
        if use_cache:
            _cache_trending_content(cache_key, trending_result)

    discovered_topics = trending_result.get("topics", [])
    print(f"Found {len(discovered_topics)} topics with videos")

    if not discovered_topics:
        errors.append({
            "error": "No topics with trending content found",
            "stage": "find_trending_content"
        })
        return {
            "articles": [],
            "batch_run_id": batch_run_id,
            "summary": {
                "topics_processed": 0,
                "total_articles": 0,
                "total_errors": 1,
                "total_time_seconds": time.time() - total_start
            },
            "errors": errors
        }

    # Step 2: Generate article for each discovered topic content
    def generate_article(topic: dict) -> dict:
        """Generate article for a topic using its video_ids and web_data."""
        topic_name = topic.get("topic_name", "Unknown Topic")
        video_ids = topic.get("video_ids", [])
        web_data = topic.get("web_data", [])

        print(f"\n=== Article Generation ===")
        print(f"Topic: {topic_name}")
        print(f"Input: {len(video_ids)} videos, {len(web_data)} web sources")

        if not video_ids:
            return {"error": "No video IDs found for topic", "topic": topic_name}

        try:
            # Start timing
            total_start = time.time()

            # 1. Plan article content - determine idea and select which content to use
            print("Step 1: Planning article content...")
            plan_start = time.time()
            content_plan = plan_article_content(topic_name, video_ids, web_data, country)
            article_idea = content_plan["article_idea"]
            selected_videos = [video_ids[i] for i in content_plan["selected_video_indices"]]
            selected_web_data = [web_data[i] for i in content_plan["selected_web_indices"]]
            print(f"  Article idea: {article_idea}")
            print(f"  Selected {len(selected_videos)} videos, {len(selected_web_data)} web sources")
            print(f"  Content strategy: {content_plan['content_strategy']}")
            print(f"  Planning took: {time.time() - plan_start:.2f}s")

            # 2. Fetch transcripts in parallel
            print("Step 2: Fetching transcripts...")
            transcript_start = time.time()

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_video = {
                    executor.submit(get_transcript, v["video_id"]): v
                    for v in selected_videos
                }
                for future in as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        transcript = future.result()
                        if transcript:
                            # Check if transcript matches expected language for country
                            if is_transcript_language_match(transcript, country):
                                video["transcript"] = transcript
                                video["has_transcript"] = True
                            else:
                                print(f"  Skipping wrong-language video: {video['video_id']} ({video.get('title', '')[:40]}...)")
                                video["transcript"] = None
                                video["has_transcript"] = False
                        else:
                            video["transcript"] = None
                            video["has_transcript"] = False
                    except Exception as e:
                        print(f"  Transcript failed for {video['video_id']}: {e}")
                        video["transcript"] = None
                        video["has_transcript"] = False

            videos_with_content = [v for v in selected_videos if v.get("transcript")]
            print(f"  Got {len(videos_with_content)}/{len(selected_videos)} transcripts matching {country} language")
            print(f"  Transcript fetching took: {time.time() - transcript_start:.2f}s")

            # 3. Extract insights from videos in parallel
            print("Step 3: Extracting video insights...")
            insights_start = time.time()
            video_insights = extract_insights_for_videos(videos_with_content)
            print(f"  Extracted insights from {len(video_insights)} videos")
            print(f"  Insight extraction took: {time.time() - insights_start:.2f}s")

            # 4. Synthesize insights across videos (optional - can skip for faster generation)
            synthesis = None
            if skip_synthesis:
                print("Step 4: Skipping synthesis (skip_synthesis=True)")
            else:
                print("Step 4: Synthesizing insights...")
                synthesis_start = time.time()
                synthesis = synthesize_video_insights(video_insights) if video_insights else None
                print(f"  Synthesis took: {time.time() - synthesis_start:.2f}s")

            # 5. Generate the article using article_idea and selected_web_data
            print("Step 5: Generating article...")
            article_start = time.time()
            article_raw = generate_article_content(
                article_idea,
                video_insights,
                synthesis,
                selected_web_data
            )
            print(f"  Article generation took: {time.time() - article_start:.2f}s")

            # Extract title from article content (first line starting with "# ")
            article_lines = article_raw.strip().split('\n')
            if article_lines and article_lines[0].startswith('# '):
                article_title = article_lines[0][2:].strip()  # Remove "# " prefix
                article = '\n'.join(article_lines[1:]).strip()  # Rest of article without title
            else:
                # Fallback if title format not found
                article_title = article_idea
                article = article_raw
            print(f"  Extracted title: {article_title}")

            total_time = time.time() - total_start
            print(f"\n=== Article Generation Complete ===")
            print(f"Total time: {total_time:.2f}s")

            # Build response
            video_infos = [
                {
                    "video_id": v["video_id"],
                    "title": v["title"],
                    "url": v["url"],
                    "thumbnail": v.get("thumbnail", ""),
                    "channel": v.get("channel", ""),
                    "has_transcript": v.get("has_transcript", False),
                    "view_count": v.get("view_count"),
                    "like_count": v.get("like_count")
                }
                for v in selected_videos
            ]

            web_sources = [
                {
                    "title": w.get("title", ""),
                    "url": w.get("url", ""),
                    "content": w.get("content", ""),
                    "score": w.get("score", 0.0)
                }
                for w in selected_web_data
            ]

            debug_info = {
                "input_videos": len(video_ids),
                "input_web_sources": len(web_data),
                "selected_videos": len(selected_videos),
                "selected_web_sources": len(selected_web_data),
                "videos_analyzed": len(videos_with_content),
                "insights_extracted": len(video_insights),
                "article_idea": article_idea,
                "article_title": article_title,
                "content_strategy": content_plan.get("content_strategy", ""),
                "timing": {
                    "total_seconds": total_time,
                },
                "video_insights": video_insights,
                "synthesis": synthesis
            }

            print(f"  ✓ Article generated for: {topic_name}")

            article_data = {
                "topic": topic,
                "article": article,
                "videos": video_infos,
                "web_sources": web_sources,
                "debug": debug_info
            }

            # Save to database
            if save_to_db:
                try:
                    article_id = db.save_article(
                        topic=topic_name,
                        article_title=article_title,
                        article_content=article,
                        videos=video_infos,
                        web_sources=web_sources,
                        topic_data=topic,
                        debug=debug_info,
                        generation_params=generation_params
                    )
                    article_data["db_id"] = article_id
                    print(f"  💾 Saved to database: ID={article_id}")
                except Exception as db_err:
                    print(f"  ⚠ Database save failed: {db_err}")

            return article_data

        except Exception as e:
            print(f"  ✗ Error generating article for '{topic_name}': {e}")
            traceback.print_exc()
            return {"error": str(e), "topic": topic_name}

    # Use ThreadPoolExecutor for parallel article generation
    print(f"\nGenerating {len(discovered_topics)} articles in parallel (max {parallel_articles} concurrent)...")
    with ThreadPoolExecutor(max_workers=parallel_articles) as executor:
        future_to_topic = {executor.submit(generate_article, topic): topic for topic in discovered_topics}

        for future in as_completed(future_to_topic):
            result = future.result()
            if result.get("error"):
                errors.append({
                    "topic": result.get("topic", "Unknown"),
                    "error": result["error"],
                    "stage": "generate_article"
                })
            else:
                all_articles.append(result)

    total_time = time.time() - total_start

    # Save errors to database and complete batch run
    if save_to_db and batch_run_id:
        for error in errors:
            db.save_batch_error(batch_run_id, error)
        db.complete_batch_run(batch_run_id, len(all_articles), len(errors), total_time)
        print(f"Database batch run completed: ID={batch_run_id}")

    print(f"\n{'='*60}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total articles generated: {len(all_articles)}")
    print(f"Errors encountered: {len(errors)}")
    print(f"Total time: {total_time:.2f}s")
    if save_to_db:
        print(f"Database batch run ID: {batch_run_id}")

    return {
        "articles": all_articles,
        "batch_run_id": batch_run_id,
        "summary": {
            "topics_processed": len(discovered_topics),
            "total_articles": len(all_articles),
            "total_errors": len(errors),
            "total_time_seconds": total_time,
            "published_after": published_after,
            "parameters": {
                "timeframe": timeframe,
                "max_videos_per_topic": max_videos_per_topic,
                "country": country,
                "parallel_articles": parallel_articles,
                "parallel_topics": parallel_topics,
                "skip_synthesis": skip_synthesis,
                "use_cache": use_cache,
                "save_to_db": save_to_db
            }
        },
        "errors": errors
    }


# Trending content cache
_trending_content_cache: dict = {}


def _get_cached_trending_content(cache_key: str, ttl_minutes: int) -> Optional[dict]:
    """Get cached trending content if not expired."""
    if cache_key in _trending_content_cache:
        cached = _trending_content_cache[cache_key]
        cache_time = cached.get("_cache_time")
        if cache_time and (datetime.utcnow() - cache_time).total_seconds() < ttl_minutes * 60:
            return cached.get("data")
    return None


def _cache_trending_content(cache_key: str, data: dict) -> None:
    """Cache trending content with timestamp."""
    _trending_content_cache[cache_key] = {
        "data": data,
        "_cache_time": datetime.utcnow()
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/debug/classify")
def debug_classify(query: str):
    """Debug endpoint to test query classification."""
    try:
        result = classify_query(query)
        return {"query": query, "classification": result}
    except Exception as e:
        return {"query": query, "error": str(e)}

@app.get("/debug/db")
def debug_db():
    """Debug endpoint to check database status."""
    import os
    import sqlite3
    bundled_db = os.path.join(os.path.dirname(__file__), "articles.db")

    # Check file sizes
    db_size = os.path.getsize(db.DB_PATH) if os.path.exists(db.DB_PATH) else 0
    bundled_size = os.path.getsize(bundled_db) if os.path.exists(bundled_db) else 0

    # Count articles in both databases
    article_count = 0
    bundled_count = 0
    try:
        conn = sqlite3.connect(db.DB_PATH)
        article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
    except Exception as e:
        article_count = f"error: {e}"

    try:
        conn = sqlite3.connect(bundled_db)
        bundled_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
    except Exception as e:
        bundled_count = f"error: {e}"

    return {
        "DB_PATH": db.DB_PATH,
        "DB_PATH_exists": os.path.exists(db.DB_PATH),
        "DB_PATH_size": db_size,
        "DB_PATH_article_count": article_count,
        "bundled_db_path": bundled_db,
        "bundled_db_exists": os.path.exists(bundled_db),
        "bundled_db_size": bundled_size,
        "bundled_db_article_count": bundled_count,
        "DATABASE_PATH_env": os.environ.get("DATABASE_PATH", "not set"),
    }


# ================
# ARTICLES API ENDPOINTS (for landing page)
# ================

@app.get("/api/articles/recent")
def get_recent_articles(days: int = 7, limit: int = 10, article_type: Optional[str] = None):
    """
    Get recent articles for the landing page 'Today's Pulse' section.

    Args:
        days: Number of days to look back (default 7)
        limit: Maximum number of articles (default 10)
        article_type: Filter by type ('trending', 'evergreen', or None for all)
    """
    try:
        articles = db.get_recent_articles(days=days, limit=limit, article_type=article_type)

        # Format articles for frontend
        formatted_articles = []
        for article in articles:
            # Get the full article to access videos
            full_article = db.get_article(article['id'])
            videos = full_article.get('videos', []) if full_article else []

            # Get representative thumbnail (first video with thumbnail)
            thumbnail = None
            for video in videos:
                if video.get('thumbnail'):
                    thumbnail = video['thumbnail']
                    break

            formatted_articles.append({
                "id": article['id'],
                "title": article['article_title'],
                "topic": article['topic'],
                "article_type": article.get('article_type', 'trending'),
                "created_at": article['created_at'],
                "videos_analyzed": len(videos),
                "thumbnail": thumbnail,
                "article_preview": (article.get('article_content') or '')[:200] + "..."
            })

        return {
            "articles": formatted_articles,
            "total": len(formatted_articles)
        }

    except Exception as e:
        print(f"Error fetching recent articles: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles/{article_id}")
def get_article_by_id(article_id: int):
    """Get a single article by ID."""
    try:
        article = db.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching article {article_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/articles/stats")
def get_articles_stats():
    """Get database statistics for the landing page."""
    try:
        stats = db.get_statistics()
        return stats
    except Exception as e:
        print(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================
# EVERGREEN ARTICLE GENERATION
# ================

class EvergreenRequest(BaseModel):
    """Request body for evergreen article generation."""
    query: str
    topic: str  # Required topic category (e.g., "EV", "Cameras", "Audio")
    max_videos: int = 10


class EvergreenResponse(BaseModel):
    """Response for evergreen article generation."""
    article_id: int
    title: str
    content: str
    videos: List[dict]
    web_sources: List[dict]
    generation_time_seconds: float


@app.post("/api/articles/evergreen/generate", response_model=EvergreenResponse)
def generate_evergreen_article(request: EvergreenRequest):
    """
    Generate an evergreen article from a query.

    Uses the same pipeline as the PRODUCT chat flow but saves as an article.
    This endpoint is for manually triggered article generation.
    """
    total_start = time.time()

    try:
        print(f"\n=== Generating Evergreen Article ===")
        print(f"Query: {request.query}")
        print(f"Topic: {request.topic}")
        print(f"Max videos: {request.max_videos}")

        # Step 1: Generate search queries
        print("Step 1: Generating search queries...")
        search_queries = generate_search_queries(request.query)
        web_search_queries = generate_web_search_queries(request.query)
        print(f"  YouTube queries: {search_queries}")
        print(f"  Web queries: {web_search_queries}")

        # Step 2: Search YouTube
        print("Step 2: Searching YouTube...")
        all_videos = []
        seen_ids = set()

        for query in search_queries:
            results = search_youtube(query, max_results=5)
            for v in results:
                if v["video_id"] not in seen_ids:
                    seen_ids.add(v["video_id"])
                    v["source_query"] = query
                    all_videos.append(v)

        print(f"  Found {len(all_videos)} unique videos")

        # Step 3: Rank videos
        print("Step 3: Ranking videos...")
        ranked_videos = rank_videos_with_llm(all_videos, request.query, max_to_select=request.max_videos)
        print(f"  Selected {len(ranked_videos)} videos")

        # Step 3b: Fetch video statistics (view_count, like_count)
        if ranked_videos:
            video_ids = [v["video_id"] for v in ranked_videos]
            try:
                stats_request = youtube.videos().list(
                    part="statistics",
                    id=",".join(video_ids)
                )
                stats_response = stats_request.execute()

                # Build a map of video_id -> statistics
                stats_map = {}
                for stats_item in stats_response.get("items", []):
                    vid_id = stats_item["id"]
                    stats = stats_item.get("statistics", {})
                    stats_map[vid_id] = {
                        "view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                        "like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else None
                    }

                # Update ranked_videos with statistics
                for video in ranked_videos:
                    vid_stats = stats_map.get(video["video_id"], {})
                    video["view_count"] = vid_stats.get("view_count", 0)
                    video["like_count"] = vid_stats.get("like_count")

                print(f"  Fetched statistics for {len(stats_map)} videos")
            except Exception as stats_err:
                print(f"  Warning: Could not fetch video statistics: {stats_err}")
                # Continue without statistics - they'll be null in DB

        # Step 4: Search web
        print("Step 4: Searching web...")
        all_web_results = []
        seen_urls = set()

        for query in web_search_queries:
            results = search_web(query, max_results=3)
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_web_results.append(r)

        print(f"  Found {len(all_web_results)} unique web results")

        # Step 5: Fetch transcripts
        print("Step 5: Fetching transcripts...")
        videos_with_content = []

        for video in ranked_videos:
            transcript = get_transcript(video["video_id"])
            is_valid = transcript is not None and is_transcript_language_match(transcript, "US")

            if is_valid:
                video["transcript"] = transcript
                video["has_transcript"] = True
                videos_with_content.append(video)
                if len(videos_with_content) >= 10:
                    break
            else:
                video["has_transcript"] = False

        print(f"  Got {len(videos_with_content)} transcripts")

        # Step 6: Extract insights
        print("Step 6: Extracting video insights...")
        video_insights = []
        if videos_with_content:
            video_insights = extract_insights_for_videos(videos_with_content)
        print(f"  Extracted insights from {len(video_insights)} videos")

        # Step 7: Synthesize insights
        print("Step 7: Synthesizing insights...")
        synthesis = None
        if video_insights:
            synthesis = synthesize_video_insights(video_insights)
        print(f"  Synthesis complete: {synthesis is not None}")

        # Step 8: Generate article content
        print("Step 8: Generating article content...")
        article_content = generate_article_content(
            user_query=request.query,
            video_insights=video_insights,
            synthesis=synthesis,
            web_results=all_web_results
        )

        # Extract title from content (first line starting with #)
        title_match = re.match(r'^#\s*(.+?)(?:\n|$)', article_content.strip())
        article_title = title_match.group(1).strip() if title_match else f"Article: {request.query[:50]}"
        print(f"  Title: {article_title}")

        # Step 9: Save to database
        print("Step 9: Saving to database...")

        # Prepare video data for storage
        videos_for_db = []
        for v in videos_with_content:
            videos_for_db.append({
                "video_id": v["video_id"],
                "title": v.get("title", ""),
                "url": v.get("url", ""),
                "channel": v.get("channel", ""),
                "thumbnail": v.get("thumbnail", ""),
                "has_transcript": v.get("has_transcript", False),
                "relevance_score": v.get("relevance_score", 0),
                "view_count": v.get("view_count", 0),
                "like_count": v.get("like_count")
            })

        # Prepare web sources for storage
        web_sources_for_db = []
        for w in all_web_results[:5]:
            web_sources_for_db.append({
                "title": w.get("title", ""),
                "url": w.get("url", ""),
                "content": (w.get("content") or "")[:500]
            })

        # Debug info
        debug_info = {
            "search_queries": search_queries,
            "web_search_queries": web_search_queries,
            "videos_found": len(all_videos),
            "videos_ranked": len(ranked_videos),
            "videos_with_transcripts": len(videos_with_content),
            "video_insights_count": len(video_insights),
            "synthesis_present": synthesis is not None
        }

        article_id = db.save_article(
            topic=request.topic,
            article_title=article_title,
            article_content=article_content,
            videos=videos_for_db,
            web_sources=web_sources_for_db,
            topic_data={"query": request.query},
            debug=debug_info,
            generation_params={"max_videos": request.max_videos, "topic": request.topic},
            article_type="evergreen"
        )

        total_time = time.time() - total_start
        print(f"  Saved as article ID: {article_id}")
        print(f"=== Evergreen Article Complete ({total_time:.2f}s) ===\n")

        return EvergreenResponse(
            article_id=article_id,
            title=article_title,
            content=article_content,
            videos=videos_for_db,
            web_sources=web_sources_for_db,
            generation_time_seconds=total_time
        )

    except Exception as e:
        print(f"Error generating evergreen article: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================
# USAGE TRACKING ENDPOINTS
# ================

class EventTrackRequest(BaseModel):
    """Request body for tracking user events."""
    event_type: str  # 'article_view', 'video_click', 'query', 'conversation_start'
    event_data: dict
    session_id: Optional[str] = None


@app.post("/api/events/track")
def track_event(request: EventTrackRequest):
    """
    Track a user behavior event.

    This is a fire-and-forget endpoint for frontend tracking.
    """
    try:
        # For now, we don't have user authentication context here
        # In a real app, you'd extract user_id from the auth token
        db.log_user_event(
            user_id=None,  # Anonymous for now
            session_id=request.session_id,
            event_type=request.event_type,
            event_data=request.event_data
        )
        return {"status": "ok"}
    except Exception as e:
        # Don't fail the request - tracking should be silent
        print(f"Warning: Failed to track event: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/usage/report")
def get_usage_report(days: int = 30):
    """
    Get API usage and event report.

    Args:
        days: Number of days to look back (default 30)

    Returns:
        Usage report with costs by model, user, request type, and event counts
    """
    try:
        report = db.get_usage_report(days=days)
        return report
    except Exception as e:
        print(f"Error fetching usage report: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/usage/user/{user_id}")
def get_user_usage(user_id: int, days: int = 30):
    """
    Get API usage summary for a specific user.

    Args:
        user_id: The user's ID
        days: Number of days to look back (default 30)

    Returns:
        User's usage summary with token counts and costs
    """
    try:
        summary = db.get_user_usage_summary(user_id=user_id, days=days)
        return summary
    except Exception as e:
        print(f"Error fetching user usage: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/events")
def get_events(
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    days: int = 7,
    limit: int = 100
):
    """
    Get user events with optional filtering.

    Args:
        user_id: Filter by user ID (optional)
        session_id: Filter by session ID (optional)
        event_type: Filter by event type (optional)
        days: Number of days to look back (default 7)
        limit: Maximum number of events to return (default 100)

    Returns:
        List of user events
    """
    try:
        events = db.get_user_events(
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            days=days,
            limit=limit
        )
        return {"events": events, "count": len(events)}
    except Exception as e:
        print(f"Error fetching events: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
