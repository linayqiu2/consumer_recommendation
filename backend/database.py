"""
Database module for storing generated articles and related data.
Uses SQLite for local storage.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

# Database file path - use environment variable for production, default for local
DB_PATH = os.environ.get("DATABASE_PATH", "articles.db")

# On first deploy, copy bundled database to volume if it's empty or doesn't exist
def _init_database_file():
    """Copy bundled articles.db to volume mount if needed."""
    if DB_PATH != "articles.db":
        bundled_db = os.path.join(os.path.dirname(__file__), "articles.db")
        if os.path.exists(bundled_db):
            # Check if volume DB is missing or smaller than bundled (likely empty)
            bundled_size = os.path.getsize(bundled_db)
            volume_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

            if volume_size < bundled_size:
                import shutil
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                shutil.copy2(bundled_db, DB_PATH)
                print(f"Copied bundled database ({bundled_size} bytes) to {DB_PATH}")

_init_database_file()

# OpenAI cost per 1K tokens (as of 2024)
OPENAI_COSTS = {
    'gpt-5-mini': {'input': 0.00015, 'output': 0.0006},
    'gpt-5.1': {'input': 0.003, 'output': 0.015},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
    'gpt-4o': {'input': 0.0025, 'output': 0.01},
}


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
    finally:
        conn.close()


def _run_migrations_internal(cursor):
    """
    Internal migration function that uses an existing cursor.
    Called during init_database() to add new columns to existing tables.
    """
    # Check if article_type column exists in articles table
    cursor.execute("PRAGMA table_info(articles)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'article_type' not in columns:
        print("Migration: Adding article_type column to articles table...")
        cursor.execute("""
            ALTER TABLE articles ADD COLUMN article_type TEXT DEFAULT 'trending'
        """)
        # Set all existing articles to 'trending'
        cursor.execute("""
            UPDATE articles SET article_type = 'trending' WHERE article_type IS NULL
        """)
        print("Migration: article_type column added successfully.")

    # Check if cost and user_email columns exist in chat_queries table
    cursor.execute("PRAGMA table_info(chat_queries)")
    chat_columns = [col[1] for col in cursor.fetchall()]

    if 'cost' not in chat_columns:
        print("Migration: Adding cost column to chat_queries table...")
        cursor.execute("""
            ALTER TABLE chat_queries ADD COLUMN cost REAL DEFAULT 0
        """)
        print("Migration: cost column added successfully.")

    if 'user_email' not in chat_columns:
        print("Migration: Adding user_email column to chat_queries table...")
        cursor.execute("""
            ALTER TABLE chat_queries ADD COLUMN user_email TEXT
        """)
        print("Migration: user_email column added successfully.")


def init_database():
    """Initialize the database with required tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Articles table - main table for generated articles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                article_title TEXT NOT NULL,
                topic_data JSON,
                article_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                generation_params JSON,
                status TEXT DEFAULT 'completed',
                article_type TEXT DEFAULT 'trending'
            )
        """)

        # Videos table - videos used for each article
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                url TEXT,
                thumbnail TEXT,
                channel TEXT,
                has_transcript BOOLEAN DEFAULT FALSE,
                view_count INTEGER,
                like_count INTEGER,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)

        # Web sources table - web sources used for each article
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_web_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                title TEXT,
                url TEXT,
                content TEXT,
                score REAL,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)

        # Debug/metadata table - stores debug info and insights
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_debug (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                video_insights JSON,
                synthesis JSON,
                video_search_queries JSON,
                web_search_queries JSON,
                timing JSON,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
            )
        """)

        # Batch runs table - tracks each batch generation run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                topics JSON,
                parameters JSON,
                total_articles INTEGER DEFAULT 0,
                total_errors INTEGER DEFAULT 0,
                total_time_seconds REAL,
                status TEXT DEFAULT 'running'
            )
        """)

        # Errors table - stores errors from batch runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_run_id INTEGER NOT NULL,
                topic TEXT,
                article_title TEXT,
                error_message TEXT,
                stage TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_run_id) REFERENCES batch_runs(id) ON DELETE CASCADE
            )
        """)

        # Topics table - stores topics with their video/web data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_name TEXT NOT NULL,
                description TEXT,
                timeframe TEXT,
                country TEXT,
                topic_videos JSON,
                web_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_after TEXT
            )
        """)

        # Users table - stores user accounts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                oauth_provider TEXT,
                oauth_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Conversation histories table - stores user conversations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_histories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                messages JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # API usage tracking table - tracks token usage and costs per user
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                request_type TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # User events tracking table - tracks user behavior
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Chat queries tracking table - tracks every chat request for debugging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id INTEGER,
                query TEXT NOT NULL,
                query_type TEXT,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                error_traceback TEXT,
                response_length INTEGER,
                videos_used INTEGER,
                duration_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Transcript cache table - persistent cache for YouTube video transcripts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcript_cache (
                video_id TEXT PRIMARY KEY,
                transcript TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Run migrations to add new columns to existing tables
        # This must be done before creating indexes on those columns
        _run_migrations_internal(cursor)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_article_title ON articles(article_title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_type ON articles(article_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_videos_article_id ON article_videos(article_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_web_sources_article_id ON article_web_sources(article_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_topic_name ON topics(topic_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_created_at ON topics(created_at)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_oauth ON users(oauth_provider, oauth_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_histories_user_id ON conversation_histories(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_histories_updated_at ON conversation_histories(updated_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_date ON api_usage(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_model ON api_usage(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_events_user ON user_events(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_events_type ON user_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_events_date ON user_events(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_events_session ON user_events(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcript_cache_created ON transcript_cache(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_queries_status ON chat_queries(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_queries_created ON chat_queries(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_queries_session ON chat_queries(session_id)")

        conn.commit()
        print("Database initialized successfully.")


def migrate_database():
    """
    Run database migrations to add new columns to existing tables.
    Safe to run multiple times - only adds columns if they don't exist.
    Can be called standalone to migrate an existing database.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        _run_migrations_internal(cursor)
        # Also create index (safe if already exists)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_type ON articles(article_type)")
        conn.commit()
        print("Migration check complete.")


def save_article(
    topic: str,
    article_title: str,
    article_content: str,
    videos: List[dict],
    web_sources: List[dict],
    topic_data: dict = None,
    debug: dict = None,
    generation_params: dict = None,
    article_type: str = "trending"
) -> int:
    """
    Save a generated article and its related data to the database.

    Args:
        topic: The category/topic (e.g., "electronics", "beauty")
        article_title: The title of the article
        article_content: The generated article content
        videos: List of video dicts used for the article
        web_sources: List of web source dicts
        topic_data: Additional topic metadata
        debug: Debug information
        generation_params: Parameters used for generation
        article_type: Type of article ('trending' or 'evergreen')

    Returns:
        The ID of the saved article.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Insert article
        cursor.execute("""
            INSERT INTO articles (topic, article_title, topic_data, article_content, generation_params, article_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            topic,
            article_title,
            json.dumps(topic_data) if topic_data else None,
            article_content,
            json.dumps(generation_params) if generation_params else None,
            article_type
        ))

        article_id = cursor.lastrowid

        # Insert videos
        for video in videos:
            cursor.execute("""
                INSERT INTO article_videos
                (article_id, video_id, title, url, thumbnail, channel, has_transcript, view_count, like_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                video.get("video_id"),
                video.get("title"),
                video.get("url"),
                video.get("thumbnail"),
                video.get("channel"),
                video.get("has_transcript", False),
                video.get("view_count"),
                video.get("like_count")
            ))

        # Insert web sources
        for source in web_sources:
            cursor.execute("""
                INSERT INTO article_web_sources (article_id, title, url, content, score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                article_id,
                source.get("title"),
                source.get("url"),
                source.get("content"),
                source.get("score")
            ))

        # Insert debug info
        if debug:
            cursor.execute("""
                INSERT INTO article_debug
                (article_id, video_insights, synthesis, video_search_queries, web_search_queries, timing)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                json.dumps(debug.get("video_insights")),
                json.dumps(debug.get("synthesis")),
                json.dumps(debug.get("video_search_queries")),
                json.dumps(debug.get("web_search_queries")),
                json.dumps(debug.get("timing"))
            ))

        conn.commit()
        return article_id


def start_batch_run(topics: List[str], parameters: dict) -> int:
    """Start a new batch run and return its ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO batch_runs (topics, parameters)
            VALUES (?, ?)
        """, (json.dumps(topics), json.dumps(parameters)))
        conn.commit()
        return cursor.lastrowid


def complete_batch_run(
    batch_run_id: int,
    total_articles: int,
    total_errors: int,
    total_time_seconds: float
):
    """Mark a batch run as completed."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE batch_runs
            SET completed_at = CURRENT_TIMESTAMP,
                total_articles = ?,
                total_errors = ?,
                total_time_seconds = ?,
                status = 'completed'
            WHERE id = ?
        """, (total_articles, total_errors, total_time_seconds, batch_run_id))
        conn.commit()


def save_batch_error(batch_run_id: int, error: dict):
    """Save an error from a batch run."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO batch_errors (batch_run_id, topic, article_title, error_message, stage)
            VALUES (?, ?, ?, ?, ?)
        """, (
            batch_run_id,
            error.get("topic"),
            error.get("article_title"),
            error.get("error"),
            error.get("stage")
        ))
        conn.commit()


def get_article(article_id: int) -> Optional[dict]:
    """Get a single article by ID with all related data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get article
        cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        article_row = cursor.fetchone()

        if not article_row:
            return None

        article = dict(article_row)
        article["topic_data"] = json.loads(article["topic_data"]) if article["topic_data"] else {}
        article["generation_params"] = json.loads(article["generation_params"]) if article["generation_params"] else {}

        # Get videos
        cursor.execute("SELECT * FROM article_videos WHERE article_id = ?", (article_id,))
        article["videos"] = [dict(row) for row in cursor.fetchall()]

        # Get web sources
        cursor.execute("SELECT * FROM article_web_sources WHERE article_id = ?", (article_id,))
        article["web_sources"] = [dict(row) for row in cursor.fetchall()]

        # Get debug info
        cursor.execute("SELECT * FROM article_debug WHERE article_id = ?", (article_id,))
        debug_row = cursor.fetchone()
        if debug_row:
            debug = dict(debug_row)
            debug["video_insights"] = json.loads(debug["video_insights"]) if debug["video_insights"] else None
            debug["synthesis"] = json.loads(debug["synthesis"]) if debug["synthesis"] else None
            debug["video_search_queries"] = json.loads(debug["video_search_queries"]) if debug["video_search_queries"] else None
            debug["web_search_queries"] = json.loads(debug["web_search_queries"]) if debug["web_search_queries"] else None
            debug["timing"] = json.loads(debug["timing"]) if debug["timing"] else None
            article["debug"] = debug

        return article


def get_articles(
    topic: str = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "created_at DESC"
) -> List[dict]:
    """Get articles with optional filtering."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM articles"
        params = []

        if topic:
            query += " WHERE topic = ?"
            params.append(topic)

        query += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        articles = []
        for row in cursor.fetchall():
            article = dict(row)
            article["topic_data"] = json.loads(article["topic_data"]) if article["topic_data"] else {}
            article["generation_params"] = json.loads(article["generation_params"]) if article["generation_params"] else {}
            articles.append(article)

        return articles


def get_articles_by_topic(topic: str) -> List[dict]:
    """Get all articles for a specific topic."""
    return get_articles(topic=topic, limit=1000)


def get_recent_articles(days: int = 7, limit: int = 50, article_type: str = None) -> List[dict]:
    """
    Get articles created within the last N days.

    Args:
        days: Number of days to look back
        limit: Maximum number of articles to return
        article_type: Filter by type ('trending', 'evergreen', or None for all)

    Returns:
        List of article dicts.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if article_type:
            cursor.execute("""
                SELECT * FROM articles
                WHERE created_at >= datetime('now', ?) AND article_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f'-{days} days', article_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM articles
                WHERE created_at >= datetime('now', ?)
                ORDER BY created_at DESC
                LIMIT ?
            """, (f'-{days} days', limit))

        articles = []
        for row in cursor.fetchall():
            article = dict(row)
            article["topic_data"] = json.loads(article["topic_data"]) if article["topic_data"] else {}
            articles.append(article)

        return articles


def search_articles(query: str, limit: int = 50) -> List[dict]:
    """Search articles by article title or content."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM articles
            WHERE article_title LIKE ? OR article_content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit))

        articles = []
        for row in cursor.fetchall():
            article = dict(row)
            article["topic_data"] = json.loads(article["topic_data"]) if article["topic_data"] else {}
            articles.append(article)

        return articles


def get_batch_runs(limit: int = 20) -> List[dict]:
    """Get recent batch runs."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM batch_runs
            ORDER BY started_at DESC
            LIMIT ?
        """, (limit,))

        runs = []
        for row in cursor.fetchall():
            run = dict(row)
            run["topics"] = json.loads(run["topics"]) if run["topics"] else []
            run["parameters"] = json.loads(run["parameters"]) if run["parameters"] else {}
            runs.append(run)

        return runs


def get_batch_run_errors(batch_run_id: int) -> List[dict]:
    """Get all errors for a specific batch run."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM batch_errors
            WHERE batch_run_id = ?
            ORDER BY created_at
        """, (batch_run_id,))
        return [dict(row) for row in cursor.fetchall()]


def delete_article(article_id: int) -> bool:
    """Delete an article and all related data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
        conn.commit()
        return cursor.rowcount > 0


def update_article_content(
    article_id: int,
    new_content: str,
    new_title: str = None
) -> bool:
    """
    Update an article's content and optionally its title.
    Used for manual editing of evergreen articles.

    Args:
        article_id: The article ID to update
        new_content: The new article content
        new_title: Optional new title (if None, title is unchanged)

    Returns:
        True if updated, False if article not found.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if new_title:
            cursor.execute("""
                UPDATE articles
                SET article_content = ?, article_title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_content, new_title, article_id))
        else:
            cursor.execute("""
                UPDATE articles
                SET article_content = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_content, article_id))

        conn.commit()
        return cursor.rowcount > 0


def get_statistics() -> dict:
    """Get database statistics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        stats = {}

        # Total articles
        cursor.execute("SELECT COUNT(*) FROM articles")
        stats["total_articles"] = cursor.fetchone()[0]

        # Articles by topic
        cursor.execute("""
            SELECT topic, COUNT(*) as count
            FROM articles
            GROUP BY topic
        """)
        stats["articles_by_topic"] = {row[0]: row[1] for row in cursor.fetchall()}

        # Articles by type
        cursor.execute("""
            SELECT article_type, COUNT(*) as count
            FROM articles
            GROUP BY article_type
        """)
        stats["articles_by_type"] = {row[0] or 'trending': row[1] for row in cursor.fetchall()}

        # Total videos referenced
        cursor.execute("SELECT COUNT(*) FROM article_videos")
        stats["total_videos"] = cursor.fetchone()[0]

        # Total web sources
        cursor.execute("SELECT COUNT(*) FROM article_web_sources")
        stats["total_web_sources"] = cursor.fetchone()[0]

        # Total batch runs
        cursor.execute("SELECT COUNT(*) FROM batch_runs")
        stats["total_batch_runs"] = cursor.fetchone()[0]

        # Recent articles (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM articles
            WHERE created_at >= datetime('now', '-7 days')
        """)
        stats["articles_last_7_days"] = cursor.fetchone()[0]

        return stats


def save_topic(
    topic_name: str,
    description: str,
    topic_videos: List[dict],
    web_data: List[dict],
    timeframe: str = None,
    country: str = None,
    published_after: str = None
) -> int:
    """
    Save a topic with its associated video IDs and web data.

    Args:
        topic_name: The name of the trending topic
        description: Description of the topic
        topic_videos: List of video info dicts with video_id, title, url, etc.
        web_data: List of web source dicts with title, url, content, etc.
        timeframe: The timeframe used to find the topic
        country: The country code used
        published_after: Date filter used

    Returns:
        The ID of the saved trending topic.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO topics
            (topic_name, description, topic_videos, web_data, timeframe, country, published_after)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            topic_name,
            description,
            json.dumps(topic_videos),
            json.dumps(web_data),
            timeframe,
            country,
            published_after
        ))
        conn.commit()
        return cursor.lastrowid


def get_topic(topic_id: int) -> Optional[dict]:
    """Get a topic by ID with its video and web data."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
        row = cursor.fetchone()
        if not row:
            return None

        topic = dict(row)
        topic["topic_videos"] = json.loads(topic["topic_videos"]) if topic["topic_videos"] else []
        topic["web_data"] = json.loads(topic["web_data"]) if topic["web_data"] else []
        return topic


def get_recent_topics(days: int = 7, limit: int = 50) -> List[dict]:
    """Get recent topics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM topics
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            LIMIT ?
        """, (f'-{days} days', limit))

        topics = []
        for row in cursor.fetchall():
            topic = dict(row)
            topic["topic_videos"] = json.loads(topic["topic_videos"]) if topic["topic_videos"] else []
            topic["web_data"] = json.loads(topic["web_data"]) if topic["web_data"] else []
            topics.append(topic)

        return topics


# ==================== User Functions ====================

def create_user(
    email: str,
    password_hash: str = None,
    oauth_provider: str = None,
    oauth_id: str = None
) -> int:
    """
    Create a new user account.

    Args:
        email: User's email address (unique)
        password_hash: Hashed password (None for OAuth-only users)
        oauth_provider: OAuth provider name ('google', 'github', etc.)
        oauth_id: User's ID from the OAuth provider

    Returns:
        The ID of the created user.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (email, password_hash, oauth_provider, oauth_id)
            VALUES (?, ?, ?, ?)
        """, (email, password_hash, oauth_provider, oauth_id))
        conn.commit()
        return cursor.lastrowid


def get_user_by_email(email: str) -> Optional[dict]:
    """Get a user by email address."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Get a user by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_oauth(provider: str, oauth_id: str) -> Optional[dict]:
    """Get a user by OAuth provider and ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE oauth_provider = ? AND oauth_id = ?",
            (provider, oauth_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_user_oauth(user_id: int, oauth_provider: str, oauth_id: str) -> bool:
    """Link an OAuth account to an existing user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users
            SET oauth_provider = ?, oauth_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (oauth_provider, oauth_id, user_id))
        conn.commit()
        return cursor.rowcount > 0


# ==================== Conversation History Functions ====================

def save_conversation(
    user_id: int,
    conversation_id: str,
    title: str,
    messages: List[dict]
) -> int:
    """
    Save a new conversation history.

    Args:
        user_id: The user's ID
        conversation_id: Unique conversation identifier (UUID)
        title: Conversation title (usually first user message)
        messages: List of message dicts with role and content

    Returns:
        The ID of the saved conversation.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_histories (user_id, conversation_id, title, messages)
            VALUES (?, ?, ?, ?)
        """, (user_id, conversation_id, title, json.dumps(messages)))
        conn.commit()
        return cursor.lastrowid


def update_conversation(
    conversation_id: str,
    title: str = None,
    messages: List[dict] = None
) -> bool:
    """
    Update an existing conversation.

    Args:
        conversation_id: The conversation's unique ID
        title: New title (optional)
        messages: Updated messages list (optional)

    Returns:
        True if updated, False if not found.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)

        if messages is not None:
            updates.append("messages = ?")
            params.append(json.dumps(messages))

        params.append(conversation_id)

        cursor.execute(f"""
            UPDATE conversation_histories
            SET {', '.join(updates)}
            WHERE conversation_id = ?
        """, params)
        conn.commit()
        return cursor.rowcount > 0


def get_conversation(conversation_id: str) -> Optional[dict]:
    """Get a conversation by its unique ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM conversation_histories WHERE conversation_id = ?",
            (conversation_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        conv = dict(row)
        conv["messages"] = json.loads(conv["messages"]) if conv["messages"] else []
        return conv


def get_user_conversations(user_id: int, limit: int = 50, offset: int = 0) -> List[dict]:
    """
    Get all conversations for a user, ordered by most recent.

    Args:
        user_id: The user's ID
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip

    Returns:
        List of conversation dicts (without full messages for efficiency).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, user_id, conversation_id, title, created_at, updated_at
            FROM conversation_histories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))

        return [dict(row) for row in cursor.fetchall()]


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation by its unique ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM conversation_histories WHERE conversation_id = ?",
            (conversation_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


# ==================== Usage Tracking Functions ====================

def log_api_usage(
    user_id: Optional[int],
    request_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int
) -> int:
    """
    Log API usage with automatic cost calculation.

    Args:
        user_id: User ID (None for anonymous)
        request_type: Type of request ('chat', 'article_generation', 'batch_generation')
        model: Model name ('gpt-5-mini', 'gpt-5.1', etc.)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        The ID of the logged usage record.
    """
    # Calculate estimated cost
    costs = OPENAI_COSTS.get(model, {'input': 0, 'output': 0})
    estimated_cost = (input_tokens / 1000 * costs['input']) + \
                     (output_tokens / 1000 * costs['output'])

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_usage (user_id, request_type, model, input_tokens,
                                   output_tokens, estimated_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, request_type, model, input_tokens, output_tokens, estimated_cost))
        conn.commit()
        return cursor.lastrowid


def log_user_event(
    user_id: Optional[int],
    session_id: Optional[str],
    event_type: str,
    event_data: Dict[str, Any]
) -> int:
    """
    Log a user behavior event.

    Args:
        user_id: User ID (None for anonymous)
        session_id: Browser session ID for tracking anonymous users
        event_type: Type of event ('query', 'article_view', 'video_click', 'conversation_start')
        event_data: Additional event data as a dictionary

    Returns:
        The ID of the logged event.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_events (user_id, session_id, event_type, event_data)
            VALUES (?, ?, ?, ?)
        """, (user_id, session_id, event_type, json.dumps(event_data)))
        conn.commit()
        return cursor.lastrowid


def log_chat_query(
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[int] = None,
    query_type: Optional[str] = None,
    user_email: Optional[str] = None
) -> int:
    """
    Log a chat query. Call this at the start of a chat request.

    Returns:
        The ID of the logged query (use to update later).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_queries (session_id, user_id, query, query_type, status, user_email)
            VALUES (?, ?, ?, ?, 'pending', ?)
        """, (session_id, user_id, query, query_type, user_email))
        conn.commit()
        return cursor.lastrowid


def update_chat_query_success(
    query_id: int,
    response_length: int = 0,
    videos_used: int = 0,
    duration_seconds: float = 0,
    cost: float = 0
) -> None:
    """Update a chat query as successful."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_queries
            SET status = 'success',
                response_length = ?,
                videos_used = ?,
                duration_seconds = ?,
                cost = ?
            WHERE id = ?
        """, (response_length, videos_used, duration_seconds, cost, query_id))
        conn.commit()


def update_chat_query_error(
    query_id: int,
    error_message: str,
    error_traceback: Optional[str] = None,
    duration_seconds: float = 0
) -> None:
    """Update a chat query as failed."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_queries
            SET status = 'error',
                error_message = ?,
                error_traceback = ?,
                duration_seconds = ?
            WHERE id = ?
        """, (error_message, error_traceback, duration_seconds, query_id))
        conn.commit()


def get_recent_chat_queries(hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent chat queries for admin dashboard."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_id, user_id, user_email, query, query_type, status,
                   error_message, error_traceback, response_length, videos_used,
                   duration_seconds, cost, created_at
            FROM chat_queries
            WHERE created_at > datetime('now', ?)
            ORDER BY created_at DESC
            LIMIT ?
        """, (f'-{hours} hours', limit))
        return [dict(row) for row in cursor.fetchall()]


def get_chat_query_stats(hours: int = 24) -> Dict[str, Any]:
    """Get chat query statistics for admin dashboard."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total_queries,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count,
                AVG(duration_seconds) as avg_duration,
                AVG(CASE WHEN status = 'success' THEN response_length END) as avg_response_length
            FROM chat_queries
            WHERE created_at > datetime('now', ?)
        """, (f'-{hours} hours',))
        row = cursor.fetchone()
        return dict(row) if row else {}


def get_user_usage_summary(user_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Get aggregated API usage for a specific user.

    Args:
        user_id: The user's ID
        days: Number of days to look back

    Returns:
        Dict with total_input_tokens, total_output_tokens, total_cost, total_requests
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) as total_cost,
                COUNT(*) as total_requests
            FROM api_usage
            WHERE user_id = ? AND created_at > datetime('now', ?)
        """, (user_id, f'-{days} days'))
        row = cursor.fetchone()
        return dict(row) if row else {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0,
            'total_requests': 0
        }


def get_usage_report(days: int = 30) -> Dict[str, Any]:
    """
    Get overall usage report with costs by model and user.

    Args:
        days: Number of days to look back

    Returns:
        Dict with 'by_model', 'by_user', 'by_request_type', and 'events' breakdowns
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Costs by model
        cursor.execute("""
            SELECT model,
                   COALESCE(SUM(estimated_cost_usd), 0) as cost,
                   COALESCE(SUM(input_tokens), 0) as input_tokens,
                   COALESCE(SUM(output_tokens), 0) as output_tokens,
                   COUNT(*) as requests
            FROM api_usage
            WHERE created_at > datetime('now', ?)
            GROUP BY model
            ORDER BY cost DESC
        """, (f'-{days} days',))
        by_model = [dict(row) for row in cursor.fetchall()]

        # Costs by user (top 20)
        cursor.execute("""
            SELECT user_id,
                   COALESCE(SUM(estimated_cost_usd), 0) as cost,
                   COUNT(*) as requests
            FROM api_usage
            WHERE created_at > datetime('now', ?)
            GROUP BY user_id
            ORDER BY cost DESC
            LIMIT 20
        """, (f'-{days} days',))
        by_user = [dict(row) for row in cursor.fetchall()]

        # Costs by request type
        cursor.execute("""
            SELECT request_type,
                   COALESCE(SUM(estimated_cost_usd), 0) as cost,
                   COUNT(*) as requests
            FROM api_usage
            WHERE created_at > datetime('now', ?)
            GROUP BY request_type
            ORDER BY cost DESC
        """, (f'-{days} days',))
        by_request_type = [dict(row) for row in cursor.fetchall()]

        # Event counts
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM user_events
            WHERE created_at > datetime('now', ?)
            GROUP BY event_type
            ORDER BY count DESC
        """, (f'-{days} days',))
        events = [dict(row) for row in cursor.fetchall()]

        # Total summary
        cursor.execute("""
            SELECT
                COALESCE(SUM(estimated_cost_usd), 0) as total_cost,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COUNT(*) as total_requests
            FROM api_usage
            WHERE created_at > datetime('now', ?)
        """, (f'-{days} days',))
        totals = dict(cursor.fetchone())

        return {
            'days': days,
            'totals': totals,
            'by_model': by_model,
            'by_user': by_user,
            'by_request_type': by_request_type,
            'events': events
        }


def get_user_events(
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    days: int = 7,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get user events with optional filtering.

    Args:
        user_id: Filter by user ID (optional)
        session_id: Filter by session ID (optional)
        event_type: Filter by event type (optional)
        days: Number of days to look back
        limit: Maximum number of events to return

    Returns:
        List of event dicts
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = """
            SELECT id, user_id, session_id, event_type, event_data, created_at
            FROM user_events
            WHERE created_at > datetime('now', ?)
        """
        params = [f'-{days} days']

        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)

        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        events = []
        for row in cursor.fetchall():
            event = dict(row)
            event['event_data'] = json.loads(event['event_data']) if event['event_data'] else {}
            events.append(event)

        return events


# ==================== Transcript Cache Functions ====================

def get_cached_transcript(video_id: str) -> Optional[str]:
    """
    Get a transcript from the persistent cache.

    Args:
        video_id: YouTube video ID

    Returns:
        Transcript text if found, None otherwise.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT transcript FROM transcript_cache WHERE video_id = ?",
            (video_id,)
        )
        row = cursor.fetchone()
        return row['transcript'] if row else None


def save_transcript_to_cache(video_id: str, transcript: str) -> None:
    """
    Save a transcript to the persistent cache.

    Args:
        video_id: YouTube video ID
        transcript: The transcript text to cache
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO transcript_cache (video_id, transcript)
            VALUES (?, ?)
        """, (video_id, transcript))
        conn.commit()


def get_transcript_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the transcript cache.

    Returns:
        Dict with total_cached, oldest_entry, newest_entry
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM transcript_cache")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM transcript_cache")
        row = cursor.fetchone()

        return {
            'total_cached': total,
            'oldest_entry': row[0],
            'newest_entry': row[1]
        }


def clear_old_transcripts(days: int = 90) -> int:
    """
    Clear transcripts older than N days to manage cache size.

    Args:
        days: Number of days after which to clear transcripts

    Returns:
        Number of transcripts deleted.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM transcript_cache
            WHERE created_at < datetime('now', ?)
        """, (f'-{days} days',))
        conn.commit()
        return cursor.rowcount


# Initialize database on module import
init_database()
