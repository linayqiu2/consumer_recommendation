# Consumer Product Recommendation

AI-powered product recommendation chat interface that aggregates insights from YouTube reviews.

## Features

- ChatGPT-style interface
- Searches YouTube for relevant product reviews
- Extracts and analyzes video transcripts
- Provides comprehensive recommendations using GPT-4

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:3000

## Environment Variables

Create `backend/.env`:
```
OPENAI_API_KEY=your_openai_key
YOUTUBE_API_KEY=your_youtube_api_key
```

## Architecture

1. User enters product query
2. Backend generates optimized YouTube search queries
3. Searches YouTube API for relevant videos
4. Fetches transcripts using youtube-transcript-api
5. GPT-4 analyzes transcripts and generates comprehensive answer
6. Returns answer with linked video sources
