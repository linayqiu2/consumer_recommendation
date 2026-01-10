'use client'

import { useState, useEffect } from 'react'

interface Article {
  id: number
  title: string
  topic: string
  article_type: string  // 'trending' | 'evergreen'
  created_at: string
  videos_analyzed: number
  thumbnail: string | null
  article_preview: string
}

interface TrendsSidebarProps {
  onViewArticle: (articleId: number) => void
  apiUrl: string
}

// Available color palettes for dynamic category coloring
const COLOR_PALETTES = [
  { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-300' },
  { bg: 'bg-pink-100', text: 'text-pink-700', border: 'border-pink-300' },
  { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-300' },
  { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-300' },
  { bg: 'bg-emerald-100', text: 'text-emerald-700', border: 'border-emerald-300' },
  { bg: 'bg-cyan-100', text: 'text-cyan-700', border: 'border-cyan-300' },
  { bg: 'bg-rose-100', text: 'text-rose-700', border: 'border-rose-300' },
  { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-300' },
  { bg: 'bg-teal-100', text: 'text-teal-700', border: 'border-teal-300' },
  { bg: 'bg-indigo-100', text: 'text-indigo-700', border: 'border-indigo-300' },
  { bg: 'bg-lime-100', text: 'text-lime-700', border: 'border-lime-300' },
  { bg: 'bg-fuchsia-100', text: 'text-fuchsia-700', border: 'border-fuchsia-300' },
]

// Simple hash function to get consistent color index for a topic
const getColorIndex = (topic: string): number => {
  let hash = 0
  for (let i = 0; i < topic.length; i++) {
    hash = ((hash << 5) - hash) + topic.charCodeAt(i)
    hash = hash & hash // Convert to 32-bit integer
  }
  return Math.abs(hash) % COLOR_PALETTES.length
}

const getCategoryStyle = (topic: string) => {
  return COLOR_PALETTES[getColorIndex(topic)]
}

// Helper: Curate articles per requirements
// - All evergreen articles
// - Latest trending article per topic
const curateArticles = (articles: Article[]): Article[] => {
  // 1. Get all evergreen articles
  const evergreen = articles.filter(a => a.article_type === 'evergreen')

  // 2. Get latest trending article per topic
  const trendingByTopic = new Map<string, Article>()
  const trending = articles
    .filter(a => a.article_type !== 'evergreen')
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())

  for (const article of trending) {
    if (!trendingByTopic.has(article.topic)) {
      trendingByTopic.set(article.topic, article)
    }
  }

  // 3. Combine: all evergreen + one trending per topic
  return [...evergreen, ...Array.from(trendingByTopic.values())]
}

// Helper: Fisher-Yates shuffle
const shuffleArray = <T,>(array: T[]): T[] => {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

export default function TrendsSidebar({ onViewArticle, apiUrl }: TrendsSidebarProps) {
  const [articles, setArticles] = useState<Article[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedTopics, setSelectedTopics] = useState<Set<string>>(new Set())
  const [selectedType, setSelectedType] = useState<'both' | 'trending' | 'evergreen'>('both')

  useEffect(() => {
    fetchRecentArticles()
  }, [])

  // Topics to completely exclude from display (not valid topics or deprecated)
  const excludedTopics = new Set(['evergreen', 'Home', 'Hotels', 'Restaurants'])

  const fetchRecentArticles = async () => {
    try {
      // Fetch more articles to have enough data for curation
      const response = await fetch(`${apiUrl}/api/articles/recent?days=30&limit=100`)
      if (response.ok) {
        const data = await response.json()
        const allArticles = data.articles || []

        // Filter out articles with excluded topics
        const validArticles = allArticles.filter((a: Article) => !excludedTopics.has(a.topic))

        // Curate: latest trending per topic + all evergreen
        const curated = curateArticles(validArticles)

        // Randomize order
        const shuffled = shuffleArray(curated)

        setArticles(shuffled)
      }
    } catch (error) {
      console.error('Failed to fetch articles:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const formatDateTime = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  }

  // Priority order for topic filters (lower = higher priority, appears first)
  const topicPriority: Record<string, number> = {
    'Electronics': 1,
    'EV': 2,
    'Cameras': 3,
    'Audio': 4,
    'Gaming': 5,
    'Beauty': 6,
    'Fashion': 7,
    'Fitness': 8,
    'Travel': 9,
    'Automotive': 10,
  }

  // Get unique categories from articles for filter, sorted by priority
  // (excluded topics are already filtered out at fetch time via excludedTopics)
  const uniqueCategories = Array.from(new Set(articles.map(a => a.topic)))
    .sort((a, b) => (topicPriority[a] || 99) - (topicPriority[b] || 99))

  // Toggle topic selection
  const toggleTopic = (topic: string) => {
    setSelectedTopics(prev => {
      const newSet = new Set(prev)
      if (newSet.has(topic)) {
        newSet.delete(topic)
      } else {
        newSet.add(topic)
      }
      return newSet
    })
  }

  // Filter articles based on selected type and topics
  const typeFilteredArticles = selectedType === 'both'
    ? articles
    : articles.filter(a => a.article_type === selectedType)

  const filteredArticles = selectedTopics.size === 0
    ? typeFilteredArticles
    : typeFilteredArticles.filter(a => selectedTopics.has(a.topic))

  return (
    <div className="w-full h-full flex flex-col bg-gradient-to-b from-slate-50 to-slate-100/50 py-10 px-5 border-l-2 border-sky-400/50">
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse"></span>
        <h2 className="text-sm font-bold text-slate-800 tracking-tight">Cross-Reviewer Trends</h2>
      </div>
      <p className="text-slate-500 text-xs mb-5">What multiple reviewers are saying right now</p>

      {/* Topic Filter */}
      {uniqueCategories.length > 0 && (
        <div className="mb-3">
          <div className="flex items-center gap-2 mb-2 text-xs text-slate-500">
            <span>Topic:</span>
            {selectedTopics.size > 0 && (
              <button
                onClick={() => setSelectedTopics(new Set())}
                className="text-sky-600 hover:text-sky-700 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {uniqueCategories.map((cat, i) => {
              const style = getCategoryStyle(cat)
              const isSelected = selectedTopics.has(cat)
              const isAllSelected = selectedTopics.size === 0
              return (
                <button
                  key={i}
                  onClick={() => toggleTopic(cat)}
                  className={`px-2 py-0.5 rounded text-xs border transition-all cursor-pointer
                    ${isSelected || isAllSelected
                      ? `${style.bg} ${style.text} ${style.border}`
                      : 'bg-slate-100 text-slate-500 border-slate-300 hover:bg-slate-200'
                    }`}
                >
                  {cat}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Type Filter */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2 text-xs text-slate-500">
          <span>Type:</span>
        </div>
        <div className="flex gap-1.5">
          {(['both', 'trending', 'evergreen'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setSelectedType(type)}
              className={`px-2 py-0.5 rounded text-xs border transition-all cursor-pointer
                ${selectedType === type
                  ? 'bg-sky-100 text-sky-700 border-sky-300'
                  : 'bg-slate-100 text-slate-500 border-slate-300 hover:bg-slate-200'
                }`}
            >
              {type === 'both' ? 'Both' : type === 'trending' ? 'Trending' : 'Evergreen'}
            </button>
          ))}
        </div>
      </div>

      {/* Articles List */}
      <div className="flex-1 overflow-y-auto pr-1">
        {isLoading ? (
          <div className="flex flex-col gap-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="bg-white rounded-lg p-3 animate-pulse border border-slate-200">
                <div className="flex gap-3">
                  <div className="w-20 h-14 bg-slate-200 rounded-lg"></div>
                  <div className="flex-1">
                    <div className="h-3 bg-slate-200 rounded w-3/4 mb-2"></div>
                    <div className="h-2.5 bg-slate-100 rounded w-1/2"></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : filteredArticles.length > 0 ? (
          <div className="flex flex-col gap-4">
            {filteredArticles.map((article) => {
              const catStyle = getCategoryStyle(article.topic)
              return (
                <div
                  key={article.id}
                  onClick={() => onViewArticle(article.id)}
                  className="bg-white rounded-xl p-3 hover:bg-slate-50 transition-all cursor-pointer group border border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md"
                >
                  <div className="flex gap-3">
                    {/* Thumbnail */}
                    <div className="w-20 h-14 rounded-lg overflow-hidden bg-slate-100 shrink-0">
                      {article.thumbnail ? (
                        <img
                          src={article.thumbnail}
                          alt={article.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </div>
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <h3 className="text-slate-800 font-medium text-sm line-clamp-2 mb-1 group-hover:text-sky-600 transition-colors leading-snug">
                        {article.title}
                      </h3>
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <span className="flex items-center gap-1">
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                          {article.videos_analyzed}
                        </span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          {formatDateTime(article.created_at)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Category and Type tags */}
                  <div className="mt-2 flex gap-1.5 flex-wrap">
                    {/* Topic tag */}
                    <span className={`inline-block px-2 py-0.5 rounded text-xs ${catStyle.bg} ${catStyle.text} border ${catStyle.border}`}>
                      {article.topic}
                    </span>
                    {/* Type tag */}
                    <span className={`inline-block px-2 py-0.5 rounded text-xs ${
                      article.article_type === 'evergreen'
                        ? 'bg-emerald-100 text-emerald-700 border border-emerald-300'
                        : 'bg-orange-100 text-orange-700 border border-orange-300'
                    }`}>
                      {article.article_type === 'evergreen' ? 'Evergreen' : 'Trending'}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500">
            <div className="w-14 h-14 rounded-full bg-slate-200 flex items-center justify-center mb-3">
              <svg className="w-7 h-7 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
              </svg>
            </div>
            <h3 className="text-slate-900 font-medium mb-1 text-sm">No trends yet</h3>
            <p className="text-xs">Cross-reviewer analysis will appear here</p>
          </div>
        )}
      </div>

      {/* Footer */}
      {filteredArticles.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-200 text-xs text-slate-500">
          <span>
            {filteredArticles.length} trend{filteredArticles.length !== 1 ? 's' : ''}
            {selectedTopics.size > 0 && ` (filtered from ${articles.length})`}
          </span>
        </div>
      )}
    </div>
  )
}
