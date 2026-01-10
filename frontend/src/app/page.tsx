'use client'

import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import LandingPage from './components/LandingPage'
import TrendsSidebar from './components/TrendsSidebar'
import { AuthProvider, useAuth } from './contexts/AuthContext'

interface VideoInfo {
  video_id: string
  title: string
  url: string
  thumbnail: string
  channel: string
  has_transcript?: boolean
  view_count?: number
  like_count?: number
}

interface QueryResult {
  query: string
  videos: VideoInfo[]
}

interface WebResult {
  title: string
  url: string
  content: string
  score: number
}

interface WebQueryResult {
  query: string
  results: WebResult[]
}

interface AnswerGenerationDebug {
  method_used: string  // "structured" | "fallback" | "none"
  fallback_reason: string | null
  video_insights: Record<string, unknown>[] | null
  synthesis: Record<string, unknown> | null
}

interface RankedVideoInfo {
  video_id: string
  title: string
  url: string
  thumbnail: string
  channel: string
  description: string
  source_query: string
  has_transcript: boolean
}

interface TimingInfo {
  total_seconds: number
  classify_query_seconds: number
  generate_queries_seconds: number
  youtube_search_seconds: number
  video_ranking_seconds: number
  web_search_seconds: number
  transcript_fetch_seconds: number
  insights_extraction_seconds: number
  synthesis_seconds: number
  answer_generation_seconds: number
}

interface DebugInfo {
  generated_queries: string[]
  query_results: QueryResult[]
  total_videos_found: number
  videos_with_transcripts: number
  videos_analyzed: number
  ranked_videos: RankedVideoInfo[]
  web_queries: string[]
  web_query_results: WebQueryResult[]
  total_web_results: number
  answer_generation?: AnswerGenerationDebug
  timing?: TimingInfo
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  videos?: VideoInfo[]
  sources?: string
  debug?: DebugInfo
}

interface ArticleData {
  id: number
  topic: string
  article_title: string
  article_content: string
  created_at: string
  videos: {
    video_id: string
    title: string
    url: string
    thumbnail: string
    channel: string
    has_transcript: boolean
    view_count?: number
    like_count?: number
  }[]
  web_sources: {
    title: string
    url: string
    content: string
    score: number
  }[]
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Generate/get session ID for anonymous tracking
const getSessionId = (): string => {
  if (typeof window === 'undefined') return ''
  let sessionId = sessionStorage.getItem('tracking_session_id')
  if (!sessionId) {
    sessionId = crypto.randomUUID()
    sessionStorage.setItem('tracking_session_id', sessionId)
  }
  return sessionId
}

// Track user events (fire and forget)
const trackEvent = (eventType: string, eventData: Record<string, unknown>) => {
  fetch(`${API_URL}/api/events/track`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: eventType,
      event_data: eventData,
      session_id: getSessionId()
    })
  }).catch(err => console.debug('Event tracking failed:', err))
}

interface ConversationHistory {
  id: number
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
}

// Sidebar component that is login-aware
function ConversationSidebar({
  onNewChat,
  conversations,
  currentConversationId,
  onSelectConversation,
  onDeleteConversation
}: {
  onNewChat: () => void
  conversations: ConversationHistory[]
  currentConversationId: string | null
  onSelectConversation: (conv: ConversationHistory) => void
  onDeleteConversation: (conversationId: string) => void
}) {
  const { user } = useAuth()

  // Ensure conversations is always an array (defensive check)
  const safeConversations = Array.isArray(conversations) ? conversations : []

  // If not logged in, show minimal toggle button
  if (!user) {
    return (
      <div className="hidden md:flex md:w-12 md:flex-col bg-gray-50 border-r border-gray-200">
        <div className="flex flex-col items-center py-3">
          <button
            onClick={onNewChat}
            className="w-10 h-10 flex items-center justify-center rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-100 transition-colors"
            title="New chat"
          >
            <svg
              stroke="currentColor"
              fill="none"
              strokeWidth="2"
              viewBox="0 0 24 24"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-5 w-5"
              xmlns="http://www.w3.org/2000/svg"
            >
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
          </button>
        </div>
      </div>
    )
  }

  // If logged in, show full sidebar with conversation history
  return (
    <div className="hidden md:flex md:w-[260px] md:flex-col bg-gray-50">
      <div className="flex h-full flex-col">
        <div className="flex items-center gap-3 p-3">
          <button
            onClick={onNewChat}
            className="flex w-full items-center gap-3 rounded-md border border-gray-300 p-3 text-sm text-gray-700 transition-colors hover:bg-gray-100"
          >
            <svg
              stroke="currentColor"
              fill="none"
              strokeWidth="2"
              viewBox="0 0 24 24"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
              xmlns="http://www.w3.org/2000/svg"
            >
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {/* Conversation history list */}
          {safeConversations.length === 0 ? (
            <p className="text-gray-500 text-sm text-center py-4">No conversations yet</p>
          ) : (
            <div className="space-y-1">
              {safeConversations.map((conv) => (
                <div
                  key={conv.conversation_id}
                  className={`group flex items-center gap-2 rounded-lg p-2 cursor-pointer transition-colors ${
                    currentConversationId === conv.conversation_id
                      ? 'bg-gray-200'
                      : 'hover:bg-gray-100'
                  }`}
                  onClick={() => onSelectConversation(conv)}
                >
                  <svg className="h-4 w-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <span className="flex-1 text-sm text-gray-700 truncate">{conv.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteConversation(conv.conversation_id)
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-200 rounded transition-all"
                    title="Delete conversation"
                  >
                    <svg className="h-4 w-4 text-gray-400 hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="border-t border-gray-300 p-4">
          <div className="flex items-center gap-3 text-sm text-gray-500">
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            Review Video Insights v1.0
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Home() {
  const [showLanding, setShowLanding] = useState(true)
  const [selectedArticle, setSelectedArticle] = useState<ArticleData | null>(null)
  const [isLoadingArticle, setIsLoadingArticle] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showDebug, setShowDebug] = useState(false)
  const [disableSearch, setDisableSearch] = useState(false)
  const [quickDebugMode, setQuickDebugMode] = useState(false)
  const [progressMessage, setProgressMessage] = useState<string>('')

  // Multi-step progress tracking (Claude Code style)
  interface ProgressStep {
    id: string
    message: string
    detail?: string
    status: 'pending' | 'in_progress' | 'completed'
  }
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([])
  const [pendingQuery, setPendingQuery] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const isLoadingFromHistoryRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  // YouTube sidebar state
  const [showVideoSidebar, setShowVideoSidebar] = useState(false)
  // Trends sidebar state (visible on landing, hidden in conversation by default)
  const [showTrendsSidebar, setShowTrendsSidebar] = useState(true)
  const [currentVideo, setCurrentVideo] = useState<{
    videoId: string;
    startTime: number;
    title?: string;
    channel?: string;
    thumbnail?: string;
    view_count?: number;
    like_count?: number;
    insights?: any; // Video insights from backend analysis
  } | null>(null)

  // Conversation history state
  const [conversations, setConversations] = useState<ConversationHistory[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const conversationIdRef = useRef<string | null>(null) // Sync ref to prevent duplicate saves

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    // When loading from history, scroll to top; otherwise scroll to bottom for new messages
    if (isLoadingFromHistoryRef.current) {
      isLoadingFromHistoryRef.current = false
      // Scroll to top of messages container
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = 0
      }
    } else {
      scrollToBottom()
    }
  }, [messages])

  // Handle starting a conversation from the landing page
  const handleStartConversation = (query?: string) => {
    setShowLanding(false)
    // Reset conversation and video state for fresh start
    setCurrentConversationId(null)
    conversationIdRef.current = null // Also reset the ref
    setCurrentVideo(null)
    setShowVideoSidebar(false)
    // Keep trends sidebar visible to maintain same-page feel
    if (query) {
      setPendingQuery(query)
    }
  }

  // Submit pending query when transitioning from landing page
  useEffect(() => {
    if (!showLanding && pendingQuery) {
      setInput(pendingQuery)
      setPendingQuery(null)
      // Auto-submit after a short delay to allow state to update
      setTimeout(() => {
        const form = document.querySelector('form')
        if (form) {
          form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
        }
      }, 100)
    }
  }, [showLanding, pendingQuery])

  // Fetch conversations on mount and when auth changes
  useEffect(() => {
    const handleAuthChange = () => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        fetchConversations()
      }
    }

    // Initial fetch if already logged in
    handleAuthChange()

    // Listen for auth changes (login/register)
    window.addEventListener('auth-change', handleAuthChange)
    return () => window.removeEventListener('auth-change', handleAuthChange)
  }, [])

  // Handle going back to landing page
  const handleBackToLanding = () => {
    setShowLanding(true)
    setShowTrendsSidebar(true) // Show trends when returning to landing
    setSelectedArticle(null)
    setMessages([])
    setCurrentConversationId(null)
    conversationIdRef.current = null // Also reset the ref
    // Reset video state
    setCurrentVideo(null)
    setShowVideoSidebar(false)
  }

  // Handle selecting a conversation from history
  const handleSelectConversation = async (conv: ConversationHistory) => {
    try {
      const token = localStorage.getItem('auth_token')
      if (!token) return

      const response = await fetch(`${API_URL}/api/conversations/${conv.conversation_id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        // Set flag to scroll to top instead of bottom when messages update
        isLoadingFromHistoryRef.current = true
        setMessages(data.messages || [])
        setCurrentConversationId(conv.conversation_id)
        conversationIdRef.current = conv.conversation_id // Sync the ref
        setShowLanding(false)
      }
    } catch (err) {
      console.error('Failed to load conversation:', err)
    }
  }

  // Handle deleting a conversation
  const handleDeleteConversation = async (conversationId: string) => {
    try {
      const token = localStorage.getItem('auth_token')
      if (!token) return

      const response = await fetch(`${API_URL}/api/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        setConversations(prev => prev.filter(c => c.conversation_id !== conversationId))
        if (currentConversationId === conversationId) {
          setCurrentConversationId(null)
          conversationIdRef.current = null // Also reset the ref
          setMessages([])
        }
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }

  // Fetch conversations when component mounts (for logged-in users)
  const fetchConversations = async () => {
    try {
      const token = localStorage.getItem('auth_token')
      if (!token) return

      const response = await fetch(`${API_URL}/api/conversations`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        // API returns { conversations: [...] }, extract the array
        const convList = data.conversations || data
        setConversations(Array.isArray(convList) ? convList : [])
      }
    } catch (err) {
      console.error('Failed to fetch conversations:', err)
      setConversations([])
    }
  }

  // Save or update a conversation
  const saveConversation = async (messagesData: Message[]) => {
    try {
      const token = localStorage.getItem('auth_token')
      if (!token || messagesData.length === 0) return

      // Use ref for synchronous ID check to prevent race conditions
      // Generate a new conversation ID if we don't have one
      let convId = conversationIdRef.current
      if (!convId) {
        convId = crypto.randomUUID()
        conversationIdRef.current = convId // Set ref immediately (sync)
        setCurrentConversationId(convId)   // Also update state (async)
      }

      // Use the first user message as the title
      const firstUserMsg = messagesData.find(m => m.role === 'user')
      const title = firstUserMsg
        ? firstUserMsg.content.slice(0, 100) + (firstUserMsg.content.length > 100 ? '...' : '')
        : 'New conversation'

      const response = await fetch(`${API_URL}/api/conversations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          conversation_id: convId,
          title,
          messages: messagesData
        })
      })

      if (response.ok) {
        // Refresh conversations list
        fetchConversations()
      }
    } catch (err) {
      console.error('Failed to save conversation:', err)
    }
  }

  // Handle viewing an article from the landing page
  const handleViewArticle = async (articleId: number) => {
    setIsLoadingArticle(true)
    setShowLanding(false)
    // Reset video sidebar state when viewing a new article
    setShowVideoSidebar(false)
    setCurrentVideo(null)
    try {
      const response = await fetch(`${API_URL}/api/articles/${articleId}`)
      if (response.ok) {
        const data = await response.json()
        // API returns article directly, not wrapped in { article: ... }
        setSelectedArticle(data)

        // Track article view event
        trackEvent('article_view', {
          article_id: articleId,
          article_title: data.article_title,
          topic: data.topic
        })
      } else {
        console.error('Failed to fetch article')
        setShowLanding(true)
      }
    } catch (error) {
      console.error('Error fetching article:', error)
      setShowLanding(true)
    } finally {
      setIsLoadingArticle(false)
    }
  }

  // Handle closing article view and starting a conversation
  const handleStartConversationFromArticle = () => {
    setSelectedArticle(null)
    // Reset video sidebar state when starting a new conversation
    setShowVideoSidebar(false)
    setCurrentVideo(null)
  }

  // Function to extract video ID from YouTube URL
  const extractVideoId = (url: string): string | null => {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\s]+)/,
      /youtube\.com\/embed\/([^&\s]+)/
    ]
    for (const pattern of patterns) {
      const match = url.match(pattern)
      if (match) return match[1]
    }
    return null
  }

  const formatNumber = (num: number | null | undefined): string => {
    if (num == null) return '0'
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M'
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K'
    }
    return num.toString()
  }

  // Function to convert [MM:SS] to seconds
  const timestampToSeconds = (timestamp: string): number => {
    const match = timestamp.match(/\[(\d+):(\d+)\]/)
    if (match) {
      const minutes = parseInt(match[1])
      const seconds = parseInt(match[2])
      return minutes * 60 + seconds
    }
    return 0
  }

  // Function to handle timestamp click
  const handleTimestampClick = (e: React.MouseEvent<HTMLElement>, messageVideos?: VideoInfo[], debugInfo?: DebugInfo) => {
    const target = e.target as HTMLElement

    // Check if clicked element is a timestamp link
    if (target.tagName === 'A' && target.textContent?.match(/\[\d+:\d+\]/)) {
      e.preventDefault()
      const timestamp = target.textContent
      const seconds = timestampToSeconds(timestamp)

      // Find the message container (scope search to current message only)
      const messageContainer = target.closest('.prose')

      // Find the video URL - look for nearby links in the same paragraph or sentence
      const parent = target.closest('p, li, blockquote')
      if (parent) {
        const links = parent.querySelectorAll('a[href*="youtube.com"], a[href*="youtu.be"]')
        for (const link of Array.from(links)) {
          const videoId = extractVideoId((link as HTMLAnchorElement).href)
          if (videoId) {
            // Try to find video metadata from the message's video list
            const videoInfo = messageVideos?.find(v => v.video_id === videoId)

            // Try to find insights for this video from debug info
            const videoInsights = debugInfo?.answer_generation?.video_insights?.find(
              (insight: any) => insight.url?.includes(videoId)
            )

            setCurrentVideo({
              videoId,
              startTime: seconds,
              title: videoInfo?.title || link.textContent || undefined,
              channel: videoInfo?.channel,
              thumbnail: videoInfo?.thumbnail,
              view_count: videoInfo?.view_count,
              like_count: videoInfo?.like_count,
              insights: videoInsights
            })
            setShowVideoSidebar(true)

            // Track video click event
            trackEvent('video_click', {
              video_id: videoId,
              video_title: videoInfo?.title || link.textContent || undefined,
              channel: videoInfo?.channel,
              start_time: seconds,
              source: 'chat'
            })
            return
          }
        }
      }

      // Fallback: search for any YouTube link in the current message only (not entire DOM)
      if (messageContainer) {
        const messageLinks = messageContainer.querySelectorAll('a[href*="youtube.com"], a[href*="youtu.be"]')
        for (const link of Array.from(messageLinks)) {
          const videoId = extractVideoId((link as HTMLAnchorElement).href)
          if (videoId) {
            const videoInfo = messageVideos?.find(v => v.video_id === videoId)

            // Try to find insights for this video from debug info
            const videoInsights = debugInfo?.answer_generation?.video_insights?.find(
              (insight: any) => insight.url?.includes(videoId)
            )

            setCurrentVideo({
              videoId,
              startTime: seconds,
              title: videoInfo?.title,
              channel: videoInfo?.channel,
              thumbnail: videoInfo?.thumbnail,
              view_count: videoInfo?.view_count,
              like_count: videoInfo?.like_count,
              insights: videoInsights
            })
            setShowVideoSidebar(true)

            // Track video click event
            trackEvent('video_click', {
              video_id: videoId,
              video_title: videoInfo?.title,
              channel: videoInfo?.channel,
              start_time: seconds,
              source: 'chat'
            })
            return
          }
        }
      }
    }
  }

  // Handle stopping/aborting the streaming response
  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')

    // Capture current messages for conversation history BEFORE adding the new user message
    const conversationHistory = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }))

    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)
    setProgressMessage('Connecting...')
    setProgressSteps([]) // Clear previous progress steps

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          max_videos: quickDebugMode ? 1 : 12,
          disable_search: disableSearch,
          conversation_history: conversationHistory,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No reader available')
      }

      const decoder = new TextDecoder()
      let streamingContent = ''
      let videos: VideoInfo[] = []
      let sources = ''
      let debug: DebugInfo | undefined
      let hasAddedAssistantMessage = false // Track if we've added the assistant message

      let buffer = ''
      let eventType = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Process complete SSE events from buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7)
            console.log('SSE Event Type:', eventType)
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)
              console.log('SSE Data for', eventType, ':', parsed)

              if (eventType === 'progress') {
                setProgressMessage(parsed.message || parsed.step)

                // Update multi-step progress (Claude Code style)
                const { step, step_index, total_steps, message, detail } = parsed
                if (typeof step_index === 'number' && step_index >= 0) {
                  setProgressSteps(prev => {
                    const steps = [...prev]

                    // Mark all previous steps as completed
                    for (let i = 0; i < step_index; i++) {
                      if (steps[i]) {
                        steps[i] = { ...steps[i], status: 'completed' }
                      }
                    }

                    // Update or add current step
                    steps[step_index] = {
                      id: step,
                      message,
                      detail,
                      status: 'in_progress'
                    }

                    return steps
                  })
                }
              } else if (eventType === 'answer_chunk') {
                streamingContent += parsed.text

                // Add assistant message on first chunk (not during progress phase)
                if (!hasAddedAssistantMessage) {
                  hasAddedAssistantMessage = true
                  // Clear progress steps when streaming starts - the response is now visible
                  setProgressSteps([])
                  setMessages(prev => [
                    ...prev,
                    {
                      role: 'assistant',
                      content: streamingContent,
                    },
                  ])
                } else {
                  // Update the last message with streaming content
                  setMessages(prev => {
                    const newMessages = [...prev]
                    const lastIdx = newMessages.length - 1
                    if (lastIdx >= 0 && newMessages[lastIdx].role === 'assistant') {
                      newMessages[lastIdx] = {
                        ...newMessages[lastIdx],
                        content: streamingContent,
                      }
                    }
                    return newMessages
                  })
                }
              } else if (eventType === 'metadata') {
                videos = parsed.videos || []
                sources = parsed.sources_summary || ''
                debug = parsed.debug
                console.log('=== Metadata Event Received ===')
                console.log('Videos count:', videos.length)
                console.log('Has debug?:', !!debug)
                console.log('Sample video:', videos[0])
                // Update with final metadata
                setMessages(prev => {
                  const newMessages = [...prev]
                  const lastIdx = newMessages.length - 1
                  if (lastIdx >= 0 && newMessages[lastIdx].role === 'assistant') {
                    newMessages[lastIdx] = {
                      ...newMessages[lastIdx],
                      videos,
                      sources,
                      debug,
                    }
                  }
                  return newMessages
                })
              } else if (eventType === 'error') {
                throw new Error(parsed.message || 'Unknown error')
              }
            } catch (parseError) {
              // Ignore parse errors for incomplete data
              if (eventType === 'error') {
                console.error('Stream error:', data)
              }
            }
          }
        }
      }

      setProgressMessage('')

      // Save conversation after successful message exchange
      // Use a callback to get the current messages state
      setMessages(prev => {
        // Only save if we have messages and the last one is from assistant with content
        const lastMsg = prev[prev.length - 1]
        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content) {
          saveConversation(prev)
        }
        return prev
      })
    } catch (error) {
      console.error('Error:', error)
      setProgressMessage('')

      // Check if this was an abort (user stopped generation)
      const isAborted = error instanceof Error && error.name === 'AbortError'

      setMessages(prev => {
        const lastMsg = prev[prev.length - 1]

        // If there's an assistant message with content, append stopped message for abort
        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content) {
          if (isAborted) {
            return [
              ...prev.slice(0, -1),
              {
                ...lastMsg,
                content: lastMsg.content + '\n\n*Generation stopped by user.*',
              },
            ]
          }
          // Keep the partial content on error
          return prev
        }

        // No assistant message yet (still in progress phase)
        if (isAborted) {
          return [
            ...prev,
            {
              role: 'assistant',
              content: '*Generation stopped by user.*',
            },
          ]
        }

        // Error with no content yet
        return [
          ...prev,
          {
            role: 'assistant',
            content: 'Sorry, I encountered an error processing your request. Please try again.',
          },
        ]
      })
    } finally {
      setIsLoading(false)
      setProgressMessage('')
      setProgressSteps([])
      abortControllerRef.current = null
    }
  }

  const exampleQueries = [
    "What's the best robot vacuum under $500?",
    "Which noise-cancelling headphones should I buy?",
    "Best budget mirrorless camera for beginners",
    "Top rated air fryers 2024",
  ]

  // Get the latest debug info from messages
  const latestDebug = messages.filter(m => m.debug).slice(-1)[0]?.debug

  // Show unified layout with landing OR conversation + trends sidebar
  if (showLanding) {
    return (
      <AuthProvider>
        <div className="flex h-screen bg-gray-50 text-gray-900">
          {/* Left sidebar - Conversation history (only shown when logged in) */}
          <ConversationSidebar
            onNewChat={handleBackToLanding}
            conversations={conversations}
            currentConversationId={currentConversationId}
            onSelectConversation={handleSelectConversation}
            onDeleteConversation={handleDeleteConversation}
          />

          {/* Main section - Landing hero content */}
          <LandingPage onStartConversation={handleStartConversation} />

          {/* Vertical Divider */}
          <div className="hidden md:block w-px bg-gradient-to-b from-transparent via-gray-300 to-transparent"></div>

          {/* Right sidebar - Trends */}
          <div className="hidden md:block w-[380px] h-screen">
            <TrendsSidebar onViewArticle={handleViewArticle} apiUrl={API_URL} />
          </div>
        </div>
      </AuthProvider>
    )
  }

  // Handle timestamp click in article view
  const handleArticleTimestampClick = (e: React.MouseEvent<HTMLElement>) => {
    if (!selectedArticle) return

    const target = e.target as HTMLElement
    const textContent = target.textContent

    // Check if clicked element is a timestamp link
    if (target.tagName === 'A' && textContent && textContent.match(/\[\d+:\d+\]/)) {
      e.preventDefault()
      const seconds = timestampToSeconds(textContent)

      // Helper function to find video and show sidebar
      const showVideoWithId = (videoId: string, title?: string | null) => {
        const videoInfo = selectedArticle.videos?.find(v => v.video_id === videoId)
        setCurrentVideo({
          videoId,
          startTime: seconds,
          title: videoInfo?.title || title || undefined,
          channel: videoInfo?.channel,
          thumbnail: videoInfo?.thumbnail,
          view_count: videoInfo?.view_count,
          like_count: videoInfo?.like_count,
        })
        setShowVideoSidebar(true)

        // Track video click event
        trackEvent('video_click', {
          video_id: videoId,
          video_title: videoInfo?.title || title || undefined,
          channel: videoInfo?.channel,
          start_time: seconds,
          source: 'article'
        })
      }

      // Find the video URL - look for nearby links in the same paragraph or sentence
      const parent = target.closest('p, li, blockquote')
      if (parent) {
        const links = parent.querySelectorAll('a[href*="youtube.com"], a[href*="youtu.be"]')
        for (const link of Array.from(links)) {
          const videoId = extractVideoId((link as HTMLAnchorElement).href)
          if (videoId) {
            showVideoWithId(videoId, link.textContent)
            return
          }
        }
      }

      // Improved fallback: find the closest YouTube link that appears BEFORE this timestamp
      const articleContent = target.closest('.prose')
      if (articleContent) {
        const allElements = Array.from(articleContent.querySelectorAll('*'))
        const timestampIndex = allElements.indexOf(target)

        // Search backwards from the timestamp to find the most recent YouTube link
        for (let i = timestampIndex - 1; i >= 0; i--) {
          const el = allElements[i]
          if (el.tagName === 'A') {
            const href = (el as HTMLAnchorElement).href
            if (href && (href.includes('youtube.com') || href.includes('youtu.be'))) {
              const videoId = extractVideoId(href)
              if (videoId) {
                showVideoWithId(videoId, el.textContent)
                return
              }
            }
          }
        }
      }

      // Last fallback: use first video from article's video list
      if (selectedArticle.videos && selectedArticle.videos.length > 0) {
        const firstVideo = selectedArticle.videos[0]
        showVideoWithId(firstVideo.video_id, firstVideo.title)
        return
      }
    }
  }

  // Show article view if an article is selected
  if (selectedArticle || isLoadingArticle) {
    return (
      <div className="flex h-screen bg-white">
        {/* Left Sidebar */}
        <div className="hidden md:flex md:w-[260px] md:flex-col bg-gray-50">
          <div className="flex h-full flex-col">
            <div className="flex items-center gap-3 p-3">
              <button
                onClick={handleBackToLanding}
                className="flex w-full items-center gap-3 rounded-md border border-gray-300 p-3 text-sm text-gray-700 transition-colors hover:bg-gray-100"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Home
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              {/* Article sources sidebar */}
              {selectedArticle && (
                <div className="space-y-4">
                  <div className="px-2">
                    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Video Sources</h3>
                    <div className="space-y-2">
                      {selectedArticle.videos.slice(0, 8).map((video, idx) => (
                        <a
                          key={idx}
                          href={video.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-start gap-2 p-2 rounded-lg hover:bg-gray-100 transition-colors group"
                        >
                          <img
                            src={video.thumbnail}
                            alt=""
                            className="w-16 h-10 object-cover rounded flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs text-gray-700 line-clamp-2 group-hover:text-cyan-600 transition-colors">
                              {video.title}
                            </p>
                            <p className="text-xs text-gray-500 truncate mt-0.5">
                              {video.channel}
                            </p>
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>
                  {selectedArticle.web_sources.length > 0 && (
                    <div className="px-2 pt-4 border-t border-gray-200">
                      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Web Sources</h3>
                      <div className="space-y-2">
                        {selectedArticle.web_sources.slice(0, 5).map((source, idx) => (
                          <a
                            key={idx}
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-2 rounded-lg hover:bg-gray-100 transition-colors group"
                          >
                            <p className="text-xs text-gray-700 line-clamp-2 group-hover:text-purple-600 transition-colors">
                              {source.title}
                            </p>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="border-t border-gray-300 p-4">
              <button
                onClick={handleStartConversationFromArticle}
                className="flex w-full items-center justify-center gap-2 rounded-md bg-cyan-600 hover:bg-cyan-700 p-3 text-sm text-white transition-colors"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Ask a question
              </button>
            </div>
          </div>
        </div>

        {/* Main Article Content */}
        <div className="flex-1 overflow-y-auto">
          {isLoadingArticle ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto mb-4"></div>
                <p className="text-gray-400">Loading article...</p>
              </div>
            </div>
          ) : selectedArticle ? (
            <div className="max-w-4xl mx-auto p-8">
              {/* Article Header */}
              <div className="mb-8">
                <div className="flex items-center gap-2 mb-4">
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-cyan-500/20 text-cyan-400">
                    {selectedArticle.topic}
                  </span>
                  <span className="text-gray-500 text-sm">
                    {new Date(selectedArticle.created_at).toLocaleDateString('en-US', {
                      month: 'long',
                      day: 'numeric',
                      year: 'numeric'
                    })}
                  </span>
                </div>
                <h1 className="text-3xl font-bold text-gray-900 mb-4">
                  {selectedArticle.article_title}
                </h1>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    {selectedArticle.videos.length} videos analyzed
                  </span>
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                    </svg>
                    {selectedArticle.web_sources.length} web sources
                  </span>
                </div>
              </div>

              {/* Article Content */}
              <div className="prose prose-lg max-w-none" onClick={handleArticleTimestampClick}>
                <ReactMarkdown
                  components={{
                    h3: ({ children }) => (
                      <h3 className="text-[#10a37f] text-xl font-semibold mt-8 mb-4 flex items-center gap-2">
                        {children}
                      </h3>
                    ),
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-4 border-[#10a37f] bg-[#10a37f]/5 px-4 py-3 my-4 rounded-r-lg italic text-gray-600">
                        {children}
                      </blockquote>
                    ),
                    a: ({ href, children }) => {
                      // Safely convert children to string for timestamp detection
                      const childText = typeof children === 'string'
                        ? children
                        : Array.isArray(children)
                          ? children.map(c => (typeof c === 'string' ? c : '')).join('')
                          : String(children || '')
                      const isTimestamp = /\[\d+:\d+\]/.test(childText)
                      return (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={isTimestamp
                            ? "text-cyan-400 hover:text-cyan-300 font-mono font-semibold px-1.5 py-0.5 bg-cyan-900/30 rounded cursor-pointer no-underline hover:bg-cyan-900/50 transition-colors"
                            : "text-[#10a37f] hover:text-[#14b88a] font-medium no-underline hover:underline"
                          }
                        >
                          {children}
                        </a>
                      )
                    },
                    strong: ({ children }) => (
                      <strong className="text-gray-900 font-semibold">{children}</strong>
                    ),
                    em: ({ children }) => (
                      <em className="text-gray-600 italic">{children}</em>
                    ),
                    li: ({ children }) => (
                      <li className="mb-2 text-gray-600">{children}</li>
                    ),
                    p: ({ children }) => (
                      <p className="text-gray-700 leading-relaxed mb-4">{children}</p>
                    ),
                  }}
                >
                  {selectedArticle.article_content}
                </ReactMarkdown>
              </div>

              {/* Video Sources Grid */}
              <div className="mt-12 pt-8 border-t border-gray-200">
                <h3 className="text-xl font-semibold text-gray-900 mb-6">Video Sources</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {selectedArticle.videos.map((video, idx) => (
                    <a
                      key={idx}
                      href={video.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="bg-gray-100 rounded-lg overflow-hidden hover:bg-gray-200 transition-colors group"
                    >
                      <div className="relative">
                        <img
                          src={video.thumbnail}
                          alt={video.title}
                          className="w-full h-32 object-cover group-hover:scale-105 transition-transform"
                        />
                        {video.has_transcript && (
                          <span className="absolute top-2 right-2 bg-green-600 text-white text-xs px-2 py-0.5 rounded">
                            Analyzed
                          </span>
                        )}
                      </div>
                      <div className="p-3">
                        <p className="text-sm font-medium text-gray-700 line-clamp-2 group-hover:text-cyan-600 transition-colors">
                          {video.title}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          {video.channel}
                        </p>
                        {(video.view_count || video.like_count) && (
                          <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                            {video.view_count && (
                              <span>{formatNumber(video.view_count)} views</span>
                            )}
                            {video.like_count && (
                              <span>{formatNumber(video.like_count)} likes</span>
                            )}
                          </div>
                        )}
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* YouTube Video Sidebar for Article View */}
        <div className={`${showVideoSidebar ? 'w-[480px]' : 'w-0'} transition-all duration-300 bg-gray-50 border-l border-gray-200 overflow-hidden flex flex-col`}>
          {showVideoSidebar && currentVideo && (
            <>
              {/* Sidebar Header */}
              <div className="p-4 border-b border-gray-200 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                  </svg>
                  <h3 className="text-gray-900 font-semibold text-sm">Video Preview</h3>
                </div>
                <button
                  onClick={() => setShowVideoSidebar(false)}
                  className="text-gray-400 hover:text-gray-900 transition-colors"
                  aria-label="Close video sidebar"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Video Player */}
              <div className="flex-1 flex flex-col p-4 overflow-y-auto">
                <div className="relative" style={{ paddingBottom: '56.25%' }}>
                  <iframe
                    className="absolute top-0 left-0 w-full h-full rounded-lg"
                    src={`https://www.youtube.com/embed/${currentVideo.videoId}?start=${currentVideo.startTime}&autoplay=1`}
                    title="YouTube video player"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>

                {/* Video Metadata Card */}
                <div className="mt-4 bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                  {/* Title & Channel Section */}
                  <div className="p-4 border-b border-gray-200">
                    {currentVideo.title && (
                      <h4 className="text-sm font-semibold text-gray-900 mb-3 line-clamp-2">
                        {currentVideo.title}
                      </h4>
                    )}

                    {currentVideo.channel && (
                      <div className="flex items-center gap-2 mb-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center">
                          <svg className="h-3 w-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                          </svg>
                        </div>
                        <p className="text-xs font-medium text-gray-700 truncate">
                          {currentVideo.channel}
                        </p>
                      </div>
                    )}

                    {/* Stats Row */}
                    {(currentVideo.view_count !== undefined || currentVideo.like_count !== undefined) && (
                      <div className="flex items-center gap-4 text-xs text-gray-400">
                        {currentVideo.view_count !== undefined && (
                          <div className="flex items-center gap-1.5">
                            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                            <span className="font-medium text-gray-700">{formatNumber(currentVideo.view_count)}</span>
                          </div>
                        )}
                        {currentVideo.like_count !== undefined && (
                          <div className="flex items-center gap-1.5">
                            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                            </svg>
                            <span className="font-medium text-gray-700">{formatNumber(currentVideo.like_count)}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Timestamp Section */}
                  <div className="p-4 bg-gray-100 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <div>
                        <p className="text-gray-400">Starting at</p>
                        <p className="font-mono font-semibold text-cyan-400">
                          {Math.floor(currentVideo.startTime / 60)}:{String(currentVideo.startTime % 60).padStart(2, '0')}
                        </p>
                      </div>
                    </div>
                    <a
                      href={`https://www.youtube.com/watch?v=${currentVideo.videoId}&t=${currentVideo.startTime}s`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-md font-medium transition-colors"
                    >
                      <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
                      </svg>
                      Open
                    </a>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    )
  }

  return (
    <AuthProvider>
    <div className="flex h-screen bg-white">
      {/* Left Sidebar - login aware */}
      <ConversationSidebar
        onNewChat={handleBackToLanding}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main content */}
      <div className="flex flex-1 flex-col">
        {/* Back Button Header - always visible */}
        <div className="flex items-center justify-between p-2 border-b border-gray-200">
          <button
            onClick={handleBackToLanding}
            className="flex items-center gap-2 text-sm text-gray-700 hover:text-gray-900 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Home
          </button>
          <div className="flex items-center gap-1.5">
            {/* Trends Toggle Button */}
            <button
              onClick={() => setShowTrendsSidebar(!showTrendsSidebar)}
              className={`px-2 py-1 rounded flex items-center gap-1 text-xs ${
                showTrendsSidebar ? 'bg-emerald-600' : 'bg-gray-100'
              } text-white hover:opacity-80 transition-all`}
              title="Toggle Trends Sidebar"
            >
              <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
              </svg>
              Trends
            </button>
            {/* Debug Toggle Button */}
            <button
              onClick={() => setShowDebug(!showDebug)}
              className={`p-1 rounded ${
                showDebug ? 'bg-[#10a37f] text-white' : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
              } hover:opacity-90 transition-all`}
              title="Toggle Reference Panel"
            >
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
            </button>
          </div>
        </div>
        {/* Messages area */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-4">
              <h1 className="mb-8 text-4xl font-semibold text-gray-900">
                Product Advisor
              </h1>
              <p className="mb-8 text-center text-lg text-gray-400 max-w-2xl">
                Get expert product recommendations powered by AI. I analyze YouTube reviews and expert opinions to give you comprehensive buying advice.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl w-full">
                {exampleQueries.map((query, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setInput(query)
                    }}
                    className="rounded-lg border border-gray-300 p-4 text-left text-sm text-gray-700 hover:bg-gray-200 transition-colors"
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="pb-32">
              {messages.map((message, index) => {
                // Skip rendering empty assistant messages (shouldn't happen, but safety check)
                if (message.role === 'assistant' && !message.content) {
                  return null
                }
                return (
                <div
                  key={index}
                  className={`${
                    message.role === 'assistant' ? 'bg-gray-50' : ''
                  }`}
                >
                  <div className="mx-auto flex max-w-3xl gap-4 p-4 md:px-6 md:py-6">
                    {/* Avatar */}
                    <div
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-sm ${
                        message.role === 'assistant'
                          ? 'bg-[#10a37f]'
                          : 'bg-purple-600'
                      }`}
                    >
                      {message.role === 'assistant' ? (
                        <svg
                          className="h-5 w-5 text-white"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                        >
                          <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
                        </svg>
                      ) : (
                        <svg
                          className="h-5 w-5 text-white"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                        >
                          <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                        </svg>
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 space-y-4 overflow-hidden" onClick={(e) => handleTimestampClick(e, message.videos, message.debug)}>
                      <div className="prose max-w-none">
                        <ReactMarkdown
                          components={{
                            // Custom heading components with icons
                            h3: ({ children }) => (
                              <h3 className="text-[#10a37f] text-lg font-semibold mt-6 mb-2 flex items-center gap-2">
                                {children}
                              </h3>
                            ),
                            // Custom blockquote for reviewer quotes
                            blockquote: ({ children }) => (
                              <blockquote className="border-l-4 border-[#10a37f] bg-[#10a37f]/5 px-4 py-3 my-3 rounded-r-lg italic text-gray-600">
                                {children}
                              </blockquote>
                            ),
                            // Custom link styling
                            a: ({ href, children }) => {
                              // Safely convert children to string for timestamp detection
                              const childText = typeof children === 'string'
                                ? children
                                : Array.isArray(children)
                                  ? children.map(c => (typeof c === 'string' ? c : '')).join('')
                                  : String(children || '')
                              const isTimestamp = /\[\d+:\d+\]/.test(childText)
                              return (
                                <a
                                  href={href}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className={isTimestamp
                                    ? "text-cyan-400 hover:text-cyan-300 font-mono font-semibold px-1.5 py-0.5 bg-cyan-900/30 rounded cursor-pointer no-underline hover:bg-cyan-900/50 transition-colors"
                                    : "text-[#10a37f] hover:text-[#14b88a] font-medium no-underline hover:underline"
                                  }
                                >
                                  {children}
                                </a>
                              )
                            },
                            // Custom strong for product names
                            strong: ({ children }) => (
                              <strong className="text-gray-900 font-semibold">{children}</strong>
                            ),
                            // Custom emphasis/italic
                            em: ({ children }) => (
                              <em className="text-gray-600 italic">{children}</em>
                            ),
                            // Custom list items
                            li: ({ children }) => (
                              <li className="mb-2 text-gray-600">{children}</li>
                            ),
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>

                      {/* Video cards */}
                      {message.videos && message.videos.length > 0 && (
                        <div className="mt-4">
                          <h4 className="text-sm font-medium text-gray-400 mb-3">
                            Sources ({message.sources})
                          </h4>
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            {message.videos.map((video, i) => (
                              <a
                                key={i}
                                href={video.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={`flex flex-col rounded-lg border overflow-hidden hover:border-gray-500 transition-colors bg-white ${
                                  video.has_transcript ? 'border-green-600' : 'border-gray-300'
                                }`}
                              >
                                <div className="relative">
                                  <img
                                    src={video.thumbnail}
                                    alt={video.title}
                                    className="w-full h-24 object-cover"
                                  />
                                  {video.has_transcript && (
                                    <span className="absolute top-1 right-1 bg-green-600 text-white text-xs px-1.5 py-0.5 rounded">
                                      Analyzed
                                    </span>
                                  )}
                                </div>
                                <div className="p-2">
                                  <p className="text-xs font-medium text-white line-clamp-2">
                                    {video.title}
                                  </p>
                                  <p className="text-xs text-gray-400 mt-1">
                                    {video.channel}
                                  </p>
                                </div>
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )})}

              {/* Multi-step Progress indicator (Claude Code style) */}
              {isLoading && progressSteps.length > 0 && (
                <div className="bg-gray-50">
                  <div className="mx-auto flex max-w-3xl gap-4 p-4 md:px-6 md:py-6">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm bg-[#10a37f]">
                      <svg
                        className="h-5 w-5 text-white animate-spin"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                      >
                        <circle className="opacity-25" cx="12" cy="12" r="10" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    </div>
                    <div className="flex flex-col gap-1.5 flex-1">
                      {progressSteps.map((step, index) => (
                        <div key={step.id || index} className="flex items-center gap-2.5 text-sm">
                          {/* Status Icon */}
                          {step.status === 'completed' ? (
                            <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          ) : step.status === 'in_progress' ? (
                            <svg className="w-4 h-4 text-cyan-400 animate-spin flex-shrink-0" viewBox="0 0 24 24" fill="none">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                          ) : (
                            <div className="w-4 h-4 rounded-full border border-gray-500 flex-shrink-0" />
                          )}

                          {/* Step Text */}
                          <span className={`${
                            step.status === 'in_progress' ? 'text-cyan-400' :
                            step.status === 'completed' ? 'text-gray-400' : 'text-gray-500'
                          }`}>
                            {step.message}
                            {step.detail && (
                              <span className="text-gray-500 ml-1.5">{step.detail}</span>
                            )}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="border-t border-gray-300 bg-white p-4">
          <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
            <div className="relative flex items-center">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask what reviewers really think..."
                className="w-full rounded-lg border border-gray-300 bg-white p-4 pr-12 text-gray-900 placeholder-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                disabled={isLoading}
              />
              {isLoading ? (
                <button
                  type="button"
                  onClick={handleStopGeneration}
                  className="absolute right-3 rounded-md p-2 text-gray-400 hover:text-red-400 transition-colors"
                  title="Stop generation"
                >
                  <svg
                    fill="currentColor"
                    viewBox="0 0 24 24"
                    className="h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="absolute right-3 rounded-md p-2 text-gray-400 hover:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg
                    stroke="currentColor"
                    fill="none"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              )}
            </div>
            <p className="mt-2 text-center text-xs text-gray-500">
              Product Advisor analyzes YouTube reviews to provide recommendations. Results may vary.
            </p>
          </form>
        </div>
      </div>

      {/* YouTube Video Sidebar */}
      <div className={`${showVideoSidebar ? 'w-[480px]' : 'w-0'} transition-all duration-300 bg-gray-50 border-l border-gray-200 overflow-hidden flex flex-col`}>
        {showVideoSidebar && currentVideo && (
          <>
            {/* Sidebar Header */}
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <svg className="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                </svg>
                <h3 className="text-gray-900 font-semibold text-sm">Video Preview</h3>
              </div>
              <button
                onClick={() => setShowVideoSidebar(false)}
                className="text-gray-400 hover:text-gray-900 transition-colors"
                aria-label="Close video sidebar"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Video Player */}
            <div className="flex-1 flex flex-col p-4 overflow-y-auto">
              <div className="relative" style={{ paddingBottom: '56.25%' }}>
                <iframe
                  className="absolute top-0 left-0 w-full h-full rounded-lg"
                  src={`https://www.youtube.com/embed/${currentVideo.videoId}?start=${currentVideo.startTime}&autoplay=1`}
                  title="YouTube video player"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>

              {/* Video Metadata Card */}
              <div className="mt-4 bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                {/* Title & Channel Section */}
                <div className="p-4 border-b border-gray-200">
                  {currentVideo.title && (
                    <h4 className="text-sm font-semibold text-gray-900 mb-3 line-clamp-2">
                      {currentVideo.title}
                    </h4>
                  )}

                  {currentVideo.channel && (
                    <div className="flex items-center gap-2 mb-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center">
                        <svg className="h-3 w-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                        </svg>
                      </div>
                      <p className="text-xs font-medium text-gray-700 truncate">
                        {currentVideo.channel}
                      </p>
                    </div>
                  )}

                  {/* Stats Row */}
                  {(currentVideo.view_count !== undefined || currentVideo.like_count !== undefined) && (
                    <div className="flex items-center gap-4 text-xs text-gray-400">
                      {currentVideo.view_count !== undefined && (
                        <div className="flex items-center gap-1.5">
                          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                          </svg>
                          <span className="font-medium text-gray-700">{formatNumber(currentVideo.view_count)}</span>
                        </div>
                      )}
                      {currentVideo.like_count !== undefined && (
                        <div className="flex items-center gap-1.5">
                          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                          </svg>
                          <span className="font-medium text-gray-700">{formatNumber(currentVideo.like_count)}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Timestamp Section */}
                <div className="p-4 bg-gray-100 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <svg className="h-4 w-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                      <p className="text-gray-400">Starting at</p>
                      <p className="font-mono font-semibold text-cyan-400">
                        {Math.floor(currentVideo.startTime / 60)}:{String(currentVideo.startTime % 60).padStart(2, '0')}
                      </p>
                    </div>
                  </div>
                  <a
                    href={`https://www.youtube.com/watch?v=${currentVideo.videoId}&t=${currentVideo.startTime}s`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-md font-medium transition-colors"
                  >
                    <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
                    </svg>
                    Open
                  </a>
                </div>
              </div>

              {/* Video Insights Section */}
              {currentVideo.insights && (
                <div className="mt-4 bg-gray-50 rounded-xl border border-gray-200 overflow-hidden">
                  <div className="p-4 border-b border-gray-200 flex items-center gap-2">
                    <svg className="h-4 w-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                    </svg>
                    <h4 className="text-sm font-semibold text-gray-900">AI Analysis</h4>
                  </div>

                  <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
                    {/* Products analyzed */}
                    {currentVideo.insights.products?.map((product: any, idx: number) => (
                      <div key={idx} className="space-y-2">
                        <div className="flex items-start justify-between">
                          <h5 className="text-sm font-semibold text-purple-300">{product.name}</h5>
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            product.overall_sentiment === 'positive' ? 'bg-green-900/30 text-green-300' :
                            product.overall_sentiment === 'negative' ? 'bg-red-900/30 text-red-300' :
                            product.overall_sentiment === 'mixed' ? 'bg-yellow-900/30 text-yellow-300' :
                            'bg-gray-200 text-gray-700'
                          }`}>
                            {product.overall_sentiment}
                          </span>
                        </div>

                        {product.summary && (
                          <p className="text-xs text-gray-700 leading-relaxed">{product.summary}</p>
                        )}

                        {/* Pros/Cons */}
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {product.pros && product.pros.length > 0 && (
                            <div className="bg-green-900/10 rounded p-2">
                              <p className="font-semibold text-green-400 mb-1">Pros</p>
                              <ul className="space-y-1">
                                {product.pros.slice(0, 3).map((pro: string, i: number) => (
                                  <li key={i} className="text-gray-700 flex items-start gap-1">
                                    <span className="text-green-400 mt-0.5">+</span>
                                    <span>{pro}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {product.cons && product.cons.length > 0 && (
                            <div className="bg-red-900/10 rounded p-2">
                              <p className="font-semibold text-red-400 mb-1">Cons</p>
                              <ul className="space-y-1">
                                {product.cons.slice(0, 3).map((con: string, i: number) => (
                                  <li key={i} className="text-gray-700 flex items-start gap-1">
                                    <span className="text-red-400 mt-0.5">−</span>
                                    <span>{con}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>

                        {/* Top quotes */}
                        {product.top_quotes && product.top_quotes.length > 0 && (
                          <div className="space-y-1.5">
                            <p className="text-xs font-semibold text-gray-400">Key Quotes</p>
                            {product.top_quotes.slice(0, 2).map((quote: any, i: number) => (
                              <div key={i} className="bg-gray-100 rounded p-2 border-l-2 border-cyan-500">
                                <p className="text-xs text-gray-700 italic">"{quote.text}"</p>
                                {quote.timestamp && (
                                  <p className="text-xs text-cyan-400 font-mono mt-1">{quote.timestamp}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}

                    {/* Overall tone */}
                    {currentVideo.insights.tone && (
                      <div className="pt-3 border-t border-gray-200">
                        <p className="text-xs text-gray-400">
                          <span className="font-semibold">Overall tone:</span>{' '}
                          <span className="text-gray-700">{currentVideo.insights.tone}</span>
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Right Debug Panel */}
      <div className={`${showDebug ? 'w-[320px]' : 'w-0'} transition-all duration-300 bg-gray-50 border-l border-gray-200 overflow-hidden`}>
        {showDebug && latestDebug && (
          <div className="p-4 h-full overflow-y-auto">
            <h3 className="text-gray-900 font-semibold mb-4 flex items-center gap-2">
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
              Reference
            </h3>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className="bg-white rounded p-2">
                <div className="text-2xl font-bold text-gray-900">{latestDebug.generated_queries.length}</div>
                <div className="text-xs text-gray-400">Video Queries</div>
              </div>
              <div className="bg-white rounded p-2">
                <div className="text-2xl font-bold text-gray-900">{latestDebug.web_queries?.length || 0}</div>
                <div className="text-xs text-gray-400">Web Queries</div>
              </div>
              <div className="bg-white rounded p-2">
                <div className="text-2xl font-bold text-green-400">{latestDebug.videos_analyzed}</div>
                <div className="text-xs text-gray-400">Videos Analyzed</div>
              </div>
              <div className="bg-white rounded p-2">
                <div className="text-2xl font-bold text-purple-400">{latestDebug.total_web_results || 0}</div>
                <div className="text-xs text-gray-400">Web Articles</div>
              </div>
            </div>

            {/* Timing Info */}
            {latestDebug.timing && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-cyan-400 mb-2 flex items-center gap-1">
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/>
                  </svg>
                  Processing Time: {latestDebug.timing.total_seconds.toFixed(1)}s
                </h4>
                <div className="bg-white rounded-lg p-3">
                  <div className="space-y-1.5">
                    {[
                      { label: 'Query Classification', value: latestDebug.timing.classify_query_seconds, color: 'bg-blue-500' },
                      { label: 'Generate Queries', value: latestDebug.timing.generate_queries_seconds, color: 'bg-indigo-500' },
                      { label: 'YouTube Search', value: latestDebug.timing.youtube_search_seconds, color: 'bg-red-500' },
                      { label: 'Video Ranking (LLM)', value: latestDebug.timing.video_ranking_seconds, color: 'bg-orange-500' },
                      { label: 'Web Search', value: latestDebug.timing.web_search_seconds, color: 'bg-purple-500' },
                      { label: 'Transcript Fetch', value: latestDebug.timing.transcript_fetch_seconds, color: 'bg-green-500' },
                      { label: 'Insights Extraction', value: latestDebug.timing.insights_extraction_seconds, color: 'bg-teal-500' },
                      { label: 'Synthesis', value: latestDebug.timing.synthesis_seconds, color: 'bg-cyan-500' },
                      { label: 'Answer Generation', value: latestDebug.timing.answer_generation_seconds, color: 'bg-yellow-500' },
                    ].map((item, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <div className="w-28 text-xs text-gray-400 truncate">{item.label}</div>
                        <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${item.color} rounded-full`}
                            style={{ width: `${Math.min((item.value / latestDebug.timing!.total_seconds) * 100, 100)}%` }}
                          />
                        </div>
                        <div className="w-12 text-xs text-gray-700 text-right">{item.value.toFixed(1)}s</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Answer Generation Debug */}
            {latestDebug.answer_generation && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-yellow-400 mb-2 flex items-center gap-1">
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/>
                  </svg>
                  Answer Generation
                </h4>
                <div className="bg-white rounded-lg p-3 space-y-2">
                  {/* Method Badge */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Method:</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      latestDebug.answer_generation.method_used === 'structured'
                        ? 'bg-green-600 text-white'
                        : latestDebug.answer_generation.method_used === 'structured_no_synthesis'
                        ? 'bg-yellow-600 text-white'
                        : latestDebug.answer_generation.method_used === 'error'
                        ? 'bg-red-600 text-white'
                        : 'bg-gray-600 text-white'
                    }`}>
                      {latestDebug.answer_generation.method_used.toUpperCase().replace('_', ' ')}
                    </span>
                  </div>

                  {/* Fallback Reason */}
                  {latestDebug.answer_generation.fallback_reason && (
                    <div className="text-xs">
                      <span className="text-gray-400">Reason: </span>
                      <span className="text-orange-300">{latestDebug.answer_generation.fallback_reason}</span>
                    </div>
                  )}

                  {/* Video Insights (Collapsible) */}
                  {latestDebug.answer_generation.video_insights && (
                    <details className="group">
                      <summary className="text-xs text-blue-400 cursor-pointer hover:text-blue-300 flex items-center gap-1">
                        <svg className="h-3 w-3 transition-transform group-open:rotate-90" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/>
                        </svg>
                        Video Insights ({latestDebug.answer_generation.video_insights.length} videos)
                      </summary>
                      <pre className="mt-2 p-2 bg-gray-100 rounded text-xs text-gray-700 overflow-x-auto max-h-64 overflow-y-auto">
                        {JSON.stringify(latestDebug.answer_generation.video_insights, null, 2)}
                      </pre>
                    </details>
                  )}

                  {/* Synthesis (Collapsible) */}
                  {latestDebug.answer_generation.synthesis && (
                    <details className="group">
                      <summary className="text-xs text-cyan-400 cursor-pointer hover:text-cyan-300 flex items-center gap-1">
                        <svg className="h-3 w-3 transition-transform group-open:rotate-90" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/>
                        </svg>
                        Cross-Video Synthesis
                      </summary>
                      <pre className="mt-2 p-2 bg-gray-100 rounded text-xs text-gray-700 overflow-x-auto max-h-64 overflow-y-auto">
                        {JSON.stringify(latestDebug.answer_generation.synthesis, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            )}

            {/* Ranked Videos for Analysis */}
            {latestDebug.ranked_videos && latestDebug.ranked_videos.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-orange-400 mb-2 flex items-center gap-1">
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/>
                  </svg>
                  Top {latestDebug.ranked_videos.length} Ranked Videos (LLM Selected)
                </h4>
                <div className="bg-white rounded-lg p-3">
                  <div className="space-y-2">
                    {latestDebug.ranked_videos.map((video, idx) => (
                      <a
                        key={idx}
                        href={video.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={`flex items-start gap-2 p-2 rounded transition-colors ${
                          video.has_transcript
                            ? 'bg-green-900/30 hover:bg-green-900/50 border border-green-600/50'
                            : 'bg-gray-100 hover:bg-gray-200 border border-transparent'
                        }`}
                      >
                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-orange-600 flex items-center justify-center text-xs font-bold text-white">
                          {idx + 1}
                        </div>
                        <img
                          src={video.thumbnail}
                          alt=""
                          className="w-16 h-10 object-cover rounded flex-shrink-0"
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-gray-700 font-medium line-clamp-2">
                            {video.title}
                          </p>
                          <p className="text-xs text-gray-500 truncate mt-0.5">
                            {video.channel}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            {video.has_transcript ? (
                              <span className="text-xs bg-green-600 text-white px-1.5 py-0.5 rounded">
                                Transcript OK
                              </span>
                            ) : (
                              <span className="text-xs bg-red-600/70 text-white px-1.5 py-0.5 rounded">
                                No Transcript
                              </span>
                            )}
                            {video.source_query && (
                              <span className="text-xs text-gray-500 truncate max-w-[120px]" title={video.source_query}>
                                Q: {video.source_query.slice(0, 20)}...
                              </span>
                            )}
                          </div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Video Query Results */}
            <div className="mb-4">
              <h4 className="text-sm font-medium text-green-400 mb-2 flex items-center gap-1">
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
                </svg>
                YouTube Search Results
              </h4>
              <div className="space-y-3">
                {latestDebug.query_results.map((qr, idx) => (
                  <div key={idx} className="bg-white rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">Query {idx + 1}</div>
                    <div className="text-sm text-gray-700 font-medium mb-2 break-words">
                      "{qr.query}"
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                      {qr.videos.length} videos returned:
                    </div>
                    <div className="space-y-1.5">
                      {qr.videos.map((video, vIdx) => (
                        <a
                          key={vIdx}
                          href={video.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 p-1.5 rounded bg-gray-100 hover:bg-gray-200 transition-colors"
                        >
                          <img
                            src={video.thumbnail}
                            alt=""
                            className="w-12 h-8 object-cover rounded"
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs text-gray-700 truncate">
                              {video.title}
                            </p>
                            <p className="text-xs text-gray-500 truncate">
                              {video.channel}
                            </p>
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Web Query Results */}
            {latestDebug.web_query_results && latestDebug.web_query_results.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-purple-400 mb-2 flex items-center gap-1">
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                  </svg>
                  Web Articles
                </h4>
                <div className="space-y-3">
                  {latestDebug.web_query_results.map((wqr, idx) => (
                    <div key={idx} className="bg-white rounded-lg p-3">
                      <div className="text-xs text-gray-400 mb-1">Query {idx + 1}</div>
                      <div className="text-sm text-gray-700 font-medium mb-2 break-words">
                        "{wqr.query}"
                      </div>
                      <div className="text-xs text-gray-400 mb-2">
                        {wqr.results.length} articles returned:
                      </div>
                      <div className="space-y-1.5">
                        {wqr.results.map((result, rIdx) => (
                          <a
                            key={rIdx}
                            href={result.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-2 rounded bg-gray-100 hover:bg-gray-200 transition-colors"
                          >
                            <p className="text-xs text-white font-medium line-clamp-2 mb-1">
                              {result.title}
                            </p>
                            <p className="text-xs text-gray-500 line-clamp-2">
                              {result.content}
                            </p>
                          </a>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Trends Sidebar - toggleable in conversation mode */}
      <div className={`${showTrendsSidebar ? 'w-[380px]' : 'w-0'} transition-all duration-300 bg-slate-100 overflow-hidden`}>
        {showTrendsSidebar && (
          <TrendsSidebar onViewArticle={handleViewArticle} apiUrl={API_URL} />
        )}
      </div>

    </div>
    </AuthProvider>
  )
}
