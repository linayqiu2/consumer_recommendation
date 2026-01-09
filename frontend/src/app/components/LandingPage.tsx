'use client'

import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import AuthModal from './AuthModal'

interface LandingPageProps {
  onStartConversation: (query?: string) => void
}

const EXAMPLE_QUERIES = [
  "Is the Model Y Highland actually quieter?",
  "Do reviewers agree on the Sony A7IV lowlight?",
  "What do critics miss about AirPods Pro 2?",
  "Samsung S24 Ultra vs iPhone 15 Pro cameras"
]

export default function LandingPage({ onStartConversation }: LandingPageProps) {
  const [inputValue, setInputValue] = useState('')
  const [showMethodology, setShowMethodology] = useState(false)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login')

  const { user, logout, isLoading } = useAuth()

  const openLogin = () => {
    setAuthMode('login')
    setShowAuthModal(true)
  }

  const openSignup = () => {
    setAuthMode('signup')
    setShowAuthModal(true)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputValue.trim()) {
      onStartConversation(inputValue.trim())
    }
  }

  const handleExampleClick = (query: string) => {
    onStartConversation(query)
  }

  return (
    <div className="relative flex-1 flex flex-col justify-center py-8 px-8 md:py-12 md:px-12 overflow-hidden bg-white -mt-24 md:-mt-28">
      {/* Auth buttons - top right */}
      <div className="absolute top-4 right-4 flex items-center gap-3 z-20">
        {isLoading ? (
          <div className="w-20 h-8 bg-gray-200 rounded animate-pulse" />
        ) : user ? (
          <div className="flex items-center gap-3">
            <span className="text-gray-500 text-sm hidden sm:inline">{user.email}</span>
            <button
              onClick={logout}
              className="text-gray-500 hover:text-gray-900 text-sm transition-colors"
            >
              Log out
            </button>
          </div>
        ) : (
          <>
            <button
              onClick={openLogin}
              className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors"
            >
              Log in
            </button>
            <button
              onClick={openSignup}
              className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-white text-sm font-semibold rounded-lg transition-all shadow-sm hover:shadow"
            >
              Sign up free
            </button>
          </>
        )}
      </div>

      {/* Decorative gradient background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'linear-gradient(to right, rgba(0,0,0,0.01), rgba(0,0,0,0))',
          boxShadow: 'inset -120px 0 160px rgba(80,180,255,0.03)',
        }}
      />

      <div className="relative z-10 w-full">
        {/* Headline */}
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-5 text-center text-gray-900 tracking-tight leading-tight">
          Review Video Insights at Scale
        </h1>

        {/* Subhead */}
        <p className="text-gray-600 text-lg sm:text-xl mb-8 text-center max-w-3xl mx-auto leading-relaxed">
          We analyze <span className="text-cyan-600 font-semibold">thousands of hours of review videos</span> to extract what reviewers{' '}
          <span className="text-gray-800 font-medium">agree on</span>,{' '}
          <span className="text-gray-800 font-medium">argue about</span>, and{' '}
          <span className="text-gray-800 font-medium">miss</span>.
        </p>

        {/* Trust Chips with hover expand */}
        <div className="group mb-8 max-w-3xl mx-auto">
          <div className="flex flex-wrap items-center justify-center gap-5 text-sm text-gray-600 font-medium">
            <span className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Timestamped quotes
            </span>
            <span className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-amber-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Disagreement detection
            </span>
            <span className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Bias-aware
            </span>
            <span className="text-gray-500 text-xs opacity-60 group-hover:opacity-0 transition-opacity">(hover for details)</span>
          </div>

          {/* Expandable feature details - CSS-only hover */}
          <div className="grid grid-cols-2 gap-3 p-4 bg-gray-50 rounded-xl border border-gray-200 mt-4 max-h-0 opacity-0 overflow-hidden group-hover:max-h-32 group-hover:opacity-100 transition-all duration-300">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <span className="text-gray-700">Video analysis</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="text-gray-700">Verified sources</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <span className="text-gray-700">Cross-review</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-8 h-8 rounded-lg bg-orange-500/20 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="text-gray-700">Real-time</span>
            </div>
          </div>
        </div>

        {/* Search Input - Primary CTA */}
        <form onSubmit={handleSubmit} className="mb-4 max-w-3xl mx-auto w-full">
          <div className="relative shadow-sm">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder='Ask what reviewers really think...'
              className="w-full h-14 px-6 pr-14 text-base border border-gray-200 bg-white text-gray-900 placeholder-gray-400 rounded-xl focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20 transition-all"
            />
            <button
              type="submit"
              disabled={!inputValue.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-lg bg-cyan-500 text-white hover:bg-cyan-400 disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center justify-center shadow-sm"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </div>
        </form>

        {/* Microcopy */}
        <p className="text-xs text-gray-500 mb-6 text-center max-w-3xl mx-auto">
          Answers include video quotes & timestamps from real reviewers
        </p>

        {/* Example Queries - 2x2 grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-10 max-w-3xl mx-auto">
          {EXAMPLE_QUERIES.map((query, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(query)}
              className="px-4 py-3 rounded-xl bg-gray-50 text-gray-700 text-sm hover:bg-gray-100 hover:text-gray-900 transition-all border border-gray-200 hover:border-gray-300 hover:shadow-sm text-left"
            >
              {query}
            </button>
          ))}
        </div>

        {/* Recent Insight + Methodology */}
        <div className="pt-8 border-t border-gray-100 max-w-3xl mx-auto">
          <h2 className="text-xs uppercase text-gray-400 tracking-widest mb-3 font-semibold">Recent insight</h2>
          <p className="italic text-gray-600 mb-5 text-sm leading-relaxed">
            "The Model Y Highland's cabin noise reduction was confirmed by 7 of 9 reviewers we analyzed..."
          </p>

          <button
            onClick={() => setShowMethodology(!showMethodology)}
            className="text-sm text-gray-500 hover:text-gray-700 transition-colors flex items-center gap-1.5 font-medium"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            How we analyze review videos
            <svg className={`w-3 h-3 transition-transform ${showMethodology ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showMethodology && (
            <div className="mt-3 p-4 bg-gray-50 rounded-xl border border-gray-200 text-sm text-gray-600">
              <p className="mb-2"><strong className="text-gray-900">1. Video Discovery:</strong> We search for highly-viewed review videos from established tech channels.</p>
              <p className="mb-2"><strong className="text-gray-900">2. Transcript Analysis:</strong> AI extracts key opinions, comparisons, and specific claims with timestamps.</p>
              <p className="mb-2"><strong className="text-gray-900">3. Cross-Reviewer Synthesis:</strong> We identify where reviewers agree, disagree, or provide unique perspectives.</p>
              <p><strong className="text-gray-900">4. Bias Detection:</strong> Sponsored content and potential biases are flagged when detected.</p>
            </div>
          )}
        </div>
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        initialMode={authMode}
      />
    </div>
  )
}
