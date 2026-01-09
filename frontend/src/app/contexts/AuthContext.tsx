'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface User {
  id: number
  email: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  loginWithGoogle: () => Promise<void>
  loginWithGithub: () => Promise<void>
  logout: () => void
  error: string | null
  clearError: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const TOKEN_KEY = 'auth_token'

// OAuth configuration (set these in environment variables)
const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || ''
const GITHUB_CLIENT_ID = process.env.NEXT_PUBLIC_GITHUB_CLIENT_ID || ''

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Check for existing token on mount
  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY)
    if (token) {
      fetchUser(token)
    } else {
      setIsLoading(false)
    }
  }, [])

  const fetchUser = async (token: string) => {
    try {
      const response = await fetch(`${API_URL}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      if (response.ok) {
        const userData = await response.json()
        setUser(userData)
      } else {
        // Token invalid, clear it
        localStorage.removeItem(TOKEN_KEY)
      }
    } catch (err) {
      console.error('Failed to fetch user:', err)
      localStorage.removeItem(TOKEN_KEY)
    } finally {
      setIsLoading(false)
    }
  }

  const login = async (email: string, password: string) => {
    setError(null)
    try {
      const response = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
      })

      // Handle non-JSON responses (like 404 for missing endpoint)
      const contentType = response.headers.get('content-type')
      if (!contentType || !contentType.includes('application/json')) {
        if (response.status === 404) {
          throw new Error('Authentication service not available. Please try again later.')
        }
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed')
      }

      localStorage.setItem(TOKEN_KEY, data.token)
      setUser(data.user)
      // Dispatch auth-change event so other components can react
      window.dispatchEvent(new Event('auth-change'))
    } catch (err) {
      let message = 'Login failed'
      if (err instanceof TypeError && err.message.includes('fetch')) {
        message = 'Unable to connect to server. Please check your connection.'
      } else if (err instanceof Error) {
        message = err.message
      }
      setError(message)
      throw err
    }
  }

  const register = async (email: string, password: string) => {
    setError(null)
    try {
      const response = await fetch(`${API_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
      })

      // Handle non-JSON responses (like 404 for missing endpoint)
      const contentType = response.headers.get('content-type')
      if (!contentType || !contentType.includes('application/json')) {
        if (response.status === 404) {
          throw new Error('Authentication service not available. Please try again later.')
        }
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed')
      }

      localStorage.setItem(TOKEN_KEY, data.token)
      setUser(data.user)
      // Dispatch auth-change event so other components can react
      window.dispatchEvent(new Event('auth-change'))
    } catch (err) {
      let message = 'Registration failed'
      if (err instanceof TypeError && err.message.includes('fetch')) {
        message = 'Unable to connect to server. Please check your connection.'
      } else if (err instanceof Error) {
        message = err.message
      }
      setError(message)
      throw err
    }
  }

  const handleOAuthCallback = async (provider: string, accessToken: string) => {
    setError(null)
    try {
      const response = await fetch(`${API_URL}/auth/oauth`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ provider, access_token: accessToken })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'OAuth login failed')
      }

      localStorage.setItem(TOKEN_KEY, data.token)
      setUser(data.user)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'OAuth login failed'
      setError(message)
      throw err
    }
  }

  const loginWithGoogle = async () => {
    if (!GOOGLE_CLIENT_ID) {
      setError('Google OAuth is not configured')
      return
    }

    // Open Google OAuth popup
    const width = 500
    const height = 600
    const left = window.screenX + (window.outerWidth - width) / 2
    const top = window.screenY + (window.outerHeight - height) / 2

    const redirectUri = `${window.location.origin}/auth/callback/google`
    const scope = 'email profile'

    const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${GOOGLE_CLIENT_ID}&` +
      `redirect_uri=${encodeURIComponent(redirectUri)}&` +
      `response_type=token&` +
      `scope=${encodeURIComponent(scope)}`

    const popup = window.open(
      authUrl,
      'Google Sign In',
      `width=${width},height=${height},left=${left},top=${top}`
    )

    // Listen for OAuth callback
    const handleMessage = async (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return
      if (event.data?.type === 'oauth_callback' && event.data?.provider === 'google') {
        window.removeEventListener('message', handleMessage)
        popup?.close()
        if (event.data.access_token) {
          await handleOAuthCallback('google', event.data.access_token)
        } else if (event.data.error) {
          setError(event.data.error)
        }
      }
    }
    window.addEventListener('message', handleMessage)
  }

  const loginWithGithub = async () => {
    if (!GITHUB_CLIENT_ID) {
      setError('GitHub OAuth is not configured')
      return
    }

    // Open GitHub OAuth popup
    const width = 500
    const height = 600
    const left = window.screenX + (window.outerWidth - width) / 2
    const top = window.screenY + (window.outerHeight - height) / 2

    const redirectUri = `${window.location.origin}/auth/callback/github`
    const scope = 'user:email'

    const authUrl = `https://github.com/login/oauth/authorize?` +
      `client_id=${GITHUB_CLIENT_ID}&` +
      `redirect_uri=${encodeURIComponent(redirectUri)}&` +
      `scope=${encodeURIComponent(scope)}`

    const popup = window.open(
      authUrl,
      'GitHub Sign In',
      `width=${width},height=${height},left=${left},top=${top}`
    )

    // Listen for OAuth callback
    const handleMessage = async (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return
      if (event.data?.type === 'oauth_callback' && event.data?.provider === 'github') {
        window.removeEventListener('message', handleMessage)
        popup?.close()
        if (event.data.access_token) {
          await handleOAuthCallback('github', event.data.access_token)
        } else if (event.data.error) {
          setError(event.data.error)
        }
      }
    }
    window.addEventListener('message', handleMessage)
  }

  const logout = () => {
    localStorage.removeItem(TOKEN_KEY)
    setUser(null)
  }

  const clearError = () => {
    setError(null)
  }

  return (
    <AuthContext.Provider value={{
      user,
      isLoading,
      login,
      register,
      loginWithGoogle,
      loginWithGithub,
      logout,
      error,
      clearError
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
