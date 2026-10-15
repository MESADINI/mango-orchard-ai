import React, { createContext, useCallback, useContext, useEffect, useState } from 'react'
import * as api from './api'

const AuthContext = createContext({})

/**
 * Authentication against the FastAPI backend (/api/auth). The JWT is kept in
 * localStorage and validated with /api/auth/me on load.
 */
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  const signOut = useCallback(() => {
    localStorage.removeItem(api.TOKEN_KEY)
    setUser(null)
  }, [])

  useEffect(() => {
    api.onUnauthorized(signOut)
    if (!localStorage.getItem(api.TOKEN_KEY)) {
      setLoading(false)
      return
    }
    api.getMe()
      .then((res) => setUser(res.data))
      .catch(() => signOut())
      .finally(() => setLoading(false))
  }, [signOut])

  const handleAuth = (res) => {
    localStorage.setItem(api.TOKEN_KEY, res.data.access_token)
    setUser(res.data.user)
    return res.data.user
  }

  const signIn = async (email, password) => handleAuth(await api.login(email, password))
  const signUp = async (email, password, fullName) => handleAuth(await api.register(email, password, fullName))

  return (
    <AuthContext.Provider value={{ user, loading, signIn, signUp, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
