import React, { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { getFarms } from './api'

const FarmContext = createContext({})
const SELECTED_FARM_KEY = 'mango_selected_farm'

/** Loads the signed-in user's farms and tracks which one is currently selected. */
export const FarmProvider = ({ children }) => {
  const [farms, setFarms] = useState([])
  const [selectedId, setSelectedId] = useState(() => localStorage.getItem(SELECTED_FARM_KEY))
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const res = await getFarms()
      setFarms(res.data)
      return res.data
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh().catch(() => {})
  }, [refresh])

  const selectFarm = (id) => {
    setSelectedId(id)
    localStorage.setItem(SELECTED_FARM_KEY, id)
  }

  const farm = farms.find((f) => f.id === selectedId) || farms[0] || null

  return (
    <FarmContext.Provider value={{ farms, farm, selectFarm, refresh, loading }}>
      {children}
    </FarmContext.Provider>
  )
}

export const useFarm = () => useContext(FarmContext)
