import React, { useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { addTree, createFarm, errorMessage, getFarmTrees, updateFarm } from '../services/api'
import { useFarm } from '../services/FarmContext'

const CULTIVARS = ['Alphonso', 'Banganapalli', 'Alphonso & Banganapalli', 'Dasheri', 'Totapuri', 'Kesar', 'Other']
const EMPTY_FARM = { name: '', cultivar: 'Alphonso', acreage: '', tree_count: '', latitude: '', longitude: '' }
const EMPTY_TREE = { tag_code: '', row_number: '', tree_number: '', variety: 'Alphonso', planted_year: '' }

const inputClass =
  'w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2.5 text-white focus:outline-none focus:border-emerald-500'

function Field({ label, children }) {
  return (
    <label className="space-y-1.5 block">
      <span className="text-slate-300 font-semibold block">{label}</span>
      {children}
    </label>
  )
}

export default function Admin() {
  const { farm, refresh, selectFarm } = useFarm()
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState(EMPTY_FARM)
  const [saving, setSaving] = useState(false)
  const [trees, setTrees] = useState([])
  const [treeForm, setTreeForm] = useState(EMPTY_TREE)

  useEffect(() => {
    if (creating || !farm) {
      setForm(EMPTY_FARM)
      setTrees([])
      return
    }
    setForm({
      name: farm.name,
      cultivar: farm.cultivar,
      acreage: farm.acreage,
      tree_count: farm.tree_count,
      latitude: farm.latitude,
      longitude: farm.longitude,
    })
    getFarmTrees(farm.id).then((res) => setTrees(res.data)).catch(() => setTrees([]))
  }, [farm, creating])

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))
  const setTree = (key) => (e) => setTreeForm((t) => ({ ...t, [key]: e.target.value }))

  const handleSave = async (e) => {
    e.preventDefault()
    const payload = {
      name: form.name.trim(),
      cultivar: form.cultivar,
      acreage: Number(form.acreage),
      tree_count: parseInt(form.tree_count, 10),
      latitude: Number(form.latitude),
      longitude: Number(form.longitude),
    }
    setSaving(true)
    try {
      if (creating || !farm) {
        const res = await createFarm(payload)
        await refresh()
        selectFarm(res.data.id)
        setCreating(false)
        toast.success('Orchard registered.')
      } else {
        await updateFarm(farm.id, payload)
        await refresh()
        toast.success('Orchard settings saved.')
      }
    } catch (err) {
      toast.error(errorMessage(err, 'Could not save orchard'))
    } finally {
      setSaving(false)
    }
  }

  const handleAddTree = async (e) => {
    e.preventDefault()
    try {
      const res = await addTree(farm.id, {
        tag_code: treeForm.tag_code.trim(),
        row_number: parseInt(treeForm.row_number, 10),
        tree_number: parseInt(treeForm.tree_number, 10),
        variety: treeForm.variety,
        planted_year: treeForm.planted_year ? parseInt(treeForm.planted_year, 10) : null,
      })
      setTrees((t) => [...t, res.data])
      setTreeForm(EMPTY_TREE)
      toast.success('Tree registered.')
    } catch (err) {
      toast.error(errorMessage(err, 'Could not add tree'))
    }
  }

  const editingNew = creating || !farm

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Farm & Orchard Administration</h2>
          <p className="text-sm text-slate-400 mt-1">
            Orchard metadata and coordinates (used for Open-Meteo weather) and the tagged tree registry.
          </p>
        </div>
        {farm && (
          <button
            type="button"
            onClick={() => setCreating(!creating)}
            className="text-xs text-emerald-400 hover:text-emerald-300 border border-emerald-500/30 rounded-lg px-3 py-1.5 shrink-0"
          >
            {creating ? 'Cancel' : '+ New orchard'}
          </button>
        )}
      </div>

      <form onSubmit={handleSave} className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-6">
        <h3 className="text-sm font-bold text-white uppercase tracking-wider border-b border-slate-800 pb-3">
          {editingNew ? 'Register Orchard' : 'Orchard Metadata'}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <Field label="Orchard Name">
            <input required maxLength={120} value={form.name} onChange={set('name')} className={inputClass} />
          </Field>
          <Field label="Primary Cultivar">
            <select value={form.cultivar} onChange={set('cultivar')} className={inputClass}>
              {[...new Set([form.cultivar, ...CULTIVARS])].filter(Boolean).map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>
          <Field label="Total Acreage">
            <input required type="number" min="0" step="0.1" value={form.acreage} onChange={set('acreage')} className={inputClass} />
          </Field>
          <Field label="Tree Inventory Count">
            <input required type="number" min="0" step="1" value={form.tree_count} onChange={set('tree_count')} className={inputClass} />
          </Field>
          <Field label="Latitude (-90 to 90)">
            <input required type="number" min="-90" max="90" step="any" value={form.latitude} onChange={set('latitude')}
              className={`${inputClass} font-mono`} />
          </Field>
          <Field label="Longitude (-180 to 180)">
            <input required type="number" min="-180" max="180" step="any" value={form.longitude} onChange={set('longitude')}
              className={`${inputClass} font-mono`} />
          </Field>
        </div>
        <div className="pt-4 border-t border-slate-800 flex justify-end">
          <button
            type="submit"
            disabled={saving}
            className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-lg shadow-emerald-600/20 transition-all disabled:opacity-50"
          >
            {saving ? 'Saving...' : editingNew ? 'Register Orchard' : 'Save Farm Settings'}
          </button>
        </div>
      </form>

      {!editingNew && (
        <div className="glass-panel rounded-2xl p-6 border-slate-700/60 space-y-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider border-b border-slate-800 pb-3">
            Tagged Trees ({trees.length})
          </h3>
          {trees.length > 0 && (
            <table className="w-full text-xs text-left">
              <thead className="text-slate-400 uppercase">
                <tr><th className="py-2">Tag</th><th>Row</th><th>Tree</th><th>Variety</th><th>Planted</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {trees.map((t) => (
                  <tr key={t.id}>
                    <td className="py-2 font-mono">{t.tag_code}</td><td>{t.row_number}</td><td>{t.tree_number}</td>
                    <td>{t.variety}</td><td>{t.planted_year ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <form onSubmit={handleAddTree} className="grid grid-cols-2 md:grid-cols-6 gap-3 text-xs items-end">
            <Field label="Tag code"><input required maxLength={50} value={treeForm.tag_code} onChange={setTree('tag_code')} className={inputClass} /></Field>
            <Field label="Row"><input required type="number" min="0" value={treeForm.row_number} onChange={setTree('row_number')} className={inputClass} /></Field>
            <Field label="Tree #"><input required type="number" min="0" value={treeForm.tree_number} onChange={setTree('tree_number')} className={inputClass} /></Field>
            <Field label="Variety"><input maxLength={100} value={treeForm.variety} onChange={setTree('variety')} className={inputClass} /></Field>
            <Field label="Planted"><input type="number" min="1800" max="2100" value={treeForm.planted_year} onChange={setTree('planted_year')} className={inputClass} /></Field>
            <button type="submit" className="px-4 py-2.5 rounded-xl bg-slate-700 hover:bg-slate-600 text-white font-medium">Add tree</button>
          </form>
        </div>
      )}
    </div>
  )
}
