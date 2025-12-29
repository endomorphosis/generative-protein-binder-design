export type DesignLibraryItem = {
  id: string
  sequence: string
  score?: number
  positions?: number[]
  source?: string
  pdbData?: string
  createdAt: string
}

const STORAGE_KEY = 'mcp-dashboard:design-library:v1'

function safeParse(text: string | null): any {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function canUseStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
}

export function loadDesignLibrary(): DesignLibraryItem[] {
  if (!canUseStorage()) return []
  const raw = safeParse(window.localStorage.getItem(STORAGE_KEY))
  if (!Array.isArray(raw)) return []
  return raw
    .filter((x) => x && typeof x === 'object' && typeof x.sequence === 'string')
    .map((x) => ({
      id: typeof x.id === 'string' ? x.id : crypto.randomUUID(),
      sequence: String(x.sequence),
      score: typeof x.score === 'number' ? x.score : undefined,
      positions: Array.isArray(x.positions) ? x.positions.map((n: any) => Number(n)).filter((n: number) => Number.isFinite(n)) : undefined,
      source: typeof x.source === 'string' ? x.source : undefined,
      pdbData: typeof x.pdbData === 'string' ? x.pdbData : undefined,
      createdAt: typeof x.createdAt === 'string' ? x.createdAt : new Date().toISOString(),
    }))
}

export function saveDesignLibrary(items: DesignLibraryItem[]) {
  if (!canUseStorage()) return
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
  window.dispatchEvent(new Event('design-library-updated'))
}

export function addToDesignLibrary(item: Omit<DesignLibraryItem, 'id' | 'createdAt'> & { id?: string; createdAt?: string }) {
  const items = loadDesignLibrary()
  const next: DesignLibraryItem = {
    id: item.id || (typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`),
    sequence: item.sequence,
    score: item.score,
    positions: item.positions,
    source: item.source,
    pdbData: item.pdbData,
    createdAt: item.createdAt || new Date().toISOString(),
  }

  // De-dupe by exact sequence (keep newest at top)
  const filtered = items.filter((x) => x.sequence !== next.sequence)
  saveDesignLibrary([next, ...filtered])
}

export function removeFromDesignLibrary(id: string) {
  const items = loadDesignLibrary()
  saveDesignLibrary(items.filter((x) => x.id !== id))
}
