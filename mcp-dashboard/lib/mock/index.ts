import type { Job } from '../types'

type Tool = {
  name: string
  description?: string
  inputSchema?: {
    type?: string
    properties?: Record<string, any>
    required?: string[]
  }
}

type MockState = {
  nextJobId: number
  jobs: Map<string, Job>
  runtimeConfig: Record<string, any>
  alphafoldSettings: Record<string, any>
}

const DEFAULT_ALPHAFOLD_SETTINGS = {
  speed_preset: 'balanced',
  disable_templates: false,
  num_recycles: 3,
  num_ensemble: 1,
  mmseqs2_max_seqs: 128,
  msa_mode: 'mmseqs2',
}

const DEFAULT_RUNTIME_CONFIG = {
  providers: {
    rfdiffusion: 'mock',
    proteinmpnn: 'mock',
    alphafold: 'mock',
  },
  mock_mode: true,
}

const examplePdb = `ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.400   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.040  14.840   2.100  1.00 20.00           C
ATOM      4  O   ALA A   1      12.370  15.800   2.100  1.00 20.00           O
ATOM      5  N   GLY A   2      14.330  14.980   2.100  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.930  16.320   2.100  1.00 20.00           C
ATOM      7  C   GLY A   2      16.420  16.370   2.100  1.00 20.00           C
ATOM      8  O   GLY A   2      17.020  15.310   2.100  1.00 20.00           O
TER
END
`

function getGlobalState(): MockState {
  const g = globalThis as any
  if (!g.__MCP_DASHBOARD_MOCK_STATE) {
    const state: MockState = {
      nextJobId: 1,
      jobs: new Map(),
      runtimeConfig: { ...DEFAULT_RUNTIME_CONFIG },
      alphafoldSettings: { ...DEFAULT_ALPHAFOLD_SETTINGS },
    }

    // Seed with an example completed job so the "design library" isn't empty.
    const seedJob = createCompletedJob(state, {
      sequence: 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKVT',
      job_name: 'Mock Example Job',
      num_designs: 3,
    })
    state.jobs.set(seedJob.job_id, seedJob)

    g.__MCP_DASHBOARD_MOCK_STATE = state
  }

  return g.__MCP_DASHBOARD_MOCK_STATE as MockState
}

export function isMockMode(): boolean {
  const v = process.env.MCP_DASHBOARD_MOCK || process.env.NEXT_PUBLIC_MCP_DASHBOARD_MOCK
  return v === '1' || v === 'true' || v === 'yes'
}

export const mockTools: Tool[] = [
  {
    name: 'check_services',
    description: 'Return status for all backend services',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'list_jobs',
    description: 'List all protein design jobs',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'get_job_status',
    description: 'Get status and results for a specific job',
    inputSchema: {
      type: 'object',
      properties: { job_id: { type: 'string', description: 'Job ID' } },
      required: ['job_id'],
    },
  },
  {
    name: 'delete_job',
    description: 'Delete a job by ID',
    inputSchema: {
      type: 'object',
      properties: { job_id: { type: 'string', description: 'Job ID' } },
      required: ['job_id'],
    },
  },
  {
    name: 'design_protein_binder',
    description: 'Design binder sequences for a target sequence',
    inputSchema: {
      type: 'object',
      properties: {
        sequence: { type: 'string', description: 'Target protein amino acid sequence' },
        job_name: { type: 'string', description: 'Optional job name' },
        num_designs: { type: 'integer', description: 'Number of designs', default: 5 },
      },
      required: ['sequence'],
    },
  },
  {
    name: 'get_runtime_config',
    description: 'Get server runtime routing/provider configuration',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'update_runtime_config',
    description: 'Update server runtime routing/provider configuration',
    inputSchema: {
      type: 'object',
      properties: { patch: { type: 'object', description: 'Config patch', default: {} } },
      required: ['patch'],
    },
  },
  {
    name: 'reset_runtime_config',
    description: 'Reset server runtime configuration to defaults',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'embedded_bootstrap',
    description: 'Best-effort embedded asset bootstrap/download into /models',
    inputSchema: {
      type: 'object',
      properties: { models: { type: 'array', items: { type: 'string' }, default: [] } },
      required: [],
    },
  },
  {
    name: 'get_alphafold_settings',
    description: 'Get current AlphaFold optimization settings',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'update_alphafold_settings',
    description: 'Update AlphaFold optimization settings',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'reset_alphafold_settings',
    description: 'Reset AlphaFold optimization settings to defaults',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'predict_structure',
    description: 'Predict structure from sequence',
    inputSchema: {
      type: 'object',
      properties: { sequence: { type: 'string' } },
      required: ['sequence'],
    },
  },
  {
    name: 'predict_complex',
    description: 'Predict complex structure from chain sequences',
    inputSchema: {
      type: 'object',
      properties: { sequences: { type: 'array', items: { type: 'string' } } },
      required: ['sequences'],
    },
  },
  {
    name: 'generate_sequence',
    description: 'Generate binder sequence from a backbone PDB',
    inputSchema: {
      type: 'object',
      properties: { backbone_pdb: { type: 'string' } },
      required: ['backbone_pdb'],
    },
  },
  {
    name: 'propose_sequence_variants',
    description:
      'Propose derivative sequences by mutating selected 1-based positions (mock-only utility for UI iteration flows)',
    inputSchema: {
      type: 'object',
      properties: {
        sequence: { type: 'string', description: 'Base amino acid sequence' },
        positions: {
          type: 'array',
          items: { type: 'integer' },
          description: '1-based positions to consider for mutation',
          default: [],
        },
        num_variants: { type: 'integer', description: 'Number of variants to propose', default: 5 },
      },
      required: ['sequence', 'positions'],
    },
  },
]

function nowIso() {
  return new Date().toISOString()
}

function stableBinderSequences() {
  return [
    'GSSHHHHHHSSGLVPRGSHMKWVTFISLLFLFSSAYSRGVFRRDGQGSSS',
    'MSTKKKNNNNGASGSGEIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASTRATGIPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYNSYPYTFGQGTKLEIK',
    'MEFGLKKLNVGDDVVAVVEPQNNLTTTKSYVAFDKNTDVEGKMVTVEEDGEEVVRR',
    'MKKTAIAIAVALAGFATVAQAAPVYVNATNSSRGVNTTVKLLDNNKDLTLEK',
    'MSDSESKQITVQGKKKTKTTTAAAPVNVGSKKGGGG',
  ]
}

function clampInt(v: any, def: number, min: number, max: number) {
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return def
  return Math.max(min, Math.min(max, Math.floor(n)))
}

function createCompletedJob(
  state: MockState,
  input: { sequence: string; job_name?: string; num_designs?: number }
): Job {
  const id = `mock_job_${String(state.nextJobId).padStart(4, '0')}`
  state.nextJobId += 1

  const created = nowIso()
  const sequences = stableBinderSequences()
  const count = clampInt(input.num_designs, 5, 1, 20)

  return {
    job_id: id,
    status: 'completed',
    created_at: created,
    updated_at: created,
    job_name: input.job_name,
    input: {
      sequence: input.sequence,
      num_designs: count,
    },
    progress: {
      alphafold: 'completed',
      rfdiffusion: 'completed',
      proteinmpnn: 'completed',
      alphafold_multimer: 'completed',
    },
    results: {
      target_structure: { pdb: examplePdb },
      designs: Array.from({ length: count }).map((_, idx) => ({
        design_id: idx,
        backbone: { pdb: examplePdb },
        sequence: sequences[idx % sequences.length],
        complex_structure: { pdb: examplePdb },
      })),
    },
  }
}

function toolTextResult(payload: any) {
  return {
    content: [
      {
        type: 'text',
        text: JSON.stringify(payload),
      },
    ],
  }
}

function clampPositions(positions: any, maxLen: number): number[] {
  if (!Array.isArray(positions)) return []
  const out: number[] = []
  for (const v of positions) {
    const n = typeof v === 'number' ? v : Number(v)
    if (!Number.isFinite(n)) continue
    const i = Math.floor(n)
    if (i < 1 || i > maxLen) continue
    out.push(i)
  }
  // Unique + stable order
  return Array.from(new Set(out)).sort((a, b) => a - b)
}

function hashScore(input: string): number {
  // Deterministic non-crypto hash -> [0, 1)
  let h = 2166136261
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  // Convert to unsigned
  const u = h >>> 0
  return (u % 1_000_000) / 1_000_000
}

function scoreFromSequence(seq: string): number {
  // Map to an affinity-like score range.
  const x = hashScore(seq)
  const min = 0.65
  const max = 0.98
  return min + x * (max - min)
}

function mutateAtPositions(sequence: string, positions: number[], variantIndex: number): string {
  const alphabet = 'ACDEFGHIKLMNPQRSTVWY'
  const chars = sequence.split('')
  for (let i = 0; i < positions.length; i++) {
    const pos = positions[i] - 1
    const current = chars[pos]
    const base = (variantIndex + i * 7 + current.charCodeAt(0)) % alphabet.length
    let next = alphabet[base]
    if (next === current) {
      next = alphabet[(base + 3) % alphabet.length]
    }
    chars[pos] = next
  }
  return chars.join('')
}

function toolError(message: string) {
  return {
    content: [
      {
        type: 'text',
        text: message,
      },
    ],
    isError: true,
  }
}

export function listMockJobs(): Job[] {
  const state = getGlobalState()
  return Array.from(state.jobs.values()).sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  )
}

export function getMockJob(jobId: string): Job | null {
  const state = getGlobalState()
  return state.jobs.get(jobId) || null
}

export function deleteMockJob(jobId: string): boolean {
  const state = getGlobalState()
  return state.jobs.delete(jobId)
}

export function createMockJob(input: { sequence: string; job_name?: string; num_designs?: number }): Job {
  const state = getGlobalState()
  const job = createCompletedJob(state, input)
  state.jobs.set(job.job_id, job)
  return job
}

export function handleMockToolCall(name: string, args: Record<string, any>) {
  const state = getGlobalState()

  switch (name) {
    case 'check_services': {
      return toolTextResult({
        proteinmpnn: { status: 'ready', url: 'mock://proteinmpnn', backend: 'mock' },
        rfdiffusion: { status: 'ready', url: 'mock://rfdiffusion', backend: 'mock' },
        alphafold: { status: 'ready', url: 'mock://alphafold', backend: 'mock' },
        alphafold_multimer: { status: 'ready', url: 'mock://alphafold-multimer', backend: 'mock' },
        mmseqs2: { status: 'ready', url: 'mock://mmseqs2', backend: 'mock' },
      })
    }

    case 'list_jobs': {
      return toolTextResult(listMockJobs())
    }

    case 'get_job_status': {
      const jobId = args?.job_id
      if (!jobId || typeof jobId !== 'string') return toolError('Missing job_id')
      const job = getMockJob(jobId)
      if (!job) return toolError(`Unknown job_id: ${jobId}`)
      return toolTextResult(job)
    }

    case 'delete_job': {
      const jobId = args?.job_id
      if (!jobId || typeof jobId !== 'string') return toolError('Missing job_id')
      const deleted = deleteMockJob(jobId)
      return toolTextResult({ success: true, deleted: deleted ? jobId : null })
    }

    case 'design_protein_binder': {
      const sequence = args?.sequence
      if (!sequence || typeof sequence !== 'string') return toolError('Missing sequence')
      const job = createMockJob({
        sequence,
        job_name: typeof args?.job_name === 'string' ? args.job_name : undefined,
        num_designs: typeof args?.num_designs === 'number' ? args.num_designs : undefined,
      })
      return toolTextResult(job)
    }

    case 'get_runtime_config': {
      return toolTextResult(state.runtimeConfig)
    }

    case 'update_runtime_config': {
      const patch = args?.patch
      if (patch && typeof patch === 'object') {
        state.runtimeConfig = { ...state.runtimeConfig, ...patch }
      }
      return toolTextResult(state.runtimeConfig)
    }

    case 'reset_runtime_config': {
      state.runtimeConfig = { ...DEFAULT_RUNTIME_CONFIG }
      return toolTextResult(state.runtimeConfig)
    }

    case 'embedded_bootstrap': {
      const models = Array.isArray(args?.models) ? args.models : []
      return toolTextResult({ ok: true, bootstrapped: models })
    }

    case 'get_alphafold_settings': {
      // This tool returns only the settings object (AlphaFoldSettings.tsx expects that).
      return toolTextResult(state.alphafoldSettings)
    }

    case 'update_alphafold_settings': {
      if (args && typeof args === 'object') {
        state.alphafoldSettings = { ...state.alphafoldSettings, ...args }
      }
      return toolTextResult({ success: true, settings: state.alphafoldSettings })
    }

    case 'reset_alphafold_settings': {
      state.alphafoldSettings = { ...DEFAULT_ALPHAFOLD_SETTINGS }
      return toolTextResult({ success: true, settings: state.alphafoldSettings })
    }

    case 'predict_structure': {
      return toolTextResult({ pdb: examplePdb })
    }

    case 'predict_complex': {
      return toolTextResult({ pdb: examplePdb })
    }

    case 'generate_sequence': {
      const seq = stableBinderSequences()[0]
      return toolTextResult({ sequence: seq })
    }

    case 'propose_sequence_variants': {
      const sequence = args?.sequence
      if (!sequence || typeof sequence !== 'string') return toolError('Missing sequence')
      const positions = clampPositions(args?.positions, sequence.length)
      if (positions.length === 0) return toolError('Missing or invalid positions')

      const num = clampInt(args?.num_variants, 5, 1, 20)
      const variants = Array.from({ length: num }).map((_, idx) => {
        const mutated = mutateAtPositions(sequence, positions, idx)
        return {
          sequence: mutated,
          positions,
          score: Number(scoreFromSequence(mutated).toFixed(4)),
        }
      })

      // Best-first for convenience.
      variants.sort((a, b) => b.score - a.score)
      return toolTextResult({ base_sequence: sequence, positions, variants })
    }

    default:
      return toolError(`Unknown tool: ${name}`)
  }
}

export function mockListResources() {
  return {
    resources: [
      {
        uri: 'mock://readme',
        name: 'Mock Mode Overview',
        mimeType: 'text/plain',
      },
      {
        uri: 'mock://example.pdb',
        name: 'Example PDB',
        mimeType: 'text/plain',
      },
    ],
  }
}

export function mockReadResource(uri: string) {
  if (uri === 'mock://example.pdb') {
    return {
      contents: [
        {
          uri,
          mimeType: 'text/plain',
          text: examplePdb,
        },
      ],
    }
  }

  return {
    contents: [
      {
        uri,
        mimeType: 'text/plain',
        text:
          'Dashboard is running in MCP_DASHBOARD_MOCK mode.\n\n' +
          '- Tool calls are handled locally and return deterministic payloads.\n' +
          '- Jobs are stored in-memory (process scope).\n' +
          '- Use this mode to validate UI flows, visualizations, and library management without GPU/services.\n',
      },
    ],
  }
}
