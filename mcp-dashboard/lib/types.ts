export interface Job {
  job_id: string
  status: 'created' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  job_name?: string
  input?: {
    sequence?: string
    num_designs?: number
  }
  progress: {
    alphafold: string
    rfdiffusion: string
    proteinmpnn: string
    alphafold_multimer: string
  }
  results?: {
    target_structure: any
    designs: Design[]
  }
  error?: string
}

export interface Design {
  design_id: number
  backbone: any
  sequence: any
  complex_structure: any
}

export interface ServiceStatus {
  [service: string]: {
    status: string
    url: string
    error?: string
    reason?: string
    http_status?: number
    backend?: string
    selected_provider?: string
  }
}

export interface ProteinSequenceInput {
  sequence: string
  job_name?: string
  num_designs: number
}
export interface AlphaFoldSettings {
  speed_preset?: 'fast' | 'balanced' | 'quality'
  disable_templates?: boolean
  num_recycles?: number
  num_ensemble?: number
  mmseqs2_max_seqs?: number
  msa_mode?: 'jackhmmer' | 'mmseqs2'
}