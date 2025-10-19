export interface Job {
  job_id: string
  status: 'created' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  job_name?: string
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
  }
}

export interface ProteinSequenceInput {
  sequence: string
  job_name?: string
  num_designs: number
}
