# AlphaFold Runtime Profiling

This guide explains how to profile AlphaFold runs for GPU utilization, CPU/memory, and disk I/O, and outlines optimization strategies to improve throughput.

## Quick Start

Use the profiling wrapper to run AlphaFold while collecting system metrics:

```bash
scripts/run_profiled_inference.sh /path/to/query.fasta mmseqs2 /tmp/af_profile_run
```

- Outputs are written under the chosen directory:
  - `run.log` — AlphaFold logs
  - `gpu_util.csv` — GPU utilization, clocks, power, memory via `nvidia-smi`
  - `gpu_dmon.csv` — per-GPU detailed counters via `nvidia-smi dmon`
  - `pidstat.csv` — CPU/memory/IO per process
  - `iostat.csv` — disk I/O stats
  - `summary.txt` — run summary with elapsed time

To compare JackHMMER vs MMseqs2:

```bash
scripts/bench_msa_comparison.sh
```

Ensure `.env` is configured and the MMseqs2 DB path is set to the desired prefix.

## What to Look For

- GPU utilization consistency: aim for high, steady utilization during model inference.
- GPU memory headroom: avoid excessive fragmentation or overallocation that causes eviction or OOM.
- CPU saturation: make sure MSA and template search threads use available cores without starving the GPU feeder.
- Disk I/O bursts: large indexes can cause I/O contention; consider SSD/NVMe and adequate temp directories.

## Optimization Strategies (Actionable)

- MSA generation:
  - Prefer MMseqs2 with indexed DB and tuned `--mmseqs2_max_seqs`.
  - Use split-indexing (e.g., `--split-memory-limit 80–96G`) to fit RAM.
  - Cache MSAs per sequence and enable precomputed MSAs for repeated runs.
- Templates:
  - Disable templates when not needed (`--model_preset=monomer`, `--max_template_date`, or omit template dirs).
  - Ensure HHsearch threads match CPU cores; avoid oversubscription.
- Model inference:
  - Warm-up JIT/graph compilation by running a single dummy forward to cache shapes.
  - Pin CPU threads: set `OMP_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` to sensible values.
  - Reduce recycles/ensembles for speed-sensitive runs.
- System:
  - Keep NVMe temp dirs for MMseqs2 indexes.
  - Monitor and adjust GPU clocks/power limits if allowed (ensure thermal headroom).

## Next Steps

- Integrate lightweight stage timers (MSA, template search, feature assembly, model) for precise attribution.
- Add optional NVML-based in-process probes for synchronized utilization logging.
- Automate summary reports aggregating metrics across runs.
