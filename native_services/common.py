from __future__ import annotations

import contextlib
import fcntl
import os
import math
import mmap
import subprocess
import time
from pathlib import Path
from typing import Optional

import ctypes


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def require_env(name: str) -> str:
    val = env_str(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def mem_available_bytes() -> Optional[int]:
    """Best-effort available RAM in bytes (Linux: /proc/meminfo MemAvailable)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        # kB
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def meminfo_snapshot_bytes() -> dict[str, int]:
    """Best-effort snapshot of key /proc/meminfo counters (bytes).

    Intended for user-facing diagnostics. All values are best-effort and may be 0
    if unavailable.
    """

    keys = {
        "MemTotal": 0,
        "MemAvailable": 0,
        "Cached": 0,
        "Buffers": 0,
        "SwapTotal": 0,
        "SwapFree": 0,
    }
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                for k in list(keys.keys()):
                    if line.startswith(k + ":"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            # kB
                            keys[k] = int(parts[1]) * 1024
    except Exception:
        pass
    return keys


def _format_gib(x: int) -> float:
    try:
        return round(x / (1024**3), 3)
    except Exception:
        return 0.0


def meminfo_snapshot() -> dict[str, object]:
    snap_b = meminfo_snapshot_bytes()
    return {
        "bytes": snap_b,
        "gib": {k: _format_gib(v) for k, v in snap_b.items()},
    }


def alphafold_db_paths(*, data_dir: str, tier: str = "auto") -> list[str]:
    """Return a conservative list of the largest AlphaFold DB files.

    This is intentionally not exhaustive (e.g. avoids mmCIF directory) to keep
    metrics/eviction cheap and safe.
    """

    d = Path(data_dir).expanduser()
    if tier == "auto":
        if (d / "uniref90" / "uniref90.fasta").exists() and (d / "small_bfd" / "bfd-first_non_consensus_sequences.fasta").exists():
            tier = "reduced"
        else:
            tier = "full"

    candidates: list[Path] = [
        d / "uniref90" / "uniref90.fasta",
        d / "mgnify" / "mgy_clusters_2022_05.fa",
        d / "small_bfd" / "bfd-first_non_consensus_sequences.fasta",
        d / "pdb70" / "pdb70",
        d / "pdb_seqres" / "pdb_seqres.txt",
        d / "uniprot" / "uniprot.fasta",
    ]
    if tier == "full":
        candidates += [
            d / "bfd" / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
            d / "uniref30" / "UniRef30_2021_03",
        ]

    out: list[str] = []
    for p in candidates:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                out.append(str(p))
        except Exception:
            continue
    return out


def _mincore_resident_byte(mm: mmap.mmap) -> bool:
    # Linux mincore(2) reports whether pages are resident in RAM.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    vec = (ctypes.c_ubyte * 1)()
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(len(mm)), ctypes.byref(vec))
    if rc != 0:
        return False
    return bool(vec[0] & 1)


def estimate_file_cache_residency(path: str, *, max_samples: int = 256) -> dict[str, object]:
    """Estimate how much of a file is resident in page cache.

    Uses mincore sampling so it does not read the file and does not warm the cache.
    Returns an approximate percent (0..100) plus metadata.
    """

    p = Path(path)
    try:
        size = p.stat().st_size
    except Exception:
        return {"path": path, "ok": False, "error": "stat_failed"}

    if size <= 0:
        return {"path": path, "ok": False, "error": "empty"}

    page = mmap.PAGESIZE
    pages = max(1, (size + page - 1) // page)
    samples = max(1, min(int(max_samples), int(pages)))
    stride = max(1, pages // samples)

    resident = 0
    checked = 0
    fd: Optional[int] = None
    try:
        fd = os.open(str(p), os.O_RDONLY)
        for i in range(0, pages, stride):
            offset = int(i * page)
            length = min(page, size - offset)
            if length <= 0:
                break
            try:
                mm = mmap.mmap(fd, length=page, access=mmap.ACCESS_COPY, offset=offset)
            except Exception:
                # Some filesystems may reject mapping at arbitrary offsets.
                continue
            try:
                if _mincore_resident_byte(mm):
                    resident += 1
                checked += 1
            finally:
                try:
                    mm.close()
                except Exception:
                    pass
            if checked >= samples:
                break
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass

    if checked <= 0:
        return {"path": path, "ok": False, "error": "no_samples"}

    pct = (resident / checked) * 100.0
    return {
        "path": path,
        "ok": True,
        "sample_pages": checked,
        "resident_samples": resident,
        "resident_pct": round(pct, 1),
        "size_bytes": size,
    }


@contextlib.contextmanager
def _exclusive_lock(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _wait_for_memory(*, min_free_bytes: int, wait_seconds: int, interval_seconds: int) -> None:
    """Block until enough RAM is available, or raise RuntimeError.

    This prevents starting a large model stage when the host is already under
    memory pressure.
    """

    start = time.time()
    while True:
        avail = mem_available_bytes()
        if avail is None:
            # Can't measure reliably; don't block.
            return
        if avail >= min_free_bytes:
            return
        if time.time() - start > wait_seconds:
            raise RuntimeError(
                f"Insufficient free RAM to start model run: need >= {min_free_bytes/1024**3:.1f}GiB, "
                f"have {avail/1024**3:.1f}GiB (MemAvailable)."
            )
        time.sleep(max(1, interval_seconds))


def run_cmd(cmd: str, *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    # Use bash -lc so users can rely on conda init / module loads, etc.
    # This also allows users to provide complex pipelines.
    #
    # IMPORTANT: These native services run under a Python venv (for FastAPI/uvicorn),
    # but the model commands often need to run under conda envs or system installs.
    # If we leak VIRTUAL_ENV and the venv's bin/ prefix into the child process,
    # some tools (notably `conda run`) may end up selecting the wrong python.
    child_env = os.environ.copy()
    venv = child_env.pop("VIRTUAL_ENV", None)
    if venv:
        path = child_env.get("PATH", "")
        venv_bin_prefix = f"{venv}/bin:"
        if path.startswith(venv_bin_prefix):
            child_env["PATH"] = path[len(venv_bin_prefix) :]

    # Safety defaults for JAX on GPU: avoid aggressive preallocation that can
    # OOM the system. Users can override by exporting these env vars.
    child_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    child_env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    # Hint MMseqs2 to use GPU when requested and available (no-op if binary lacks GPU).
    if _truthy_env("ALPHAFOLD_MMSEQS2_USE_GPU"):
        child_env.setdefault("MMSEQS_FORCEGPU", "1")

    # Optional safety rails:
    # - single-flight lock to prevent multiple concurrent heavy runs
    # - RAM preflight to avoid starting under low-memory conditions
    use_lock = _truthy_env("MODEL_RUN_USE_LOCK") or (os.getenv("MODEL_RUN_USE_LOCK") is None)
    lock_path = env_str("MODEL_RUN_LOCK_PATH", "/tmp/mcp_model_run.lock") or "/tmp/mcp_model_run.lock"
    min_mem_gb = float(env_str("MODEL_RUN_MIN_MEM_GB", "8") or "8")
    wait_seconds = int(env_str("MODEL_RUN_WAIT_SECONDS", "600") or "600")
    interval_seconds = int(env_str("MODEL_RUN_WAIT_INTERVAL_SECONDS", "10") or "10")

    def _run() -> subprocess.CompletedProcess[str]:
        if min_mem_gb > 0:
            _wait_for_memory(
                min_free_bytes=int(min_mem_gb * 1024**3),
                wait_seconds=wait_seconds,
                interval_seconds=interval_seconds,
            )
        return subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=child_env,
        )

    if use_lock:
        with _exclusive_lock(lock_path):
            return _run()

    return _run()


def available_cpu_count() -> int:
    """Best-effort CPU count available to this process.

    - Respects Linux CPU affinity (sched_getaffinity)
    - Respects cgroup CPU quota when detectable (v1/v2)
    """

    affinity_count: Optional[int] = None
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except Exception:
        affinity_count = None

    os_count = os.cpu_count() or 1

    # cgroup v2: /sys/fs/cgroup/cpu.max -> "<quota> <period>" (or "max <period>")
    quota_count: Optional[int] = None
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            parts = cpu_max.read_text(encoding="utf-8").strip().split()
            if len(parts) >= 2 and parts[0] != "max":
                quota_us = int(parts[0])
                period_us = int(parts[1])
                if quota_us > 0 and period_us > 0:
                    quota_count = max(1, int(math.ceil(quota_us / period_us)))
    except Exception:
        quota_count = None

    # cgroup v1 fallback.
    if quota_count is None:
        try:
            quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            if quota_path.exists() and period_path.exists():
                quota_us = int(quota_path.read_text(encoding="utf-8").strip())
                period_us = int(period_path.read_text(encoding="utf-8").strip())
                if quota_us > 0 and period_us > 0:
                    quota_count = max(1, int(math.ceil(quota_us / period_us)))
        except Exception:
            quota_count = None

    counts = [os_count]
    if affinity_count:
        counts.append(affinity_count)
    if quota_count:
        counts.append(quota_count)

    return max(1, min(counts))


def nvidia_gpu_present() -> bool:
    # Avoid invoking nvidia-smi (can be slow / absent); just check device nodes.
    try:
        if Path("/dev/nvidiactl").exists():
            return True
        if any(Path("/dev").glob("nvidia[0-9]*")):
            return True
        if Path("/proc/driver/nvidia/gpus").exists():
            return True
    except Exception:
        return False
    return False


def read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def ensure_file_exists(path: str, *, label: str) -> None:
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
        raise RuntimeError(f"{label} not found or empty: {path}")
