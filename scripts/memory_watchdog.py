#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


POSIX_FADV_DONTNEED = 4


def _meminfo_kb(key: str) -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except Exception:
        return 0
    return 0


def mem_available_bytes() -> int:
    return _meminfo_kb("MemAvailable") * 1024


def swap_free_bytes() -> int:
    return _meminfo_kb("SwapFree") * 1024


def swap_total_bytes() -> int:
    return _meminfo_kb("SwapTotal") * 1024


def fmt_gib(n: int) -> str:
    return f"{n / (1024**3):.1f} GiB"


def iter_default_paths(data_dir: Path, tier: str) -> list[Path]:
    paths: list[Path] = []

    # Reduced tier (the common fast/default path).
    paths += [
        data_dir / "uniref90" / "uniref90.fasta",
        data_dir / "mgnify" / "mgy_clusters_2022_05.fa",
        data_dir / "small_bfd" / "bfd-first_non_consensus_sequences.fasta",
        data_dir / "pdb70" / "pdb70",
        # Multimer extras if present.
        data_dir / "pdb_seqres" / "pdb_seqres.txt",
        data_dir / "uniprot" / "uniprot.fasta",
    ]

    if tier == "full":
        paths += [
            data_dir / "bfd" / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
            data_dir / "uniref30" / "UniRef30_2021_03",
        ]

    # Filter to existing regular files only.
    out: list[Path] = []
    for p in paths:
        try:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                out.append(p)
        except Exception:
            continue
    return out


def _try_vmtouch_evict(paths: Iterable[Path]) -> bool:
    vmtouch = shutil.which("vmtouch")
    if not vmtouch:
        return False

    # vmtouch -e: evict pages from the file-backed cache (best-effort).
    # Use one invocation to reduce process overhead.
    args = [vmtouch, "-e"] + [str(p) for p in paths]
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def _posix_fadvise_dontneed(path: Path) -> None:
    # Best-effort. Not all filesystems honor this fully.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd: Optional[int] = None
    try:
        fd = os.open(str(path), os.O_RDONLY)
        # length=0 means "to end of file" on Linux.
        libc.posix_fadvise(ctypes.c_int(fd), ctypes.c_long(0), ctypes.c_long(0), ctypes.c_int(POSIX_FADV_DONTNEED))
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def evict_paths(paths: list[Path]) -> int:
    if not paths:
        return 0

    if _try_vmtouch_evict(paths):
        return len(paths)

    count = 0
    for p in paths:
        try:
            _posix_fadvise_dontneed(p)
            count += 1
        except Exception:
            continue
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Continuously monitor memory pressure and evict known AlphaFold DB files from Linux page cache. "
            "This is safe for non-technical users: it does not lock RAM and does not drop global caches."
        )
    )
    ap.add_argument("--data-dir", default=os.getenv("ALPHAFOLD_DATA_DIR") or str(Path.home() / ".cache/alphafold"))
    ap.add_argument("--tier", choices=["auto", "reduced", "full"], default="auto")
    ap.add_argument("--interval-seconds", type=int, default=5)
    ap.add_argument("--low-mem-gb", type=float, default=float(os.getenv("MCP_MEM_WATCHDOG_LOW_GB") or "6"))
    ap.add_argument("--cooldown-seconds", type=int, default=30)
    ap.add_argument("--once", action="store_true", help="Run a single check/evict pass then exit")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    tier = args.tier
    if tier == "auto":
        # Heuristic: if reduced DB files exist, call it reduced.
        if (data_dir / "uniref90" / "uniref90.fasta").exists() and (data_dir / "small_bfd" / "bfd-first_non_consensus_sequences.fasta").exists():
            tier = "reduced"
        else:
            tier = "full"

    if args.low_mem_gb <= 0:
        print("[mem-watchdog] ERR: --low-mem-gb must be > 0", file=sys.stderr)
        return 2
    if args.low_mem_gb > 1024:
        print("[mem-watchdog] WARN: unusually large --low-mem-gb; clamping to 1024 GiB to avoid surprises")
        args.low_mem_gb = 1024
    low_bytes = int(args.low_mem_gb * 1024**3)

    watch_paths = iter_default_paths(data_dir, tier)

    print(f"[mem-watchdog] data_dir={data_dir} tier={tier} low_mem={fmt_gib(low_bytes)} interval={args.interval_seconds}s cooldown={args.cooldown_seconds}s")
    if not watch_paths:
        print("[mem-watchdog] WARN: no AlphaFold DB files found to manage cache for (nothing to evict).")

    last_evict_ts: float = 0.0

    def tick() -> None:
        nonlocal last_evict_ts
        avail = mem_available_bytes()
        swap_free = swap_free_bytes()
        swap_total = swap_total_bytes()
        under = avail > 0 and avail < low_bytes

        if under:
            now = time.time()
            if now - last_evict_ts < args.cooldown_seconds:
                return

            evicted = evict_paths(watch_paths)
            last_evict_ts = now
            print(
                f"[mem-watchdog] pressure: MemAvailable={fmt_gib(avail)} (< {fmt_gib(low_bytes)}). "
                f"Evicted {evicted} file(s) from page cache. SwapFree={fmt_gib(swap_free)}/{fmt_gib(swap_total)}"
            )
        else:
            # Periodic heartbeat every ~60s.
            if int(time.time()) % 60 == 0:
                print(
                    f"[mem-watchdog] ok: MemAvailable={fmt_gib(avail)} SwapFree={fmt_gib(swap_free)}/{fmt_gib(swap_total)}"
                )

    tick()
    if args.once:
        return 0

    while True:
        time.sleep(max(1, args.interval_seconds))
        tick()


if __name__ == "__main__":
    raise SystemExit(main())
