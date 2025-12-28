#!/usr/bin/env bash
set -euo pipefail

# Read-only diagnostics to help non-technical users understand memory/swap/page-cache.
# Safe: does not change sysctls, does not drop caches, does not require root.

fmt_gib() {
  awk -v b="$1" 'BEGIN{printf "%.1f GiB", b/1024/1024/1024}'
}

mem_kb() {
  awk -v k="$1" '$1==k":" {print $2}' /proc/meminfo 2>/dev/null || echo 0
}

mem_total_b=$(( $(mem_kb MemTotal) * 1024 ))
mem_avail_b=$(( $(mem_kb MemAvailable) * 1024 ))
swap_total_b=$(( $(mem_kb SwapTotal) * 1024 ))
swap_free_b=$(( $(mem_kb SwapFree) * 1024 ))

swappiness="$(cat /proc/sys/vm/swappiness 2>/dev/null || echo "?")"
cache_pressure="$(cat /proc/sys/vm/vfs_cache_pressure 2>/dev/null || echo "?")"

echo "== Memory summary =="
echo "MemTotal:      $(fmt_gib "$mem_total_b")"
echo "MemAvailable:  $(fmt_gib "$mem_avail_b")"
echo "SwapTotal:     $(fmt_gib "$swap_total_b")"
echo "SwapFree:      $(fmt_gib "$swap_free_b")"
echo
echo "== Kernel knobs (current) =="
echo "vm.swappiness:        $swappiness"
echo "vm.vfs_cache_pressure: $cache_pressure"
echo
echo "== Active swap devices =="
if command -v swapon >/dev/null 2>&1; then
  swapon --show --bytes || true
else
  echo "swapon not found"
fi

echo
echo "== zram (if present) =="
if ls /sys/block/zram* >/dev/null 2>&1; then
  for z in /sys/block/zram*; do
    dev="$(basename "$z")"
    sz="$(cat "$z/disksize" 2>/dev/null || echo 0)"
    algo="$(cat "$z/comp_algorithm" 2>/dev/null || echo "?")"
    echo "$dev disksize=$(fmt_gib "$sz") algo=$algo"
  done
else
  echo "No zram devices found"
fi

echo
echo "== Notes (for non-technical users) =="
cat <<'EOF'
- Linux uses free RAM as a file-cache (page cache) automatically.
- This cache is *reclaimable*: when apps need memory, the kernel evicts cached file pages.
- "Low free" in GUIs can be normal during heavy I/O (databases stay hot).
- Swap (or zram swap) is a safety net for transient spikes; it can prevent sudden OOM failures.
EOF
