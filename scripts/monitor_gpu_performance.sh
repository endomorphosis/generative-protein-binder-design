#!/bin/bash
# GPU Performance Monitoring Script
# Monitors GPU utilization, memory, and performance during AlphaFold/MMseqs2 operations

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
INTERVAL=${1:-5}  # Sampling interval in seconds
OUTPUT_DIR=${2:-"./gpu_monitoring"}

echo -e "${CYAN}GPU Performance Monitoring${NC}"
echo "======================================"
echo "Interval: ${INTERVAL}s"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/gpu_monitor_$TIMESTAMP.csv"

# Check GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: nvidia-smi not found${NC}"
    exit 1
fi

echo -e "${GREEN}Starting GPU monitoring...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Create CSV header
echo "timestamp,gpu_name,gpu_util_%,mem_used_MB,mem_total_MB,mem_util_%,temp_C,power_W,sm_clock_MHz,mem_clock_MHz" > "$LOG_FILE"

# Trap Ctrl+C to show summary
trap 'show_summary; exit 0' INT

show_summary() {
    echo -e "\n${CYAN}Monitoring Summary${NC}"
    echo "======================================"
    echo "Log file: $LOG_FILE"
    
    if [[ -f "$LOG_FILE" ]]; then
        SAMPLES=$(( $(wc -l < "$LOG_FILE") - 1 ))
        echo "Samples collected: $SAMPLES"
        
        if [[ $SAMPLES -gt 0 ]]; then
            echo ""
            echo "Average GPU Utilization:"
            awk -F',' 'NR>1 {sum+=$3; count++} END {if(count>0) printf "  %.1f%%\n", sum/count}' "$LOG_FILE"
            
            echo "Average Memory Utilization:"
            awk -F',' 'NR>1 {sum+=$6; count++} END {if(count>0) printf "  %.1f%%\n", sum/count}' "$LOG_FILE"
            
            echo "Peak GPU Utilization:"
            awk -F',' 'NR>1 {if($3>max) max=$3} END {printf "  %.1f%%\n", max}' "$LOG_FILE"
            
            echo "Peak Memory Used:"
            awk -F',' 'NR>1 {if($4>max) {max=$4; total=$5}} END {printf "  %d MB / %d MB\n", max, total}' "$LOG_FILE"
            
            echo "Average Temperature:"
            awk -F',' 'NR>1 {sum+=$7; count++} END {if(count>0) printf "  %.1f°C\n", sum/count}' "$LOG_FILE"
            
            echo "Average Power:"
            awk -F',' 'NR>1 {sum+=$8; count++} END {if(count>0) printf "  %.1f W\n", sum/count}' "$LOG_FILE"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}Monitoring stopped${NC}"
}

# Main monitoring loop
while true; do
    # Get current timestamp
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Query GPU metrics
    METRICS=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem \
        --format=csv,noheader,nounits 2>/dev/null || echo "")
    
    if [[ -n "$METRICS" ]]; then
        # Parse metrics
        IFS=',' read -r GPU_NAME GPU_UTIL MEM_USED MEM_TOTAL MEM_UTIL TEMP POWER SM_CLOCK MEM_CLOCK <<< "$METRICS"
        
        # Trim whitespace
        GPU_NAME=$(echo "$GPU_NAME" | xargs)
        GPU_UTIL=$(echo "$GPU_UTIL" | xargs)
        MEM_USED=$(echo "$MEM_USED" | xargs)
        MEM_TOTAL=$(echo "$MEM_TOTAL" | xargs)
        MEM_UTIL=$(echo "$MEM_UTIL" | xargs)
        TEMP=$(echo "$TEMP" | xargs)
        POWER=$(echo "$POWER" | xargs)
        SM_CLOCK=$(echo "$SM_CLOCK" | xargs)
        MEM_CLOCK=$(echo "$MEM_CLOCK" | xargs)
        
        # Write to CSV
        echo "$TS,$GPU_NAME,$GPU_UTIL,$MEM_USED,$MEM_TOTAL,$MEM_UTIL,$TEMP,$POWER,$SM_CLOCK,$MEM_CLOCK" >> "$LOG_FILE"
        
        # Display current metrics
        printf "\r${BLUE}[%s]${NC} GPU: %3s%% | Mem: %5s/%5s MB (%3s%%) | Temp: %3s°C | Power: %6s W" \
            "$TS" "$GPU_UTIL" "$MEM_USED" "$MEM_TOTAL" "$MEM_UTIL" "$TEMP" "$POWER"
    else
        printf "\r${YELLOW}[%s] No GPU data available${NC}" "$TS"
    fi
    
    sleep "$INTERVAL"
done
