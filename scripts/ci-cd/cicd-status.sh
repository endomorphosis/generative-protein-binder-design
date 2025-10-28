#!/bin/bash
set -euo pipefail

# CI/CD Status Monitor Script
# Monitors GitHub Actions workflows and provides status reports

echo "=== CI/CD Status Monitor ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
REPO_OWNER="${GITHUB_REPOSITORY_OWNER:-hallucinate-llc}"
REPO_NAME="${GITHUB_REPOSITORY_NAME:-generative-protein-binder-design}"
BRANCH="${1:-$(git branch --show-current 2>/dev/null || echo 'main')}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Function to check if gh CLI is available
check_gh_cli() {
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed"
        echo "Install it from: https://cli.github.com/"
        return 1
    fi
    
    # Check if authenticated
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated"
        echo "Run: gh auth login"
        return 1
    fi
    
    return 0
}

# Function to get workflow status
get_workflow_status() {
    local workflow_name=$1
    local branch=${2:-$BRANCH}
    
    log_info "Checking workflow: $workflow_name (branch: $branch)"
    
    # Get latest workflow run
    local workflow_data
    workflow_data=$(gh run list \
        --repo "$REPO_OWNER/$REPO_NAME" \
        --workflow "$workflow_name" \
        --branch "$branch" \
        --limit 1 \
        --json status,conclusion,createdAt,htmlUrl,workflowName \
        2>/dev/null) || {
        log_warn "Failed to get workflow data for: $workflow_name"
        return 1
    }
    
    if [[ "$workflow_data" == "[]" ]]; then
        echo "  Status: No runs found"
        return 0
    fi
    
    # Parse workflow data
    local status conclusion created_at html_url
    status=$(echo "$workflow_data" | jq -r '.[0].status // "unknown"')
    conclusion=$(echo "$workflow_data" | jq -r '.[0].conclusion // "none"')
    created_at=$(echo "$workflow_data" | jq -r '.[0].createdAt // "unknown"')
    html_url=$(echo "$workflow_data" | jq -r '.[0].htmlUrl // "unknown"')
    
    # Format status
    local status_icon
    case "$status" in
        "completed")
            case "$conclusion" in
                "success") status_icon="✅" ;;
                "failure") status_icon="❌" ;;
                "cancelled") status_icon="🚫" ;;
                *) status_icon="⚠️" ;;
            esac
            ;;
        "in_progress") status_icon="🔄" ;;
        "queued") status_icon="⏳" ;;
        *) status_icon="❓" ;;
    esac
    
    echo "  Status: $status_icon $status"
    if [[ "$conclusion" != "none" ]] && [[ "$conclusion" != "null" ]]; then
        echo "  Result: $conclusion"
    fi
    echo "  Created: $created_at"
    echo "  URL: $html_url"
}

# Function to monitor all workflows
monitor_workflows() {
    log_header "GitHub Actions Workflow Status"
    echo "Repository: $REPO_OWNER/$REPO_NAME"
    echo "Branch: $BRANCH"
    echo "Timestamp: $(date)"
    echo

    # List of workflows to monitor
    local workflows=(
        "CI/CD Main Pipeline"
        "ARM64 Native CI/CD Pipeline" 
        "Docker Multi-Platform CI/CD"
        "Protein Design Pipeline"
        "Docker Compatibility Test"
    )
    
    for workflow in "${workflows[@]}"; do
        echo "---"
        get_workflow_status "$workflow" "$BRANCH" || true
        echo
    done
}

# Function to get recent workflow runs summary
get_recent_runs() {
    log_header "Recent Workflow Runs (Last 10)"
    
    gh run list \
        --repo "$REPO_OWNER/$REPO_NAME" \
        --limit 10 \
        --json status,conclusion,workflowName,createdAt,headBranch \
        | jq -r '.[] | "\(.createdAt | strftime("%Y-%m-%d %H:%M")) | \(.headBranch) | \(.workflowName) | \(.status)/\(.conclusion // "none")"' \
        | column -t -s '|' || log_warn "Failed to get recent runs"
}

# Function to get workflow artifacts
get_artifacts() {
    local workflow_name=${1:-""}
    
    if [[ -z "$workflow_name" ]]; then
        log_header "Available Artifacts (All Workflows)"
        gh run list \
            --repo "$REPO_OWNER/$REPO_NAME" \
            --limit 5 \
            --json workflowName,conclusion,artifacts \
            | jq -r '.[] | select(.artifacts | length > 0) | "\(.workflowName): \(.artifacts | length) artifacts"' \
            || log_warn "Failed to get artifacts"
    else
        log_header "Artifacts for: $workflow_name"
        gh run list \
            --repo "$REPO_OWNER/$REPO_NAME" \
            --workflow "$workflow_name" \
            --limit 1 \
            --json artifacts \
            | jq -r '.[0].artifacts[]? | "- \(.name) (\(.size_in_bytes) bytes)"' \
            || log_warn "Failed to get artifacts for workflow"
    fi
}

# Function to check self-hosted runners
check_runners() {
    log_header "Self-Hosted Runners Status"
    
    if gh api "/repos/$REPO_OWNER/$REPO_NAME/actions/runners" --jq '.runners[] | "\(.name): \(.status) (\(.os)/\(.architecture))"' 2>/dev/null; then
        echo
    else
        log_warn "Failed to get runner information (may require admin access)"
    fi
}

# Function to generate CI/CD health report
generate_health_report() {
    log_header "CI/CD Health Report"
    
    local total_workflows=0
    local failed_workflows=0
    local success_workflows=0
    local in_progress_workflows=0
    
    # Get recent workflow statistics
    local stats
    stats=$(gh run list \
        --repo "$REPO_OWNER/$REPO_NAME" \
        --limit 50 \
        --json status,conclusion \
        2>/dev/null) || {
        log_warn "Failed to generate health report"
        return 1
    }
    
    total_workflows=$(echo "$stats" | jq '. | length')
    success_workflows=$(echo "$stats" | jq '[.[] | select(.conclusion == "success")] | length')
    failed_workflows=$(echo "$stats" | jq '[.[] | select(.conclusion == "failure")] | length')
    in_progress_workflows=$(echo "$stats" | jq '[.[] | select(.status == "in_progress")] | length')
    
    echo "Recent Workflow Statistics (Last 50 runs):"
    echo "  Total: $total_workflows"
    echo "  Success: $success_workflows"
    echo "  Failed: $failed_workflows" 
    echo "  In Progress: $in_progress_workflows"
    
    if [[ $total_workflows -gt 0 ]]; then
        local success_rate
        success_rate=$(echo "scale=1; $success_workflows * 100 / $total_workflows" | bc -l 2>/dev/null || echo "N/A")
        echo "  Success Rate: $success_rate%"
    fi
    
    echo
    
    # Check for frequent failures
    local frequent_failures
    frequent_failures=$(echo "$stats" | jq -r '[.[] | select(.conclusion == "failure")] | group_by(.workflowName) | map({workflow: .[0].workflowName, count: length}) | sort_by(.count) | reverse | .[0:3][] | "\(.workflow): \(.count) failures"' 2>/dev/null) || true
    
    if [[ -n "$frequent_failures" ]]; then
        echo "Most Failed Workflows:"
        echo "$frequent_failures" | sed 's/^/  /'
    fi
}

# Function to watch workflows in real-time
watch_workflows() {
    local interval=${1:-30}
    
    log_info "Starting workflow monitoring (refresh every ${interval}s, press Ctrl+C to stop)"
    
    while true; do
        clear
        echo "=== Live CI/CD Status Monitor ==="
        echo "Repository: $REPO_OWNER/$REPO_NAME"
        echo "Update interval: ${interval}s"
        echo "Last update: $(date)"
        echo
        
        monitor_workflows
        
        echo
        echo "Press Ctrl+C to stop monitoring"
        
        sleep "$interval"
    done
}

# Function to show usage
show_usage() {
    cat << 'EOL'
Usage: ./cicd-status.sh [COMMAND] [OPTIONS]

Commands:
  status      Show current workflow status (default)
  recent      Show recent workflow runs
  artifacts   Show available artifacts
  runners     Show self-hosted runners status
  health      Generate CI/CD health report
  watch       Watch workflows in real-time
  help        Show this help message

Options:
  --branch BRANCH    Check specific branch (default: current)
  --workflow NAME    Check specific workflow
  --interval SEC     Watch interval in seconds (default: 30)

Examples:
  ./cicd-status.sh                           # Show status for current branch
  ./cicd-status.sh status --branch main      # Show status for main branch
  ./cicd-status.sh recent                    # Show recent runs
  ./cicd-status.sh watch --interval 15       # Watch with 15s updates
  ./cicd-status.sh artifacts --workflow "CI/CD Main Pipeline"

Environment Variables:
  GITHUB_REPOSITORY_OWNER  Repository owner (default: hallucinate-llc)
  GITHUB_REPOSITORY_NAME   Repository name (default: generative-protein-binder-design)
EOL
}

# Parse command line arguments
parse_args() {
    local command="${1:-status}"
    shift || true
    
    local workflow_name=""
    local interval=30
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --workflow)
                workflow_name="$2"
                shift 2
                ;;
            --interval)
                interval="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    case $command in
        "status")
            monitor_workflows
            ;;
        "recent")
            get_recent_runs
            ;;
        "artifacts")
            get_artifacts "$workflow_name"
            ;;
        "runners")
            check_runners
            ;;
        "health")
            generate_health_report
            ;;
        "watch")
            watch_workflows "$interval"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Main execution
main() {
    # Check prerequisites
    if ! check_gh_cli; then
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir &> /dev/null; then
        log_error "Not in a Git repository"
        exit 1
    fi
    
    # Parse arguments and execute command
    parse_args "$@"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n\nMonitoring stopped."; exit 0' INT

# Run main function
main "$@"