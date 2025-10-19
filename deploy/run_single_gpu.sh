#!/bin/bash

# NVIDIA BioNeMo Blueprint: Single GPU Sequential Service Runner
# This script runs NIM services one at a time on systems with limited GPUs

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Available services
SERVICES=("alphafold" "rfdiffusion" "proteinmpnn" "alphafold-multimer")

# Function to start a specific service
start_service() {
    local service=$1
    print_info "Starting $service service..."
    docker compose --profile $service up -d
    print_status "$service service started"
}

# Function to stop all services
stop_all_services() {
    print_info "Stopping all services..."
    docker compose down
    print_status "All services stopped"
}

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    print_info "Checking $service health..."
    
    # Wait for service to be ready
    local retries=60
    local count=0
    
    while [ $count -lt $retries ]; do
        if curl -s http://localhost:$port/v1/health/ready > /dev/null 2>&1; then
            print_status "$service is ready!"
            return 0
        fi
        sleep 10
        count=$((count + 1))
        print_info "Waiting for $service... ($count/$retries)"
    done
    
    print_error "$service failed to become ready within 10 minutes"
    return 1
}

# Function to run a single service workflow
run_single_service() {
    local service=$1
    local port=$2
    
    echo "======================================"
    echo "🧬 Running $service Service"
    echo "======================================"
    
    # Stop any running services first
    stop_all_services
    
    # Start the requested service
    start_service $service
    
    # Check if service is ready
    if check_service $service $port; then
        print_status "$service is running and ready at http://localhost:$port"
        print_info "You can now use this service in your workflow"
        print_info "Press Ctrl+C to stop the service when done"
        
        # Keep service running until user interrupts
        trap 'stop_all_services; exit 0' INT
        while true; do
            sleep 30
            if ! curl -s http://localhost:$port/v1/health/ready > /dev/null 2>&1; then
                print_error "$service appears to have stopped"
                break
            fi
        done
    else
        print_error "Failed to start $service"
        stop_all_services
        exit 1
    fi
}

# Function to start all services simultaneously (original behavior)
start_all_services() {
    echo "======================================"
    echo "🧬 Starting All Services (Single GPU)"
    echo "======================================"
    
    print_warning "All services will share GPU 0 - performance may be limited"
    
    # Start all services
    docker compose up -d
    
    # Check each service
    print_info "Checking service readiness..."
    sleep 30  # Give services time to start
    
    local services_ready=0
    local total_services=4
    
    if curl -s http://localhost:8081/v1/health/ready > /dev/null 2>&1; then
        print_status "AlphaFold2 is ready"
        services_ready=$((services_ready + 1))
    else
        print_warning "AlphaFold2 not ready yet"
    fi
    
    if curl -s http://localhost:8082/v1/health/ready > /dev/null 2>&1; then
        print_status "RFDiffusion is ready"
        services_ready=$((services_ready + 1))
    else
        print_warning "RFDiffusion not ready yet"
    fi
    
    if curl -s http://localhost:8083/v1/health/ready > /dev/null 2>&1; then
        print_status "ProteinMPNN is ready"
        services_ready=$((services_ready + 1))
    else
        print_warning "ProteinMPNN not ready yet"
    fi
    
    if curl -s http://localhost:8084/v1/health/ready > /dev/null 2>&1; then
        print_status "AlphaFold2-Multimer is ready"
        services_ready=$((services_ready + 1))
    else
        print_warning "AlphaFold2-Multimer not ready yet"
    fi
    
    print_info "$services_ready/$total_services services are ready"
    
    if [ $services_ready -gt 0 ]; then
        print_status "At least one service is running - you can begin your workflow"
    else
        print_error "No services are ready - check logs with: docker compose logs"
    fi
}

# Main menu
show_menu() {
    echo "======================================"
    echo "🧬 NVIDIA BioNeMo Single GPU Manager"
    echo "======================================"
    echo
    echo "Choose an option:"
    echo "1) Start AlphaFold2 only (port 8081)"
    echo "2) Start RFDiffusion only (port 8082)"
    echo "3) Start ProteinMPNN only (port 8083)"
    echo "4) Start AlphaFold2-Multimer only (port 8084)"
    echo "5) Start all services (may cause GPU memory issues)"
    echo "6) Stop all services"
    echo "7) Check service status"
    echo "8) View logs"
    echo "9) Exit"
    echo
}

# Function to check status of all services
check_all_status() {
    echo "======================================"
    echo "🔍 Service Status Check"
    echo "======================================"
    
    local ports=(8081 8082 8083 8084)
    local names=("AlphaFold2" "RFDiffusion" "ProteinMPNN" "AlphaFold2-Multimer")
    
    for i in "${!ports[@]}"; do
        local port=${ports[$i]}
        local name=${names[$i]}
        
        if curl -s http://localhost:$port/v1/health/ready > /dev/null 2>&1; then
            print_status "$name is ready (port $port)"
        else
            print_warning "$name is not ready (port $port)"
        fi
    done
}

# Function to show logs
show_logs() {
    echo "======================================"
    echo "📋 Service Logs"
    echo "======================================"
    docker compose logs --tail=50
}

# Main execution
main() {
    if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        echo "NVIDIA BioNeMo Single GPU Manager"
        echo
        echo "Usage: $0 [service_name]"
        echo
        echo "Available services:"
        echo "  alphafold          - Start only AlphaFold2"
        echo "  rfdiffusion        - Start only RFDiffusion"
        echo "  proteinmpnn        - Start only ProteinMPNN"
        echo "  alphafold-multimer - Start only AlphaFold2-Multimer"
        echo "  all                - Start all services"
        echo "  stop               - Stop all services"
        echo "  status             - Check service status"
        echo
        echo "If no service is specified, interactive menu will be shown."
        exit 0
    fi
    
    # Handle command line arguments
    case "$1" in
        "alphafold")
            run_single_service alphafold 8081
            ;;
        "rfdiffusion")
            run_single_service rfdiffusion 8082
            ;;
        "proteinmpnn")
            run_single_service proteinmpnn 8083
            ;;
        "alphafold-multimer")
            run_single_service alphafold-multimer 8084
            ;;
        "all")
            start_all_services
            ;;
        "stop")
            stop_all_services
            ;;
        "status")
            check_all_status
            ;;
        "")
            # Interactive mode
            while true; do
                show_menu
                read -p "Enter your choice (1-9): " choice
                
                case $choice in
                    1)
                        run_single_service alphafold 8081
                        ;;
                    2)
                        run_single_service rfdiffusion 8082
                        ;;
                    3)
                        run_single_service proteinmpnn 8083
                        ;;
                    4)
                        run_single_service alphafold-multimer 8084
                        ;;
                    5)
                        start_all_services
                        break
                        ;;
                    6)
                        stop_all_services
                        ;;
                    7)
                        check_all_status
                        ;;
                    8)
                        show_logs
                        ;;
                    9)
                        print_info "Goodbye!"
                        exit 0
                        ;;
                    *)
                        print_error "Invalid choice. Please try again."
                        ;;
                esac
                echo
                read -p "Press Enter to continue..."
                clear
            done
            ;;
        *)
            print_error "Unknown service: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Change to the deploy directory
cd "$(dirname "$0")"

# Run main function
main "$@"