#!/bin/bash
# run_docker.sh - Run commands in Docker container with proper volume mounts
#
# Usage:
#   ./run_docker.sh python -m obsidian.api.app
#   ./run_docker.sh python script.py              # Run any Python script
#   ./run_docker.sh --api                         # Start API server (port 8000)
#   ./run_docker.sh --dash                        # Start Dash app (port 8050)
#   ./run_docker.sh bash
#   ./run_docker.sh --prod bash                   # Use prod image (no volume mounts)
#
# Options:
#   --prod           Use prod image (code baked in, no volumes)
#                    Default: Use dev image (with volume mounts for live editing)
#   --api            Start API server on port 8000 (maps to host 8000)
#   --dash           Start Dash app on port 8050 (maps to host 8050)
#   --port N         Override default port for --api or --dash
#   --session-dir D  Mount host directory for session persistence (default: ~/.obsidian)
#                    Use "none" to disable session mounting
#
# Dev mode:
# - Uses obsidian-server-dev:latest image
# - Mounts ../obsidian and current directory as volumes
# - Code changes are immediately visible (no rebuild needed)
#
# Prod mode:
# - Uses obsidian-server-prod:latest image
# - Code is baked into the image (no volume mounts)
# - Simulates HPC Singularity environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OBSIDIAN_DIR="$PROJECT_ROOT"

# Parse flags (order-independent)
USE_PROD=false
USE_API=false
USE_DASH=false
PORT=""
SESSION_DIR=""
ARGS=()

# First pass: collect all flags
while [ $# -gt 0 ]; do
    case "$1" in
        --prod)
            USE_PROD=true
            shift
            ;;
        --api)
            USE_API=true
            PORT="${PORT:-8000}"
            shift
            ;;
        --dash)
            USE_DASH=true
            PORT="${PORT:-8050}"
            shift
            ;;
        --port)
            shift
            PORT="$1"
            shift
            ;;
        --session-dir)
            shift
            SESSION_DIR="$1"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Set default session directory (only for API/Dash)
if [ "$USE_API" = true ] || [ "$USE_DASH" = true ]; then
    if [ -z "$SESSION_DIR" ]; then
        # Default: use host's ~/.obsidian
        SESSION_DIR="$HOME/.obsidian"
    fi
fi

# Set command based on shortcuts
if [ "$USE_API" = true ]; then
    ARGS=("python" "-m" "obsidian.api.app" "--host" "0.0.0.0" "--port" "$PORT")
elif [ "$USE_DASH" = true ]; then
    ARGS=("python" "app.py")
fi

# Set image name based on mode
if [ "$USE_PROD" = true ]; then
    IMAGE_NAME="obsidian-server-prod"
    IMAGE_MODE="prod"
else
    IMAGE_NAME="obsidian-server-dev"
    IMAGE_MODE="dev"
fi

# Check if Docker image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}"; then
    echo "❌ Error: Docker image '${IMAGE_NAME}' not found"
    echo ""
    echo "Please build it first:"
    echo "  cd $PROJECT_ROOT"
    echo "  ./container/scripts/build_docker.sh ${IMAGE_MODE}"

    # Suggest alternative if the other mode exists
    if [ "$USE_PROD" = true ]; then
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^obsidian-server-dev"; then
            echo ""
            echo "💡 Tip: Dev image exists. Try without --prod flag for development."
        fi
    else
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^obsidian-server-prod"; then
            echo ""
            echo "💡 Tip: Prod image exists. Try with --prod flag to use it."
        fi
    fi
    exit 1
fi

# Set up volume mounts (only for dev mode)
if [ "$USE_PROD" = false ]; then
    echo "✓ Found obsidian at: $OBSIDIAN_DIR"
    MOUNT_OBSIDIAN="-v $OBSIDIAN_DIR:/opt/obsidian"
else
    # Prod mode: no volume mounts
    MOUNT_OBSIDIAN=""
fi

# Set up port mapping
PORT_MAPPING=""
if [ -n "$PORT" ]; then
    PORT_MAPPING="-p $PORT:$PORT"
fi

# Set up session directory mount
MOUNT_SESSION=""
if [ -n "$SESSION_DIR" ] && [ "$SESSION_DIR" != "none" ]; then
    # Create directory if it doesn't exist
    mkdir -p "$SESSION_DIR"
    MOUNT_SESSION="-v $SESSION_DIR:/root/.obsidian"
fi

# Set working directory
if [ "$USE_PROD" = true ]; then
    WORK_DIR="/work"
else
    WORK_DIR="/opt/obsidian"
fi

# Run Docker container
echo "🐳 Running command in Docker container..."
echo "   Image: ${IMAGE_NAME}:latest"
echo "   Mode:  ${IMAGE_MODE}"
echo "   Working directory: $WORK_DIR"

if [ "$USE_PROD" = true ]; then
    echo "   Packages: Baked into image"
else
    echo "   Packages: Pre-installed in editable mode (live editing)"
fi

if [ -n "$PORT" ]; then
    echo "   Port mapping: $PORT:$PORT (host:container)"
fi

if [ -n "$MOUNT_SESSION" ]; then
    echo "   Session dir: $SESSION_DIR → /root/.obsidian"
elif [ "$USE_API" = true ] || [ "$USE_DASH" = true ]; then
    echo "   ⚠️  Session dir: ephemeral (use --session-dir for persistence)"
fi

if [ "$USE_API" = true ]; then
    echo "   🌐 API server will be available at: http://localhost:$PORT"
    echo "   Command: ${ARGS[*]}"
elif [ "$USE_DASH" = true ]; then
    echo "   🌐 Dash app will be available at: http://localhost:$PORT"
    echo "   Command: ${ARGS[*]}"
elif [ ${#ARGS[@]} -gt 0 ]; then
    echo "   Command: ${ARGS[*]}"
fi
echo ""

# Run container
# Dev mode: with volume mounts for live editing
# Prod mode: code baked in, simulates HPC environment
docker run --rm -it \
    $MOUNT_OBSIDIAN \
    $MOUNT_SESSION \
    $PORT_MAPPING \
    -w "$WORK_DIR" \
    -e PYTHONHASHSEED=0 \
    ${IMAGE_NAME}:latest \
    "${ARGS[@]}"
