#!/bin/bash
# run_singularity.sh - Run Obsidian API server as a Singularity instance on HPC
#
# Usage:
#   ./run_singularity.sh start --api               # Start API server as instance
#   ./run_singularity.sh stop                      # Stop the instance
#   ./run_singularity.sh status                    # Check instance status
#   ./run_singularity.sh logs                      # Show instance logs
#
# Options:
#   --port N         Port for API server (default: 8000)
#   --session-dir D  Mount host directory for session persistence (default: ~/.obsidian)
#   --instance NAME  Instance name (default: obsidian-api)
#   --image PATH     Path to Singularity image (default: container/obsidian.sif)
#
# Singularity instances run as background services, perfect for HPC environments.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
COMMAND="start"
PORT="8000"
LOG_DIR="$HOME/.singularity/logs"
SESSION_DIR="$HOME/.obsidian"
INSTANCE_NAME="obsidian-api"
IMAGE_PATH="$PROJECT_ROOT/container/obsidian-server.sif"
USE_API=false

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        start|stop|status|logs)
            COMMAND="$1"
            shift
            ;;
        --api)
            USE_API=true
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
        --instance)
            shift
            INSTANCE_NAME="$1"
            shift
            ;;
        --image)
            shift
            IMAGE_PATH="$1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [start|stop|status|logs] [options]"
            echo ""
            echo "Commands:"
            echo "  start    Start API server as Singularity instance (default)"
            echo "  stop     Stop the running instance"
            echo "  status   Check if instance is running"
            echo "  logs     Show instance logs"
            echo ""
            echo "Options:"
            echo "  --api            Start API server (required for start)"
            echo "  --port N         Port for API server (default: 8000)"
            echo "  --session-dir D  Session directory (default: ~/.obsidian)"
            echo "  --instance NAME  Instance name (default: obsidian-api)"
            echo "  --image PATH     Singularity image path (default: container/obsidian-server.sif)"
            echo ""
            echo "Examples:"
            echo "  $0 start --api                          # Start on port 8000"
            echo "  $0 start --api --port 8080              # Start on port 8080"
            echo "  $0 start --api --session-dir ~/sessions # Custom session dir"
            echo "  $0 stop                                  # Stop the instance"
            echo "  $0 status                                # Check status"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if singularity is available
if ! command -v singularity &> /dev/null; then
    echo "❌ Error: singularity command not found"
    echo ""
    echo "On HPC systems, try:"
    echo "  module load singularity"
    echo "  # or"
    echo "  module load apptainer"
    exit 1
fi

# Execute command
case "$COMMAND" in
    start)
        if [ "$USE_API" != true ]; then
            echo "❌ Error: --api flag is required for start command"
            echo "Usage: $0 start --api [options]"
            exit 1
        fi

        # Check if image exists
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "❌ Error: Singularity image not found: $IMAGE_PATH"
            echo ""
            echo "Build it first:"
            echo "  cd $PROJECT_ROOT"
            echo "  ./container/scripts/build_singularity.sh"
            exit 1
        fi

        # Check if instance already running
        if singularity instance list | grep -q "^$INSTANCE_NAME"; then
            echo "⚠️  Instance '$INSTANCE_NAME' is already running"
            echo ""
            echo "Stop it first: $0 stop"
            echo "Or check status: $0 status"
            exit 1
        fi

        # Create session directory if it doesn't exist
        if [ "$SESSION_DIR" != "none" ]; then
            mkdir -p "$SESSION_DIR"
        fi

        if [ ! -d "$LOG_DIR" ]; then
            mkdir -p "$LOG_DIR"
        fi

        echo "🚀 Starting Obsidian API server as Singularity instance..."
        echo "   Instance: $INSTANCE_NAME"
        echo "   Image: $IMAGE_PATH"
        echo "   Port: $PORT"
        echo "   Log dir: $LOG_DIR"
        if [ "$SESSION_DIR" != "none" ]; then
            echo "   Session dir: $SESSION_DIR → /root/.obsidian"
        else
            echo "   Session dir: ephemeral (no persistence)"
        fi
        echo ""

        # Build bind mounts
        BIND_ARGS=""
        if [ "$SESSION_DIR" != "none" ]; then
            BIND_ARGS="--bind $SESSION_DIR:/root/.obsidian"
        fi

        # Start instance
        # Note: singularity instance doesn't support port mapping, so we rely on host networking
        singularity instance start \
            $BIND_ARGS \
            "$IMAGE_PATH" \
            "$INSTANCE_NAME"

        # Wait a moment for instance to start
        sleep 2

        # Run the API server inside the instance
        singularity exec instance://$INSTANCE_NAME \
            python -m obsidian.api.app --host 0.0.0.0 --port $PORT \
            >> "$LOG_DIR/${INSTANCE_NAME}.log" 2>&1 &

        echo ""
        echo "✅ Instance started successfully!"
        echo ""
        echo "🌐 API server should be available at:"
        echo "   http://$(hostname):$PORT"
        echo "   (or http://localhost:$PORT if on the same machine)"
        echo ""
        echo "Management:"
        echo "  Check status: $0 status"
        echo "  View logs:    $0 logs"
        echo "  Stop server:  $0 stop"
        ;;

    stop)
        echo "🛑 Stopping Singularity instance: $INSTANCE_NAME"

        if ! singularity instance list | grep -q "^$INSTANCE_NAME"; then
            echo "⚠️  Instance '$INSTANCE_NAME' is not running"
            exit 0
        fi

        singularity instance stop "$INSTANCE_NAME"

        echo "✅ Instance stopped"
        ;;

    status)
        echo "📊 Instance status for: $INSTANCE_NAME"
        echo ""

        if singularity instance list | grep -q "^$INSTANCE_NAME"; then
            echo "✅ Instance is RUNNING"
            echo ""
            singularity instance list | grep -E "INSTANCE|^$INSTANCE_NAME"
        else
            echo "❌ Instance is NOT running"
            echo ""
            echo "Start it: $0 start --api"
        fi
        ;;

    logs)
        echo "📜 Logs for instance: $INSTANCE_NAME"
        echo ""

        if ! singularity instance list | grep -q "^$INSTANCE_NAME"; then
            echo "❌ Instance '$INSTANCE_NAME' is not running"
            exit 1
        fi

        LOG_FILE="$LOG_DIR/${INSTANCE_NAME}.log"

        if [ ! -f "$LOG_FILE" ]; then
            echo "No logs found. Check if instance has been started."
            exit 1
        fi

        echo "Instance logs (last 50 lines):"
        tail -50 "$LOG_FILE"
        ;;
esac
