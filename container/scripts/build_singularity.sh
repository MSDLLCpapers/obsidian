#!/bin/bash
# build_singularity.sh - Build Singularity image from Docker image
#
# Usage (HPC):
#   ./build_singularity.sh                    # Build from local Docker image
#   ./build_singularity.sh --from-registry    # Pull from registry (uses .env config)
#
# This script converts the Docker production image to Singularity format
# for use on HPC systems.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env file if it exists (for registry configuration)
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading configuration from .env file..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | grep -v '^$' | xargs)
fi

# Default values
FROM_REGISTRY=false
REGISTRY_URL=""
OUTPUT_FILE="container/obsidian-server.sif"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --from-registry)
            FROM_REGISTRY=true
            if [ -n "$2" ] && [[ ! "$2" =~ ^-- ]]; then
                REGISTRY_URL="$2"
                shift 2
            else
                shift
            fi
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --from-registry [URL]  Pull from Docker registry"
            echo "                         If URL not provided, uses DOCKER_REGISTRY and IMAGE_TAG from .env"
            echo "  --output FILE          Output file path (default: container/obsidian-server.sif)"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Building Singularity Image"
echo "========================================="
echo "Output: $OUTPUT_FILE"
echo ""

# Check if Singularity is available
if ! command -v singularity &> /dev/null; then
    echo "❌ Error: singularity command not found"
    echo ""
    echo "On HPC, ensure Singularity module is loaded:"
    echo "  module load singularity"
    echo ""
    echo "On Mac, install Apptainer/Singularity:"
    echo "  brew install apptainer"
    exit 1
fi

singularity --version

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

if [ "$FROM_REGISTRY" = true ]; then
    # If no URL provided, construct from .env
    if [ -z "$REGISTRY_URL" ]; then
        if [ -z "$DOCKER_REGISTRY" ]; then
            echo "❌ Error: --from-registry requires either:"
            echo "  1. A URL argument: --from-registry docker://registry/image:tag"
            echo "  2. Or a .env file with DOCKER_REGISTRY and IMAGE_TAG defined"
            exit 1
        fi
        REGISTRY_URL="docker://${DOCKER_REGISTRY}/obsidian-server-prod:${IMAGE_TAG}"
    fi

    echo "📥 Pulling from registry: $REGISTRY_URL"
    singularity pull "$OUTPUT_FILE" "$REGISTRY_URL"

else
    # Build from local Docker image
    DOCKER_IMAGE="obsidian-server-prod:${IMAGE_TAG}"
    echo "🔧 Converting Docker image to Singularity..."
    echo "   Docker image: $DOCKER_IMAGE"

    # Check if Docker image exists
    if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^obsidian-server-prod:${IMAGE_TAG}$"; then
        echo "❌ Error: Docker image '$DOCKER_IMAGE' not found"
        echo ""
        echo "Please build it first:"
        echo "  ./container/scripts/build_docker.sh prod"
        exit 1
    fi

    # Save Docker image to tar
    echo "   Exporting Docker image..."
    TEMP_TAR=$(mktemp -u).tar
    trap "rm -f $TEMP_TAR" EXIT

    docker save "$DOCKER_IMAGE" > "$TEMP_TAR"

    # Build Singularity image from tar
    echo "   Building Singularity image..."
    singularity build "$OUTPUT_FILE" "docker-archive://$TEMP_TAR"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Singularity image built successfully!"
    echo "========================================="
    echo "Image: $OUTPUT_FILE"
    echo "Size:  $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo ""
    echo "Test it:"
    echo "  singularity exec $OUTPUT_FILE python --version"
    echo "  singularity exec $OUTPUT_FILE python -m obsidian.api.app --help"
    echo ""
    echo "Run API server:"
    echo "  # One-off execution"
    echo "  singularity exec $OUTPUT_FILE python -m obsidian.api.app --host 0.0.0.0 --port 8000"
    echo ""
    echo "  # As a persistent service (recommended for HPC)"
    echo "  ./container/scripts/run_singularity.sh --api"
    echo ""
    echo "Session persistence:"
    echo "  By default, sessions are stored in ~/.obsidian on the host"
    echo "  To customize: singularity exec --bind ~/my-sessions:/root/.obsidian ..."
else
    echo "❌ Failed to build Singularity image"
    exit 1
fi
