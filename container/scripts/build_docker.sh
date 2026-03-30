#!/bin/bash
# build_docker.sh - Build Docker images with smart hash-based rebuild detection
#
# Usage:
#   ./build_docker.sh base         # Build base image only (if deps changed)
#   ./build_docker.sh dev          # Build dev image (if base or setup changed)
#   ./build_docker.sh prod         # Build prod image (always - code changes)
#   ./build_docker.sh all          # Build all images (smart detection)
#
# Options:
#   --no-cache                     # Disable Docker cache AND skip detection
#   --python-version 3.10.19       # Override Python version
#   --push                         # Tag and push to registry (requires .env file)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure we're in the obsidian directory
if [ ! -f "pyproject.toml" ] || ! grep -q "obsidian-apo" pyproject.toml 2>/dev/null; then
    echo "❌ Error: Must run from obsidian project root"
    echo "   Current directory: $PROJECT_ROOT"
    exit 1
fi

# ============================================
# Configuration and Validation
# ============================================

load_env() {
    if [ -f "$SCRIPT_DIR/.env" ]; then
        echo "Loading configuration from .env file..."
        export $(grep -v '^#' "$SCRIPT_DIR/.env" | grep -v '^$' | xargs)
    fi
}

validate_push_config() {
    if [ "$PUSH_TO_REGISTRY" != "true" ]; then
        return
    fi

    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        echo "❌ Error: --push requires a .env file in $SCRIPT_DIR"
        echo "   Copy .env.example to .env and configure your registry settings"
        exit 1
    fi

    if [ -z "$DOCKER_REGISTRY" ] || [ -z "$IMAGE_TAG" ]; then
        echo "❌ Error: .env file must define DOCKER_REGISTRY and IMAGE_TAG"
        exit 1
    fi

    echo "Registry push enabled:"
    echo "  Registry: $DOCKER_REGISTRY"
    echo "  Tag:      $IMAGE_TAG"
    echo ""
}

detect_python_version() {
    if [ -n "$PYTHON_VERSION" ]; then
        return
    fi

    if [ -f "container/docker/requirements_docker.txt" ]; then
        PYTHON_VERSION=$(grep "# Python:" container/docker/requirements_docker.txt | awk '{print $3}')
    fi

    if [ -z "$PYTHON_VERSION" ]; then
        PYTHON_VERSION="3.10.19"
        echo "Using default Python version: $PYTHON_VERSION"
    else
        echo "Auto-detected Python version: $PYTHON_VERSION"
    fi
}

load_base_image_digest() {
    local LOCK_FILE="container/docker/python-3.10-slim-digest.lock"
    
    if [ ! -f "$LOCK_FILE" ]; then
        echo "❌ Error: Lock file not found: $LOCK_FILE"
        exit 1
    fi

    PYTHON_BASE_IMAGE=$(grep "^${PYTHON_VERSION}=" "$LOCK_FILE" 2>/dev/null | cut -d'=' -f2)

    if [ -z "$PYTHON_BASE_IMAGE" ]; then
        echo "❌ Error: Python ${PYTHON_VERSION} not found in $LOCK_FILE"
        echo ""
        echo "Available versions:"
        grep "^[0-9]" "$LOCK_FILE" | cut -d'=' -f1 | sed 's/^/  /'
        echo ""
        echo "To add version ${PYTHON_VERSION}:"
        echo "  ./container/scripts/update_base_image_digest.sh ${PYTHON_VERSION}"
        exit 1
    fi

    echo "Using pinned base image from lock file:"
    echo "  Version: $PYTHON_VERSION"
    echo "  Digest:  ${PYTHON_BASE_IMAGE#*@}"
}

validate_requirements() {
    if [ ! -f "container/docker/requirements_docker.txt" ]; then
        echo "❌ Error: container/docker/requirements_docker.txt not found!"
        echo ""
        echo "Please generate it first:"
        echo "  conda activate obsidian"
        echo "  python container/generate_requirements.py"
        exit 1
    fi
}

validate_obsidian() {
    if [ "$TARGET" = "base" ]; then
        return
    fi

    if [ ! -d "obsidian" ]; then
        echo "❌ Error: obsidian/ directory not found!"
        echo "   Dev and prod images require obsidian source code"
        exit 1
    fi
}

# ============================================
# Build Context Preparation
# ============================================

prepare_build_context() {
    # Create temporary build directory with obsidian
    # Returns the temp directory path via echo
    # Caller is responsible for cleanup via trap

    local TEMP_BUILD_DIR=$(mktemp -d)

    echo "   Preparing build context..." >&2

    # Copy obsidian (excluding files in .dockerignore)
    echo "     - Copying obsidian..." >&2
    rsync -a --exclude-from=.dockerignore \
        . "$TEMP_BUILD_DIR/obsidian/"

    # Explicitly copy readme.md (may be excluded by .dockerignore but needed for package)
    if [ -f "readme.md" ]; then
        cp readme.md "$TEMP_BUILD_DIR/obsidian/"
    fi

    echo "$TEMP_BUILD_DIR"
}

# ============================================
# Hash Computation Functions
# ============================================

compute_base_deps_tag() {
    local PACKAGES_HASH=$(grep -v '^#' container/docker/requirements_docker.txt | grep -v '^$' | sort | md5sum | cut -d' ' -f1)
    local PLATFORM="linux/amd64"
    local COMBINED="${PYTHON_VERSION}${PYTHON_BASE_IMAGE}${PACKAGES_HASH}${PLATFORM}"
    local HASH=$(echo -n "$COMBINED" | md5sum | cut -c1-12)
    echo "deps-${HASH}"
}

compute_dev_deps_tag() {
    local BASE_TAG=$1

    local OBSIDIAN_HASH=""
    if [ -f "pyproject.toml" ]; then
        OBSIDIAN_HASH=$(md5sum pyproject.toml | cut -d' ' -f1)
    elif [ -f "setup.py" ]; then
        OBSIDIAN_HASH=$(md5sum setup.py | cut -d' ' -f1)
    fi

    local COMBINED="${BASE_TAG}${OBSIDIAN_HASH}"
    local HASH=$(echo -n "$COMBINED" | md5sum | cut -c1-12)
    echo "dev-${HASH}"
}

extract_git_versions() {
    # Get package version from pyproject.toml
    OBSIDIAN_PKG_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

    # Get git SHA
    OBSIDIAN_SHA=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    if ! git diff-index --quiet HEAD 2>/dev/null; then
        OBSIDIAN_SHA="${OBSIDIAN_SHA}-dirty"
    fi

    echo "   Obsidian versions:"
    echo "     Package: ${OBSIDIAN_PKG_VERSION}"
    echo "     Git SHA: ${OBSIDIAN_SHA}"
}

# ============================================
# Build and Push Functions
# ============================================

image_exists() {
    local IMAGE_NAME=$1
    local IMAGE_TAG=$2
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}:${IMAGE_TAG}$"
}

ensure_base_image_exists() {
    if ! image_exists "obsidian-server-base" "$BASE_DEPS_TAG"; then
        echo "❌ Error: Base image 'obsidian-server-base:${BASE_DEPS_TAG}' not found"
        echo "   Please build it first: ./container/scripts/build_docker.sh base"
        exit 1
    fi
}

push_to_registry() {
    if [ "$PUSH_TO_REGISTRY" != "true" ]; then
        return
    fi

    local IMAGE_NAME=$1
    local IMAGE_TAG=$2
    local REGISTRY_IMAGE="${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    echo ""
    echo "📤 Pushing to registry: ${REGISTRY_IMAGE}"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} "${REGISTRY_IMAGE}"
    docker push "${REGISTRY_IMAGE}"
    docker rmi "${REGISTRY_IMAGE}"
    echo "✅ Pushed successfully"
    
    # Track what was pushed for summary
    PUSHED_IMAGES+=("${IMAGE_NAME}:${IMAGE_TAG}")
}

tag_as_latest() {
    local IMAGE_NAME=$1
    local IMAGE_TAG=$2
    
    echo ""
    echo "🔖 Setting :latest alias"
    echo "   ${IMAGE_NAME}:latest → ${IMAGE_NAME}:${IMAGE_TAG}"
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
}

build_image() {
    local IMAGE_TYPE=$1       # "base", "dev", or "prod"
    local IMAGE_NAME=$2
    local IMAGE_TAG=$3
    local DOCKERFILE=$4
    local BASE_BUILD_CONTEXT=$5  # For base: ".", for dev/prod: ignored
    shift 5
    local EXTRA_ARGS="$@"
    
    local BUILT_NOW=false
    local BUILD_CONTEXT="$BASE_BUILD_CONTEXT"
    local TEMP_BUILD_DIR=""
    
    # Decide if we should build
    local SHOULD_BUILD=false
    if [ "$IMAGE_TYPE" = "prod" ]; then
        SHOULD_BUILD=true  # Prod always builds
    elif ! image_exists "$IMAGE_NAME" "$IMAGE_TAG" || [ -n "$NO_CACHE" ]; then
        SHOULD_BUILD=true  # Base/dev builds if doesn't exist or --no-cache
    fi
    
    if [ "$SHOULD_BUILD" = true ]; then
        # Prepare context for dev/prod (base uses passed-in context)
        if [ "$IMAGE_TYPE" != "base" ]; then
            TEMP_BUILD_DIR=$(prepare_build_context)
            BUILD_CONTEXT="$TEMP_BUILD_DIR"
            trap "rm -rf $TEMP_BUILD_DIR" EXIT
        fi
        
        # Build the image
        echo "📦 Building ${IMAGE_NAME}:${IMAGE_TAG}"
        docker build \
            --platform linux/amd64 \
            $NO_CACHE \
            $EXTRA_ARGS \
            --build-arg MAINTAINER="${MAINTAINER:-unknown}" \
            -f "$DOCKERFILE" \
            -t ${IMAGE_NAME}:${IMAGE_TAG} \
            "$BUILD_CONTEXT"
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to build ${IMAGE_NAME}"
            exit 1
        fi
        
        echo "✅ ${IMAGE_NAME} built successfully"
        BUILT_NOW=true
    else
        echo "📦 ${IMAGE_NAME}:${IMAGE_TAG}"
        echo "   ✅ Already exists locally - skipping build"
        echo "   (Dependencies unchanged since last build)"
    fi
    
    # Always tag as latest and push (for all cases)
    tag_as_latest "$IMAGE_NAME" "$IMAGE_TAG"
    push_to_registry "$IMAGE_NAME" "$IMAGE_TAG" "$BUILT_NOW"
}

# ============================================
# Build Functions for Each Image Type
# ============================================

build_base_image() {
    echo "📦 Base Image: obsidian-server-base:${BASE_DEPS_TAG}"
    echo "   Layer: Python $PYTHON_VERSION + dependencies"
    echo "   Platform: linux/amd64 (for HPC compatibility)"
    echo ""
    
    build_image \
        "base" \
        "obsidian-server-base" \
        "$BASE_DEPS_TAG" \
        "container/docker/Dockerfile.base" \
        "." \
        "--build-arg PYTHON_BASE_IMAGE=$PYTHON_BASE_IMAGE" \
        "--build-arg BASE_TAG=$BASE_DEPS_TAG"
    
    if [ "$TARGET" = "base" ]; then
        echo ""
        echo "Next step: Build dev or prod image"
        echo "  ./container/scripts/build_docker.sh dev   # For local development"
        echo "  ./container/scripts/build_docker.sh prod  # For HPC with Singularity"
    fi
}

build_dev_image() {
    ensure_base_image_exists

    echo "📦 Dev Image: obsidian-server-dev:${DEV_DEPS_TAG}"
    echo "   Extends: obsidian-server-base:${BASE_DEPS_TAG}"
    echo "   Adds: Packages in editable mode (--no-deps)"
    echo ""

    build_image \
        "dev" \
        "obsidian-server-dev" \
        "$DEV_DEPS_TAG" \
        "container/docker/Dockerfile.dev" \
        "" \
        "--build-arg BASE_TAG=$BASE_DEPS_TAG" \
        "--build-arg DEV_TAG=$DEV_DEPS_TAG"
    
    echo "   Packages pre-installed in editable mode"
    echo "   Use ./container/scripts/run_docker.sh for live code editing"
}

build_prod_image() {
    ensure_base_image_exists

    echo "📦 Prod Image: obsidian-server-prod:${IMAGE_TAG}"
    echo "   Extends: obsidian-server-base:${BASE_DEPS_TAG}"
    echo "   Adds: Packages baked in (--no-deps)"
    echo "   Note: Prod always rebuilds (application code may have changed)"
    echo ""

    extract_git_versions

    echo ""
    build_image \
        "prod" \
        "obsidian-server-prod" \
        "$IMAGE_TAG" \
        "container/docker/Dockerfile.prod" \
        "" \
        "--build-arg BASE_TAG=$BASE_DEPS_TAG" \
        "--build-arg PROD_TAG=$IMAGE_TAG" \
        "--build-arg OBSIDIAN_PKG_VERSION=$OBSIDIAN_PKG_VERSION" \
        "--build-arg OBSIDIAN_SHA=$OBSIDIAN_SHA"

    echo "   All code baked in - ready for Singularity conversion"
    echo "   Next: ./container/scripts/build_singularity.sh"
}

# ============================================
# Summary Display
# ============================================

show_build_summary() {
    echo ""
    echo "========================================="
    echo "✅ Build complete!"
    echo "========================================="
    echo ""

    # Show actual images in order: base, dev, prod (non-:latest tags)
    echo "Images built:"

    # Query each image type in the desired order
    local HAS_IMAGES=false
    for IMAGE_PREFIX in "obsidian-server-base" "obsidian-server-dev" "obsidian-server-prod"; do
        local IMAGE_OUTPUT=$(docker images --format "  {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" \
            --filter "reference=${IMAGE_PREFIX}:*" | \
            grep -v ":latest")
        if [ -n "$IMAGE_OUTPUT" ]; then
            echo "$IMAGE_OUTPUT"
            HAS_IMAGES=true
        fi
    done

    if [ "$HAS_IMAGES" = false ]; then
        echo "  (no images built yet)"
    fi

    echo ""
    echo "Latest tag aliases:"

    # For each image type, find what :latest points to
    for IMAGE_BASE in "obsidian-server-base" "obsidian-server-dev" "obsidian-server-prod"; do
        local LATEST_ID=$(docker images --format "{{.ID}}" ${IMAGE_BASE}:latest 2>/dev/null)
        if [ -n "$LATEST_ID" ]; then
            # Find the non-latest tag with the same ID (only from our local namespace)
            local ACTUAL_TAG=$(docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" \
                --filter "reference=${IMAGE_BASE}:*" | \
                grep -v ":latest" | \
                grep "${LATEST_ID}" | \
                awk '{print $1}' | \
                head -1)

            if [ -n "$ACTUAL_TAG" ]; then
                echo "  ${IMAGE_BASE}:latest → ${ACTUAL_TAG}"
            fi
        fi
    done

    # Show pushed images if any
    if [ ${#PUSHED_IMAGES[@]} -gt 0 ]; then
        echo ""
        echo "Pushed to registry (${DOCKER_REGISTRY}):"
        for IMAGE in "${PUSHED_IMAGES[@]}"; do
            echo "  ✓ ${IMAGE}"
        done
    fi

    # Check for and report dangling images
    local DANGLING_COUNT=$(docker images --filter "dangling=true" --filter "label=project=obsidian-server" -q 2>/dev/null | wc -l | tr -d ' ')
    if [ "$DANGLING_COUNT" -gt 0 ]; then
        echo ""
        echo "⚠️  Found ${DANGLING_COUNT} dangling image(s) from previous builds"
        echo "   This happens when rebuilding with the same tag (e.g., prod:latest)"
        echo "   Clean up with: docker image prune -f --filter label=project=obsidian-server"
    fi

    echo ""
    echo "Next steps:"
    echo "  - Test locally: ./container/scripts/run_docker.sh python --version"
    echo "  - For HPC: ./container/scripts/build_singularity.sh"
}

# ============================================
# Argument Parsing
# ============================================

PYTHON_VERSION=""
NO_CACHE=""
TARGET="base"
PUSH_TO_REGISTRY=""
IMAGE_TAG="${IMAGE_TAG:-latest}"
MAINTAINER="${MAINTAINER:-unknown}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --push)
            PUSH_TO_REGISTRY="true"
            shift
            ;;
        base|dev|prod|all)
            TARGET="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [base|dev|prod|all] [--python-version X.Y.Z] [--no-cache] [--push]"
            exit 1
            ;;
    esac
done

# ============================================
# Main Execution
# ============================================

load_env
validate_push_config
detect_python_version
load_base_image_digest
validate_requirements
validate_obsidian

# Compute dependency tags
BASE_DEPS_TAG=$(compute_base_deps_tag)
DEV_DEPS_TAG=$(compute_dev_deps_tag "$BASE_DEPS_TAG")

# Initialize array for tracking pushed images
PUSHED_IMAGES=()

echo "========================================="
echo "Building Docker Images"
echo "========================================="
echo "Python version: $PYTHON_VERSION"
echo "Target:         $TARGET"
echo "Project root:   $PROJECT_ROOT"
echo "Maintainer:     $MAINTAINER"
echo "No cache:       ${NO_CACHE:-no}"
echo ""
echo "Computed dependency tags:"
echo "  Base: $BASE_DEPS_TAG"
echo "  Dev:  $DEV_DEPS_TAG"
echo "  Prod: $IMAGE_TAG"
echo "========================================="
echo ""

# Build requested targets
case $TARGET in
    base)
        build_base_image
        ;;
    dev)
        build_base_image
        build_dev_image
        ;;
    prod)
        build_base_image
        build_prod_image
        ;;
    all)
        build_base_image
        build_dev_image
        build_prod_image
        ;;
esac

# Optional: Auto-cleanup dangling images after prod/all builds
if [ "$TARGET" = "prod" ] || [ "$TARGET" = "all" ]; then
    echo ""
    echo "🧹 Cleaning up dangling images from previous builds..."
    PRUNED=$(docker image prune -f --filter "label=project=obsidian-server" 2>&1)
    if echo "$PRUNED" | grep -q "Total reclaimed space"; then
        echo "   $(echo "$PRUNED" | grep "Total reclaimed space")"
    else
        echo "   No dangling images to clean"
    fi
fi

# Summary
show_build_summary
