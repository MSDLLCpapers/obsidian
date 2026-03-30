#!/bin/bash
# Update Python base image digest lock file
#
# This script fetches the SHA256 digest for a specific Python version and adds/updates
# it in the lock file. This ensures reproducible builds across different machines and times.
#
# Usage:
#   ./update_base_image_digest.sh 3.10.20          # Add new version
#   ./update_base_image_digest.sh 3.10.19 --force  # Update existing version
#
# The lock file tracks:
# - Exact image digest (immutable reference)
# - Fetch timestamp with timezone
# - Only Python 3.10.17+ on Debian bookworm (stable)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

VERSION=$1
FORCE=${2:-}
LOCK_FILE="container/docker/python-3.10-slim-digest.lock"

# Validate arguments
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [--force]"
    echo ""
    echo "Examples:"
    echo "  $0 3.10.20          # Add new version"
    echo "  $0 3.10.19 --force  # Update existing"
    exit 1
fi

# Validate version format and minimum version (>= 3.10.17)
if [[ ! "$VERSION" =~ ^3\.10\.([0-9]+)$ ]]; then
    echo "❌ Error: Invalid version format: $VERSION"
    echo "   Expected format: 3.10.X"
    exit 1
fi

PATCH_VERSION="${BASH_REMATCH[1]}"
if [ "$PATCH_VERSION" -lt 17 ]; then
    echo "❌ Error: Version must be >= 3.10.17"
    echo "   Requested: $VERSION"
    exit 1
fi

# Check if lock file is read-only and make writable if needed
FILE_WAS_READONLY=false
if [ -f "$LOCK_FILE" ] && [ ! -w "$LOCK_FILE" ]; then
    echo "🔓 Lock file is read-only, temporarily making it writable for updates..."
    chmod 644 "$LOCK_FILE"
    FILE_WAS_READONLY=true
fi

# Check if version already exists in lock file
VERSION_EXISTS=false
if [ -f "$LOCK_FILE" ] && grep -q "^${VERSION}=" "$LOCK_FILE"; then
    VERSION_EXISTS=true
    if [ "$FORCE" != "--force" ]; then
        echo "⚠️  Version $VERSION already exists in lock file:"
        grep "^${VERSION}=" "$LOCK_FILE"
        echo ""
        echo "Use --force to overwrite:"
        echo "  $0 $VERSION --force"

        # Restore read-only if needed
        if [ "$FILE_WAS_READONLY" = true ]; then
            chmod 444 "$LOCK_FILE"
        fi
        exit 1
    fi
    echo "🔄 Forcing update of existing version $VERSION"
    echo "   Previous entry will be preserved as comment for restoration"
    echo ""
fi

# Query digest from registry without pulling the image
IMAGE="python:${VERSION}-slim-bookworm"
echo "🔍 Querying digest from Docker Hub..."

# Use manifest inspect to get digest without downloading image
MANIFEST_OUTPUT=$(docker manifest inspect "$IMAGE" 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Error: Could not fetch manifest for $IMAGE"
    echo "   Make sure the image exists on Docker Hub"
    echo "   Error: $MANIFEST_OUTPUT"
    exit 1
fi

# Extract digest from manifest (format: sha256:abc123...)
SHA256_HASH=$(echo "$MANIFEST_OUTPUT" | grep -o '"digest": *"sha256:[a-f0-9]*"' | head -1 | grep -o 'sha256:[a-f0-9]*')

if [ -z "$SHA256_HASH" ]; then
    echo "❌ Error: Could not parse digest from manifest"
    exit 1
fi

# Validate digest format
if [[ ! "$SHA256_HASH" =~ ^sha256:[a-f0-9]{64}$ ]]; then
    echo "❌ Error: Invalid digest format: $SHA256_HASH"
    exit 1
fi

# Construct full image reference with tag and digest
# Format: python:3.10.19-slim-bookworm@sha256:abc123...
DIGEST="${IMAGE}@${SHA256_HASH}"

# Get timestamp with timezone
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S %Z')

# Create lock file if it doesn't exist
if [ ! -f "$LOCK_FILE" ]; then
    cat > "$LOCK_FILE" <<'EOF'
# Python 3.10 slim-bookworm (Debian 12) base image digests
# Architecture: linux/amd64
# Do not edit manually - use container/scripts/update_base_image_digest.sh
#
# Format:
# # Fetched: YYYY-MM-DD HH:MM:SS TZ
# VERSION=IMAGE@DIGEST
#
# Only versions >= 3.10.17 are supported

EOF
fi

# Update lock file
if [ "$VERSION_EXISTS" = true ]; then
    # Replace existing entry (--force mode)
    # Keep only the most recent old entry (remove any previous REPLACED entries)
    TEMP_FILE=$(mktemp)

    # Extract current entry for preservation
    OLD_FETCH_TIME=$(grep -B1 "^${VERSION}=" "$LOCK_FILE" | grep "^# Fetched:" | sed 's/^# //')
    OLD_DIGEST=$(grep "^${VERSION}=" "$LOCK_FILE")

    # Copy file, removing:
    # 1. Current entry and its fetch comment
    # 2. Any previous REPLACED comments for this version
    awk -v ver="$VERSION" '
        # Skip REPLACED blocks for this version
        /^# REPLACED:/ {
            replaced_comment=$0
            getline
            if ($0 ~ "^# "ver"=") {
                # This is a REPLACED entry for our version, skip it and the separator
                getline  # Skip the separator line (#)
                next
            } else {
                # Not our version, keep the REPLACED comment and line
                print replaced_comment
                print $0
                next
            }
        }
        # Skip current active entry for this version
        /^# Fetched:/ {
            comment=$0
            getline
            if ($0 !~ "^"ver"=") {
                print comment
                print $0
            }
            next
        }
        { print }
    ' "$LOCK_FILE" > "$TEMP_FILE"

    # Add old entry as comment (only keep one previous version)
    echo "# REPLACED: $OLD_FETCH_TIME" >> "$TEMP_FILE"
    echo "# ${OLD_DIGEST}" >> "$TEMP_FILE"
    echo "#" >> "$TEMP_FILE"

    # Add new entry
    echo "# Fetched: $TIMESTAMP" >> "$TEMP_FILE"
    echo "${VERSION}=${DIGEST}" >> "$TEMP_FILE"

    mv "$TEMP_FILE" "$LOCK_FILE"
else
    # Append new entry
    echo "" >> "$LOCK_FILE"
    echo "# Fetched: $TIMESTAMP" >> "$LOCK_FILE"
    echo "${VERSION}=${DIGEST}" >> "$LOCK_FILE"
fi

# Set lock file to read-only (444)
chmod 444 "$LOCK_FILE"
echo "🔒 Lock file set to read-only (444) to prevent accidental modification"

echo ""
echo "========================================="
echo "✅ Successfully updated lock file"
echo "========================================="
echo "Version:    $VERSION"
echo "Full ref:   $DIGEST"
echo "SHA256:     ${SHA256_HASH#sha256:}"
echo "Timestamp:  $TIMESTAMP"
echo "Lock file:  $LOCK_FILE"
echo ""
if [ "$VERSION_EXISTS" = true ]; then
    echo "Note: Previous entry preserved as comment in lock file"
    echo "      Only the most recent old version is kept"
    echo "      For full history, use: git log container/docker/python-3.10-slim-digest.lock"
fi
echo ""
echo "Next steps:"
echo "  1. Review: cat $LOCK_FILE"
echo "  2. Rebuild base image: ./container/scripts/build_docker.sh base"
echo "  3. Commit lock file to git for reproducibility"
echo ""
echo "Lock file is read-only (444). To modify manually:"
echo "  chmod 644 $LOCK_FILE  # Edit file"
echo "  chmod 444 $LOCK_FILE  # Restore read-only"
