#!/bin/bash
# export_for_hpc.sh - Export Docker image for HPC Singularity build
#
# Usage:
#   ./export_for_hpc.sh [output_dir]
#
# This script:
# 1. Saves the Docker prod image as a tar file
# 2. Creates instructions for building on HPC
# 3. Packages everything for transfer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/container/hpc_transfer}"

echo "========================================="
echo "Exporting Docker Image for HPC"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if prod image exists (use docker images list, more reliable on Mac)
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^obsidian-server-prod:latest$"; then
    echo "❌ Error: obsidian-server-prod:latest not found"
    echo "   Please build it first: ./container/scripts/build_docker.sh prod"
    exit 1
fi

# Save Docker image as tar
echo "📦 Saving Docker image to tar file..."
echo "   This may take a few minutes..."
docker save obsidian-server-prod:latest | gzip > "$OUTPUT_DIR/obsidian-server-prod.tar.gz"

if [ $? -eq 0 ]; then
    TAR_SIZE=$(du -h "$OUTPUT_DIR/obsidian-server-prod.tar.gz" | cut -f1)
    echo "✅ Image saved: $OUTPUT_DIR/obsidian-server-prod.tar.gz ($TAR_SIZE)"
else
    echo "❌ Failed to save image"
    exit 1
fi

# Create HPC build script
cat > "$OUTPUT_DIR/build_on_hpc.sh" <<'HPCEOF'
#!/bin/bash
#$ -N build_singularity
#$ -o logs/build_singularity.out
#$ -e logs/build_singularity.err
#$ -cwd
#$ -V
#$ -pe threads 4

# build_on_hpc.sh - Build Singularity image from Docker tar on HPC
#
# Usage:
#   Interactive (head node): ./build_on_hpc.sh
#   Submit to compute node:  qsub build_on_hpc.sh
#
# Recommended: Submit to compute node for faster builds

set -e

# Create logs directory if needed
mkdir -p logs

echo "========================================="
echo "Building Singularity Image on HPC"
echo "========================================="
echo "Job ID: ${JOB_ID:-interactive}"
echo "Host: $HOSTNAME"
echo "Date: $(date)"
echo "========================================="
echo ""

# Load singularity if not already available
if ! command -v singularity &>/dev/null; then
    echo "Loading singularity module..."
    module load singularity
fi

echo "Singularity version: $(singularity --version)"
echo ""

# Check if compressed tar file exists
if [ ! -f "obsidian-server-prod.tar.gz" ]; then
    echo "❌ Error: obsidian-server-prod.tar.gz not found in current directory"
    exit 1
fi

echo "📦 Decompressing Docker tar archive..."
gunzip obsidian-server-prod.tar.gz

if [ ! -f "obsidian-server-prod.tar" ]; then
    echo "❌ Error: Failed to decompress tar.gz"
    exit 1
fi

echo "✅ Tar archive ready"
echo ""

# Build Singularity image directly from Docker tar
echo "🔨 Building Singularity image from Docker archive..."
echo "   This may take 5-10 minutes..."
echo "   (Singularity reads directly from tar - no Docker daemon needed)"
echo ""

singularity build obsidian-server.sif docker-archive://obsidian-server-prod.tar

if [ $? -eq 0 ]; then
    SIF_SIZE=$(du -h obsidian-server.sif | cut -f1)
    echo ""
    echo "========================================="
    echo "✅ Singularity image built successfully"
    echo "========================================="
    echo "Output: obsidian-server.sif ($SIF_SIZE)"
    echo ""
echo "Test the image:"
echo "  singularity exec obsidian-server.sif python --version"
echo ""
echo "Run the API server:"
echo "  singularity exec obsidian-server.sif python -m obsidian.api.app --host 0.0.0.0 --port 8000"
else
    echo "❌ Failed to build Singularity image"
    exit 1
fi
HPCEOF

chmod +x "$OUTPUT_DIR/build_on_hpc.sh"

# Create README
cat > "$OUTPUT_DIR/README.md" <<'READMEEOF'
# HPC Singularity Build Package

This directory contains everything needed to build a Singularity image on HPC.

## Files

- `obsidian-server-prod.tar.gz` - Docker image (compressed)
- `build_on_hpc.sh` - Script to build Singularity image on HPC
- `README.md` - This file

## Instructions

### 1. Transfer to HPC

```bash
# On your local machine:
scp -r hpc_transfer/ username@hpc.example.com:~/

# Or use rsync:
rsync -avz --progress hpc_transfer/ username@hpc.example.com:~/hpc_transfer/
```

### 2. Build on HPC

**Option A: Submit to compute node (recommended)**
```bash
# SSH to HPC
ssh username@hpc.example.com

# Navigate to transfer directory
cd ~/hpc_transfer

# Submit build job
qsub build_on_hpc.sh

# Check status
qstat

# Once complete, check logs
cat logs/build_singularity.out
```

**Option B: Run interactively on head node**
```bash
cd ~/hpc_transfer
module load singularity
./build_on_hpc.sh
```

This will create `obsidian-server.sif` (~600MB).

**Note:** Option A (qsub) is recommended as building can take 5-10 minutes
and benefits from running on a compute node with more resources.

### 3. Test the Image

```bash
singularity exec obsidian-server.sif python --version
# Should print: Python 3.10.19

singularity exec obsidian-server.sif python -c "import obsidian; print(obsidian.__version__)"
# Should work without errors
```

### 4. Use in PBS Jobs

Move the .sif file to your desired location:

```bash
mv obsidian-server.sif ~/containers/
```

Then in your PBS scripts, set:

```bash
export USE_CONTAINER=1
export CONTAINER_IMAGE=~/containers/obsidian-server.sif
```

## Notes

- The tar file is ~600MB compressed, ~2GB uncompressed
- Building the Singularity image takes ~5-10 minutes
- The final .sif file is ~600MB
- Singularity images are immutable and portable across HPC systems

## Troubleshooting

**"singularity command not found"**
- Load the module: `module load singularity`
- Check available modules: `module avail singularity`

**"Failed to load Docker image"**
- Ensure tar file is complete: `md5sum obsidian-server-prod.tar.gz`
- Check disk space: `df -h .`

**Build is very slow**
- This is normal, be patient
- Building on a compute node may be faster than login node

READMEEOF

echo ""
echo "========================================="
echo "✅ Export complete!"
echo "========================================="
echo ""
echo "Package contents:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Total size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Transfer to HPC:"
echo "     scp -r $OUTPUT_DIR username@hpc:/path/to/destination"
echo ""
echo "  2. On HPC, run:"
echo "     cd /path/to/destination/$(basename $OUTPUT_DIR)"
echo "     module load singularity"
echo "     ./build_on_hpc.sh"
echo ""
echo "  3. Move .sif file to permanent location and use in PBS jobs"
