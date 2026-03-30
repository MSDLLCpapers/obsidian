# Container Setup for Obsidian Server

This directory contains containerization setup for building and deploying the Obsidian HTTP API server.

## Overview

The containerization strategy supports three execution modes:

1. **Direct Python** (fastest) - Use conda environment directly, no containers
2. **Development Mode** (Docker with volumes) - Local development with live code changes
3. **Production Mode** (Singularity on HPC) - Fully reproducible HPC runs

## Quick Start

### Prerequisites

- **Local (Mac)**: Docker Desktop installed
- **HPC**: Singularity/Apptainer available (usually via `module load singularity`)
- **Both**: Conda environment "obsidian" with all dependencies installed

### 1. Generate Requirements File

First, generate the `requirements_docker.txt` file from your current conda environment:

```bash
# Activate your conda environment
conda activate obsidian

# Generate requirements (will parse obsidian's pyproject.toml)
python container/generate_requirements.py
```

This creates:

- `container/docker/requirements_docker.txt` - Pinned dependencies for base image
- `container/docker/requirements_docker_dev.txt` - Pinned dev dependencies for dev image
- `container/requirements_report.json` - Detailed report of all decisions

**What it does:**

- Reads `pyproject.toml` to find core + LLM extras dependencies
- Reads `pyproject.toml` to find dev extras dependencies (pytest, flake8, etc.)
- Pins exact versions from your current conda environment
- Generates two files:
  - `requirements_docker.txt` - for base image (core + LLM)
  - `requirements_docker_dev.txt` - for dev image (testing tools)

**Options:**

```bash
# Use custom obsidian path
python container/generate_requirements.py --obsidian-path /path/to/obsidian

# Upgrade to latest patch versions (instead of pinning current)
python container/generate_requirements.py --strategy upgrade-patch

# Override Python version
python container/generate_requirements.py --python-version 3.10.19
```

### 2. Build Docker Images (Local)

```bash
# Build base image (dependencies only)
./container/scripts/build_docker.sh base

# Build dev image (for local development)
./container/scripts/build_docker.sh dev

# Build production image (for HPC)
./container/scripts/build_docker.sh prod

# Build all images
./container/scripts/build_docker.sh all
```

### 3. Test Locally

```bash
# Run a command in the container
./container/scripts/run_docker.sh python --version

# Run the API server (quick shortcuts)
./container/scripts/run_docker.sh --api              # Starts on port 8000
./container/scripts/run_docker.sh --api --port 8080  # Custom port
./container/scripts/run_docker.sh --dash             # Start Dash app on port 8050

# Session persistence (for API/Dash apps)
# By default, sessions are stored in ~/.obsidian on your host machine
./container/scripts/run_docker.sh --api --session-dir ~/my-sessions  # Custom location
./container/scripts/run_docker.sh --api --session-dir none           # Ephemeral (no persistence)

# Or run directly
./container/scripts/run_docker.sh python -m obsidian.api.app
./container/scripts/run_docker.sh python script.py  # Any Python script

# Start interactive shell
./container/scripts/run_docker.sh bash
```

### 4. Build Singularity Image (HPC)

On HPC, convert the Docker image to Singularity:

```bash
# Option 1: Build from local Docker image
./container/scripts/build_singularity.sh

# Option 2: Pull from Docker registry
./container/scripts/build_singularity.sh --from-registry docker://username/obsidian-server-prod:latest
```

This creates `container/obsidian-server.sif`.

### 5. Run on HPC

```bash
# Run the API server with Singularity
singularity exec container/obsidian.sif python -m obsidian.api.app --host 0.0.0.0 --port 8000

# Or use in a PBS script
# See container/scripts/build_singularity.sh for examples
```

## Detailed Workflows

### Development Workflow (Local Mac)

For rapid iteration with live code changes:

```bash
# 1. Build dev image (one-time, or when dependencies change)
./container/scripts/build_docker.sh dev

# 2. Edit code in your editor (VSCode, etc.)
# Changes to obsidian and acquisition_functions are immediately visible

# 3. Run tests/API server
./container/scripts/run_docker.sh python -m pytest
./container/scripts/run_docker.sh --api           # API server shortcut
./container/scripts/run_docker.sh --dash          # Dash app shortcut

# 4. When dependencies change, rebuild
python container/generate_requirements.py
./container/scripts/build_docker.sh dev
```

**Volume mounts:**

- Repository root → `/opt/obsidian` (editable mode)
- Working directory: `/opt/obsidian`
- Code changes are immediately visible

**Shortcuts:**

- `--api` - Start API server on port 8000
- `--dash` - Start Dash app on port 8050
- `--port N` - Override default port

### Production Workflow (HPC)

For reproducible large-scale runs:

```bash
# On your Mac (one-time setup):
# 1. Ensure everything works
python container/generate_requirements.py
./container/scripts/build_docker.sh prod

# 2. Test locally
./container/scripts/run_docker.sh python -m obsidian.api.app --help

# 3. Build Singularity image
./container/scripts/build_singularity.sh

# 4. Transfer to HPC
scp container/obsidian.sif hpc:~/obsidian/container/

# On HPC:
# 5. Load Singularity module (if needed)
module load singularity

# 6. Test the image
singularity exec container/obsidian.sif python --version

# 7a. Run API server as a persistent service (recommended)
./container/scripts/run_singularity.sh start --api
./container/scripts/run_singularity.sh status  # Check if running
./container/scripts/run_singularity.sh stop    # Stop when done

# 7b. Or run directly (non-persistent)
singularity exec container/obsidian.sif python -m obsidian.api.app --host 0.0.0.0 --port 8000
```

### Incremental Updates

**Scenario 1: Only Python code changed (obsidian)**

```bash
# For development (no rebuild needed - using volumes):
./container/scripts/run_docker.sh python -m obsidian.api.app

# For production (rebuild prod image):
./container/scripts/build_docker.sh prod
./container/scripts/build_singularity.sh
# Transfer to HPC
```

**Scenario 2: Dependencies changed**

```bash
# 1. Update dependencies in obsidian (edit pyproject.toml)
# 2. Install in conda environment
conda activate obsidian
cd ../obsidian && pip install -e . && cd -

# 3. Regenerate requirements
python container/generate_requirements.py

# 4. Rebuild images
./container/scripts/build_docker.sh all

# 5. For HPC, rebuild and transfer Singularity image
./container/scripts/build_singularity.sh
```

**Scenario 3: Python version upgrade**

```bash
# 1. Update conda environment
conda create -n obsidian python=3.11
conda activate obsidian
# Install packages...

# 2. Generate requirements with new Python version
python container/generate_requirements.py --python-version 3.11

# 3. Rebuild everything
./container/scripts/build_docker.sh all
```


### Running Singularity as a Service (HPC)

For long-running API servers on HPC, use Singularity instances:

```bash
# Start API server as a persistent background service
./container/scripts/run_singularity.sh start --api

# With custom options
./container/scripts/run_singularity.sh start --api --port 8080
./container/scripts/run_singularity.sh start --api --session-dir ~/my-sessions

# Manage the service
./container/scripts/run_singularity.sh status    # Check if running
./container/scripts/run_singularity.sh logs      # View logs
./container/scripts/run_singularity.sh stop      # Stop the service
```

**Why use instances instead of \`singularity exec\`?**

- Runs as a persistent background service
- Survives terminal disconnection
- Better for long-running API servers
- Easier to manage (start/stop/status commands)

**Session persistence:**

- Sessions stored in \`~/.obsidian\` by default (persists across restarts)
- Customize with \`--session-dir\` flag
- Set to \`none\` for ephemeral sessions

## Files and Structure

```
container/
├── README.md                          # This file
├── generate_requirements.py           # Smart dependency generator
├── requirements_report.json           # Generation report (created by script)
├── obsidian.sif                       # Singularity image (created by script)
│
├── docker/
│   ├── Dockerfile.base               # Base: Python + core dependencies
│   ├── Dockerfile.dev                # Dev: Base + dev dependencies
│   ├── Dockerfile.prod               # Prod: Base + obsidian (no dev tools)
│   ├── requirements_docker.txt       # Pinned core dependencies (generated)
│   ├── requirements_docker_dev.txt   # Pinned dev dependencies (generated)
│   └── python-3.10-slim-digest.lock  # Pinned Python base image digests
│
├── scripts/
│   ├── build_docker.sh               # Build Docker images
│   ├── build_singularity.sh          # Convert to Singularity
│   ├── run_docker.sh                 # Run commands in Docker
│   ├── run_singularity.sh            # Run Singularity as HPC service
│   └── update_base_image_digest.sh   # Update Python base image locks
│
└── docker-compose.yml                # Docker Compose for local dev
```

## Environment Variables

### PBS Scripts

- `USE_CONTAINER=0|1` - Enable container execution (default: 0)
- `CONTAINER_IMAGE` - Path to Singularity image (default: container/obsidian-server.sif)

Example:
```bash
# Use container with default image
qsub -v USE_CONTAINER=1 run_benchmark.pbs /path/to/job

# Use container with custom image
qsub -v USE_CONTAINER=1,CONTAINER_IMAGE=/scratch/my.sif run_benchmark.pbs /path/to/job
```

### Container Environment

For reproducibility, the following are set in containers:
- `PYTHONHASHSEED=0` - Deterministic Python hashing
- `PYTHONUNBUFFERED=1` - Immediate output flushing
- CPU-only PyTorch (via `+cpu` wheels)

**Thread Parallelization:**
The containers do NOT hardcode thread limits. Instead:
- **On HPC**: Singularity automatically inherits environment variables from the PBS script
  - PBS script sets `OMP_NUM_THREADS=$NSLOTS` (e.g., 16 threads)
  - Container respects these settings
- **On local Mac**: Python code uses `ncpu - 1` by default (application logic)
- This ensures **consistent behavior** with non-containerized execution

## Troubleshooting

### "Requirements file not found"

```bash
# Generate it first:
python container/generate_requirements.py
```

### "Obsidian directory not found"

Ensure obsidian is in `../obsidian` relative to this project:
```bash
ls ../obsidian/pyproject.toml  # Should exist
```

### "Docker image not found"

Build the image first:
```bash
./container/scripts/build_docker.sh base  # or dev, or prod
```

### "Singularity command not found" (HPC)

Load the module:
```bash
module load singularity
# or
module load apptainer
```

### "Container image not found" (HPC PBS job)

Check the path in your PBS submission:
```bash
# Make sure the image exists:
ls container/obsidian-server.sif

# Use absolute path if needed:
qsub -v USE_CONTAINER=1,CONTAINER_IMAGE=$PWD/container/obsidian-server.sif ...
```

### "Import error: obsidian not found" (in container)

For dev mode, ensure obsidian is mounted:
```bash
# Check if obsidian exists
ls ../obsidian

# The run_docker.sh script should mount it automatically
./container/scripts/run_docker.sh python -c "import obsidian; print(obsidian.__file__)"
```

For prod mode, obsidian should be baked in. If not, rebuild:
```bash
./container/scripts/build_docker.sh prod
```

## Thread Parallelization Behavior

The container images **do not hardcode thread limits**, ensuring flexibility across environments:

### HPC with PBS/UGE

When running with Singularity on HPC:
```bash
# PBS script sets thread count based on allocated cores
export OMP_NUM_THREADS=$NSLOTS  # e.g., 16
export MKL_NUM_THREADS=$NSLOTS

# Singularity automatically inherits these from host
singularity exec container.sif python -m obsidian.api.app
```

Result: Container uses **$NSLOTS threads** (consistent with non-container execution)

### Local Mac with Docker

When running locally:
```bash
./container/scripts/run_docker.sh python -m obsidian.api.app
```

Result: Python code uses **ncpu - 1 threads** by default (application logic)

### Verification

Check thread settings inside container:
```bash
# On HPC after PBS sets variables
singularity exec container.sif bash -c 'echo $OMP_NUM_THREADS'

# On Mac with Docker
./container/scripts/run_docker.sh bash -c 'python -c "import os; print(os.cpu_count())"'
```

## Tips and Best Practices

### When to use containers vs conda

**Use containers when:**
- Need exact reproducibility across platforms
- Running large HPC jobs that must be reproducible
- Sharing results that others need to replicate
- Testing with different Python/package versions

**Use direct conda when:**
- Quick local testing and iteration
- Debugging and development
- Rapid prototyping
- Performance is critical (containers have slight overhead)

### Container image management

**Local development:**
- Keep `obsidian-server-dev:latest` image updated with dependencies
- Rebuild only when dependencies change (weekly/monthly)
- Use volume mounts for code changes (no rebuild needed)

**HPC production:**
- Build Singularity image once per major experiment
- Version your images: `obsidian-server-v1.sif`, `v2.sif`, etc.
- Store images in stable location (not scratch space)
- Document which image was used for which results

### Dependency management

- Regenerate `requirements_docker.txt` when obsidian dependencies change
- Use `--strategy pin-current` for reproducibility
- Use `--strategy upgrade-patch` for longevity before major runs
- Commit `requirements_docker.txt` to git for reproducibility

## Advanced Topics

### Using Docker Compose

For more complex local workflows:

```bash
cd container

# Build and start dev environment
docker-compose up -d dev

# Run commands
docker-compose run dev python -m pytest
docker-compose run dev bash

# Stop
docker-compose down
```

### Custom base images

To use a specific Python version:

```bash
./container/scripts/build_docker.sh base --python-version 3.10.19
```

### Pushing to Docker Registry

To share images via Docker Hub or private registry:

```bash
# Tag image
docker tag obsidian-server-prod:latest username/obsidian-server-prod:v1.0

# Push to registry
docker push username/obsidian-server-prod:v1.0

# On HPC, pull directly to Singularity
singularity pull obsidian-server-v1.0.sif docker://username/obsidian-server-prod:v1.0
```

## Support

For issues or questions:
1. Check this README
2. Review the scripts in `container/scripts/` for detailed comments
3. Check PBS script logs for container execution details
4. Contact the maintainer

## Future Improvements

Potential enhancements:
- [ ] Automated testing of container builds in CI/CD
- [ ] Multi-architecture support (ARM64 for Apple Silicon)
- [ ] GPU-enabled containers (if needed)
- [ ] Automated image versioning and tagging
- [ ] Container registry automation
