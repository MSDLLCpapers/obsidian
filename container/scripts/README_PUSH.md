# Docker Registry Push Configuration

This guide explains how to use the `--push` feature in `build_docker.sh` to automatically tag and push images to your private Docker registry.

## Setup

1. **Create a `.env` file** in the `container/scripts/` directory:
   ```bash
   cd container/scripts
   cp .env.example .env
   ```

2. **Edit `.env`** with your registry settings:
   ```bash
   # Docker Registry Configuration
   DOCKER_REGISTRY=dre-dev.dock.merck.com
   IMAGE_TAG=0.0.1
   ```

3. **Note:** The `.env` file is gitignored and will not be committed to the repository.

## Usage

### Build without pushing (default behavior)
```bash
./container/scripts/build_docker.sh base
./container/scripts/build_docker.sh dev
./container/scripts/build_docker.sh prod
./container/scripts/build_docker.sh all
```

### Build and push to registry
```bash
# Build and push base image
./container/scripts/build_docker.sh base --push

# Build and push dev image
./container/scripts/build_docker.sh dev --push

# Build and push prod image
./container/scripts/build_docker.sh prod --push

# Build all and push to registry
./container/scripts/build_docker.sh all --push
```

### Combined with other flags
```bash
# Build without cache and push
./container/scripts/build_docker.sh base --no-cache --push

# Build with specific Python version and push
./container/scripts/build_docker.sh base --python-version 3.10.19 --push
```

## What happens when you use --push

For each image built (e.g., `obsidian-server-base:latest`), the script will:

1. **Tag** the image with your registry path and version:
   ```
   obsidian-server-base:latest → dre-dev.dock.merck.com/obsidian-server-base:0.0.1
   ```

2. **Push** the tagged image to your private registry:
   ```
   docker push dre-dev.dock.merck.com/obsidian-server-base:0.0.1
   ```

3. **Remove** the local registry-tagged image to save disk space:
   ```
   docker rmi dre-dev.dock.merck.com/obsidian-server-base:0.0.1
   ```

   Note: The local `obsidian-server-base:latest` image is kept for further use.

## Security

- The `.env` file is excluded from version control via `.gitignore`
- The registry URL is never hardcoded in the script
- Only `.env.example` (with placeholder values) is committed to the repository

## Troubleshooting

### Error: "--push requires a .env file"
Create a `.env` file in `container/scripts/` with your registry configuration.

### Error: ".env file must define DOCKER_REGISTRY and IMAGE_TAG"
Ensure your `.env` file contains both required variables:
```bash
DOCKER_REGISTRY=dre-dev.dock.merck.com
IMAGE_TAG=0.0.1
```

### Push fails with authentication error
Ensure you're logged into your Docker registry:
```bash
docker login dre-dev.dock.merck.com
```
