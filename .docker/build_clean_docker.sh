#!/bin/bash

# Check if a version argument was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a base version (e.g., ./build_clean_docker.sh v1.1.2)"
    exit 1
fi

# Capture the base version from the first argument
BASE_VERSION=$1

# Generate the timestamp in YYYYMMDD format
TIMESTAMP=$(date +%Y%m%d)

# Combine them with an underscore
TAG_VERSION="${BASE_VERSION}_${TIMESTAMP}"

# Export or print the result
echo "Generated Tag: $TAG_VERSION"
#TAG_VERSION="v1.1.3_$(date +%Y%m%d)"

#docker compose build --no-cache && docker tag docker-crawler:latest "docker-crawler:$TAG_VERSION"
cd ..
docker  build --no-cache -f .docker/Dockerfile --tag "docker-crawler:$TAG_VERSION" .
cd .docker/
#docker compose build --no-cache && docker tag docker-crawler:latest "docker-crawler:$TAG_VERSION" 
