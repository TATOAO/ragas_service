#!/bin/bash
# bash ./docker/build_and_pack_image.sh

# exit when error
set -e

# Build with cache optimization
docker-compose -f ./docker/docker-compose.yml build --no-cache=false --progress=plain

# with git commit hash and timestamp
commit_hash=$(git rev-parse --short HEAD)
timestamp=$(date +%Y%m%d_%H%M)
mkdir -p ./docker/images
docker save -o ./docker/images/app_image_${timestamp}_${commit_hash}.tar ragas-service:latest