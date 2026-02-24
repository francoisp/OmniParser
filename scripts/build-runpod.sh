#!/bin/bash
# Build and optionally push the RunPod serverless image
set -e

REGISTRY="${REGISTRY:-docker.io}"
IMAGE_NAME="${IMAGE_NAME:-zparser-runpod}"
VERSION="${VERSION:-latest}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "Building RunPod serverless image: ${FULL_IMAGE}"
echo "Note: If local weights/ are present they'll be copied in; otherwise models download during build (~2.5GB)"
echo ""

# Build from repo root
cd "$(dirname "$0")/.." || exit 1

docker build \
  -f Dockerfile.runpod \
  -t "${FULL_IMAGE}" \
  .

echo ""
echo "Build complete: ${FULL_IMAGE}"
echo ""
echo "To push:"
echo "  docker push ${FULL_IMAGE}"
echo ""
echo "To test locally:"
echo "  docker run --gpus all ${FULL_IMAGE}"
echo ""
echo "RunPod deployment:"
echo "  1. Push to Docker Hub or your registry"
echo "  2. Create Serverless Endpoint in RunPod with this image"
echo "  3. Container Disk: 20GB, GPU: A4000/A5000/A6000"
echo "  4. Test with runsync:"
echo '     curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \'
echo '       -H "Authorization: Bearer <api-key>" \'
echo '       -H "Content-Type: application/json" \'
echo "       -d '{\"input\": {\"base64_image\": \"...\"}}'"""
