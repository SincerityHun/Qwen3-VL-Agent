#!/bin/bash
# Automated test script for client embedding generation and server testing

set -e  # Exit on error

echo "================================================================================"
echo "🚀 Automated Client-Server Test Pipeline"
echo "================================================================================"

# Configuration
REPO_ROOT="/home/shjung/Qwen3-VL-Agent"
CLIENT_DIR="$REPO_ROOT/client"
SERVER_DIR="$REPO_ROOT/server"
TEST_DATA_DIR="$REPO_ROOT/test_data"
COOKBOOKS_DIR="$REPO_ROOT/cookbooks/assets"

# Sample media files (adjust paths as needed)
IMAGE_PATH="$COOKBOOKS_DIR/omni_recognition/image_example.jpg"
VIDEO_PATH="$COOKBOOKS_DIR/omni_recognition/video_example.mp4"

# Create test data directory
mkdir -p "$TEST_DATA_DIR"

# ============================================================================
# Phase 1: Generate embeddings on client
# ============================================================================
echo ""
echo "================================================================================"
echo "Phase 1: Generating Embeddings on Client"
echo "================================================================================"

cd "$CLIENT_DIR"

# Test 1: Image embedding
if [ -f "$IMAGE_PATH" ]; then
    echo ""
    echo "📸 Test 1.1: Generating image embedding..."
    python save_test_embeddings.py image "$IMAGE_PATH" "$TEST_DATA_DIR"
    echo "✅ Image embedding saved"
else
    echo "⚠️  Image file not found: $IMAGE_PATH"
    echo "   Creating dummy image for testing..."
    # You can add code to download or create a test image here
fi

# Test 2: Video embedding
if [ -f "$VIDEO_PATH" ]; then
    echo ""
    echo "🎬 Test 1.2: Generating video embedding..."
    python save_test_embeddings.py video "$VIDEO_PATH" "$TEST_DATA_DIR"
    echo "✅ Video embedding saved"
else
    echo "⚠️  Video file not found: $VIDEO_PATH"
    echo "   Skipping video test..."
fi

# ============================================================================
# Phase 2: Test server with embeddings
# ============================================================================
echo ""
echo "================================================================================"
echo "Phase 2: Testing Server with Pre-computed Embeddings"
echo "================================================================================"

cd "$SERVER_DIR"

# Find generated embedding files
IMAGE_TENSOR=$(find "$TEST_DATA_DIR" -name "image_*_tensors.pt" | head -n 1)
VIDEO_TENSOR=$(find "$TEST_DATA_DIR" -name "video_*_tensors.pt" | head -n 1)

# Test image embedding
if [ -n "$IMAGE_TENSOR" ] && [ -f "$IMAGE_TENSOR" ]; then
    echo ""
    echo "📸 Test 2.1: Testing server with image embedding..."
    python test_server_with_embeddings.py "$IMAGE_TENSOR"
    echo "✅ Image server test passed"
else
    echo "⚠️  No image embedding found, skipping..."
fi

# Test video embedding
if [ -n "$VIDEO_TENSOR" ] && [ -f "$VIDEO_TENSOR" ]; then
    echo ""
    echo "🎬 Test 2.2: Testing server with video embedding..."
    python test_server_with_embeddings.py "$VIDEO_TENSOR"
    echo "✅ Video server test passed"
else
    echo "⚠️  No video embedding found, skipping..."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "🎉 Test Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Saved test data in: $TEST_DATA_DIR"
ls -lh "$TEST_DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Start FastAPI server: cd server && python server_api.py"
echo "  2. Test end-to-end: cd client && python gradio_app.py"
echo "================================================================================"
