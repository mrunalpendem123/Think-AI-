#!/bin/bash
set -e

# Script to deploy custom Leap model for MLC integration
# Usage: ./deploy_leap_model.sh

echo ">>> Step 1: Installing leap-bundle..."
pip3 install leap-bundle

# Define Setup Variables
MODEL_NAME="LFM2-2.6B-Transcript"
LOCAL_MODEL_DIR="models_work/LFM2-2.6B-Transcript-Raw"
MLC_DIST_DIR="dist/LFM2-2.6B-Transcript-MLC"

# Ensure directories exist
mkdir -p "models_work"
mkdir -p "dist"

echo ">>> Step 2: Downloading $MODEL_NAME..."
# Note: We use the command provided. 
# If '--quantization=Q4_0' provides pre-quantized weights (like GGUF), MLC might typically need raw weights.
# We will try downloading to the local directory.
leap-bundle download "$MODEL_NAME" --quantization=Q4_0 "$LOCAL_MODEL_DIR"

echo ">>> Download Complete. Model is at: $LOCAL_MODEL_DIR"

echo ">>> Step 3: converting to MLC Format..."
# Check if mlc_llm is installed
if ! command -v mlc_llm &> /dev/null; then
    echo "mlc_llm could not be found. Please install it (e.g. pip3 install --pre --force-reinstall mlc-ai-nightly-cu121 -f https://mlc.ai/wheels) or ensure it is in your PATH."
    echo "Skipping conversion step."
    exit 1
fi

# Configuration Variables for MLC
# q4f16_1 is a standard MLC quantization compatible with WebGPU
MLC_QUANTIZATION="q4f16_1"
CONV_TEMPLATE="chatml" # Assuming ChatML template, adjust if the model uses a specific one (e.g., phi-2, llama-3)

echo "Generating MLC Config..."
mlc_llm gen_config "$LOCAL_MODEL_DIR" \
    --quantization "$MLC_QUANTIZATION" \
    --conv-template "$CONV_TEMPLATE" \
    -o "$MLC_DIST_DIR"

echo "Converting Weights..."
mlc_llm convert_weight "$LOCAL_MODEL_DIR" \
    --quantization "$MLC_QUANTIZATION" \
    -o "$MLC_DIST_DIR"

echo ">>> Conversion Complete!"
echo "Your MLC-ready model is located at: $MLC_DIST_DIR"
echo ""
echo "Next Steps:"
echo "1. Verify the 'params' folder exists in $MLC_DIST_DIR"
echo "2. Update your WebLLM store (src/stores/webllm.ts) to include this new model entry."
