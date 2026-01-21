#!/bin/bash
set -e

# Configuration
MODEL_ID="LiquidAI/LFM2.5-1.2B-Instruct"
LOCAL_DIR="models_work/LFM2.5-1.2B-Instruct"
MLC_DIST_DIR="dist/LFM2.5-1.2B-Instruct-MLC"
QUANTIZATION="q4f16_1"

echo ">>> Step 0: Environment Setup"
# Try installing mlc_llm if not present. 
# Note: This might fail on Python 3.13 if wheels aren't ready. 
# We explicitly look for nightly wheels which might have broader support.
if ! command -v mlc_llm &> /dev/null; then
    echo "Installing MLC LLM..."
    # Attempt install for Mac (using nightly for best arch support)
    ./venv/bin/pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu || ./venv/bin/pip install mlc-llm
fi

echo ">>> Step 1: Download Model ($MODEL_ID)"
mkdir -p models_work
if [ ! -d "$LOCAL_DIR" ]; then
    # Use hf downloader
    ./venv/bin/huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR" --local-dir-use-symlinks False
else
    echo "Model already downloaded at $LOCAL_DIR"
fi

echo ">>> Step 2: Generate MLC Config"
mkdir -p dist
./venv/bin/python3 -m mlc_llm gen_config "$LOCAL_DIR" \
    --quantization "$QUANTIZATION" \
    --conv-template chatml \
    -o "$MLC_DIST_DIR"

echo ">>> Step 3: Convert Weights"
./venv/bin/python3 -m mlc_llm convert_weight "$LOCAL_DIR" \
    --quantization "$QUANTIZATION" \
    -o "$MLC_DIST_DIR"

echo ">>> Step 4: Compile WebWASM Library"
# We need to compile the model library to WASM for it to run in the browser
# This produces the .wasm file
WASM_FILE="$MLC_DIST_DIR/LFM2.5-1.2B-Instruct-$QUANTIZATION-webgpu.wasm"
./venv/bin/python3 -m mlc_llm compile "$MLC_DIST_DIR" \
    --device webgpu \
    -o "$WASM_FILE"

echo ">>> Done!"
echo "Model Artifacts located in: $MLC_DIST_DIR"
echo "WASM Binary: $WASM_FILE"
