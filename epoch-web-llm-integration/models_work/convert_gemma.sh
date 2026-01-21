#!/bin/bash
set -e

# Create venv
echo "Recreating virtual environment with Python 3.11..."
rm -rf venv
/opt/homebrew/bin/python3.11 -m venv venv

# Define paths
VENV_PYTHON="./venv/bin/python"
VENV_PIP="./venv/bin/pip"

# Install dependencies
echo "Upgrading pip..."
$VENV_PIP install --upgrade pip

echo "Installing mlc-llm..."
$VENV_PIP install --pre mlc-llm mlc-ai-nightly -f https://mlc.ai/wheels

# Install git-lfs just in case
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs not found, checking brew..."
    if command -v brew &> /dev/null; then
         brew install git-lfs || echo "Brew install failed"
    fi
    git lfs install || echo "git lfs install failed or skipped"
fi

# Clone the model
echo "Cloning google/translategemma-4b-it..."
if [ ! -d "translategemma-4b-it" ]; then
    git clone https://huggingface.co/google/translategemma-4b-it
else
    echo "Directory translategemma-4b-it already exists, skipping clone."
fi

# Define mlc command
MLC_CMD="$VENV_PYTHON -m mlc_llm"

# Generate Config
echo "Generating Config..."
$MLC_CMD gen_config ./translategemma-4b-it \
    --quantization q4f16_1 \
    --conv-template gemma_it \
    --device webgpu \
    -o translategemma-4b-it-q4f16_1-MLC/

# Convert Weights
echo "Converting Weights..."
$MLC_CMD convert_weight ./translategemma-4b-it \
    --quantization q4f16_1 \
    -o translategemma-4b-it-q4f16_1-MLC/

# Compile WASM
echo "Compiling WASM..."
$MLC_CMD compile ./translategemma-4b-it-q4f16_1-MLC/mlc-chat-config.json \
    --device webgpu \
    -o translategemma-4b-it-q4f16_1-MLC/translategemma-4b-it-q4f16_1-webgpu.wasm

echo "Moving artifacts to public folder..."
rm -rf ../public/translategemma-4b-it-q4f16_1-MLC
mv translategemma-4b-it-q4f16_1-MLC ../public/

echo "Done! Model is ready in public/translategemma-4b-it-q4f16_1-MLC/"
