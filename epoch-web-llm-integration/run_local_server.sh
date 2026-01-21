#!/bin/bash
set -e

# Create virtual env if not exists
if [ ! -d "venv_server" ]; then
    echo "Creating virtual environment for server..."
    python3 -m venv venv_server
fi

# Activate
source venv_server/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install fastapi uvicorn transformers torch accelerate protobuf

# Run server
echo "Starting Local LLM Server..."
python3 local_llm_server.py
