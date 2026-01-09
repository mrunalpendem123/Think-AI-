import os
import subprocess
from huggingface_hub import snapshot_download

MODEL_ID = "LiquidAI/LFM2-VL-3B"
LOCAL_DIR = "./LFM2-VL-3B"
OUTPUT_DIR = "./dist/LFM2-VL-3B-MLC"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    print(f"Downloading {MODEL_ID}...")
    # Download only essential files to save space/time if possible
    # But for conversion we usually need everything (safetensors, config, tokenizer)
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, local_dir_use_symlinks=False)
    print("Download complete.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Converting weights...")
    # Using q4f16_1 quantization as standard for WebLLM
    # We might need to specify --model-type if it's not auto-detected correctly.
    # For now, we rely on auto-detection or error out.
    convert_cmd = f"mlc_llm convert_weight {LOCAL_DIR} --quantization q4f16_1 -o {OUTPUT_DIR}"
    run_command(convert_cmd)

    print("Generating config...")
    # Using chatml as a fallback template, though Liquid might need custom
    gen_config_cmd = f"mlc_llm gen_config {LOCAL_DIR} --quantization q4f16_1 --conv-template chatml -o {OUTPUT_DIR}"
    run_command(gen_config_cmd)

    print(f"Success! Model files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
