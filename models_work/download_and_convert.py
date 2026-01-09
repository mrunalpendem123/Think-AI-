import os
import subprocess
import shutil
import sys

MODEL_ID = "LiquidAI/LFM2-VL-3B"
LOCAL_DIR = "./LFM2-VL-3B"
OUTPUT_DIR = "./dist/LFM2-VL-3B-MLC"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    # Model check, assume already downloading in background or done
    if not os.path.exists(LOCAL_DIR):
        print("Model directory not found, assuming download is needed...")
        # We rely on user or previous steps for download, OR we can try again
        # But let's assume it's mostly there
        from huggingface_hub import snapshot_download
        try:
             snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, local_dir_use_symlinks=False)
        except Exception as e:
             print(f"Download warning: {e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Attempting to convert weights...")
    
    # We use 'python -m mlc_llm' instead of 'mlc_llm' directly to avoid path issues
    python_exe = sys.executable
    
    try:
        # 1. Convert Weights
        convert_cmd = f"\"{python_exe}\" -m mlc_llm convert_weight {LOCAL_DIR} --quantization q4f16_1 -o {OUTPUT_DIR}"
        run_command(convert_cmd)

        # 2. Generate Config
        # Using chatml as template. 
        # Note: If Liquid/LFM2 architecture is not supported by MLC, this step might fail or produce a config that crashes later.
        gen_config_cmd = f"\"{python_exe}\" -m mlc_llm gen_config {LOCAL_DIR} --quantization q4f16_1 --conv-template chatml -o {OUTPUT_DIR}"
        run_command(gen_config_cmd)

        print(f"Success! Model files are in {OUTPUT_DIR}")
    except Exception as e:
         print(f"Conversion failed: {e}")
         print("NOTE: If this failed due to 'Unknown model type', it means MLC LLM does not support LFM2 architecture yet.")

if __name__ == "__main__":
    main()
