"""
Modal deployment script for CS234 PPO Paragraph Retrieval pipeline.

Usage:
    pip install modal
    modal setup
    modal run run_modal.py
"""

import asyncio
import modal
import subprocess
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = "/ollama_models"
MODEL_NAME = "qwen3:8b"
JUDGE_MODEL = "qwen3:8b"  # Same model for judging (avoids slow model swap on GPU)
OLLAMA_VERSION = "0.6.5"
OLLAMA_PORT = 11434

# ---------------------------------------------------------------------------
# Modal Image: install Ollama + Python deps inside the container
# ---------------------------------------------------------------------------
ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(
        "echo 'Installing Ollama...'",
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        "echo 'Ollama installed at $(which ollama)'",
        f"mkdir -p {MODEL_DIR}",
        "echo 'build_v5_retrieval'",  # bump to force image rebuild
    )
    .env(
        {
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": MODEL_DIR,
        }
    )
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "datasets>=2.14.0",
        "transformers>=4.35.0",
    )
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/project",
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/node_modules/**",
            "checkpoints/**",
            "results/**",
            "run_modal.py",
        ],
    )
)

# ---------------------------------------------------------------------------
# Modal App + Volume for model cache
# ---------------------------------------------------------------------------
app = modal.App("cs234-ppo-paragraph-retrieval", image=ollama_image)
model_volume = modal.Volume.from_name("ollama-models-store", create_if_missing=True)


# ---------------------------------------------------------------------------
# Pipeline Runner (class-based, following Modal Ollama pattern)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A10G",
    volumes={MODEL_DIR: model_volume},
    timeout=7200,
    memory=32768,
)
class PipelineRunner:
    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        """Start Ollama server and pull model."""
        print("Starting Ollama server...")
        self.ollama_process = subprocess.Popen(["ollama", "serve"])
        print(f"Ollama PID: {self.ollama_process.pid}")

        # Wait for server
        await asyncio.sleep(10)
        print("Ollama server ready.")

        # Check if model already cached
        list_proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        current_models = list_proc.stdout if list_proc.returncode == 0 else ""
        print(f"Cached models: {current_models}")

        models_pulled = False

        model_tag = MODEL_NAME if ":" in MODEL_NAME else f"{MODEL_NAME}:latest"
        if model_tag not in current_models:
            print(f"Pulling {MODEL_NAME}...")
            pull_proc = await asyncio.create_subprocess_exec("ollama", "pull", MODEL_NAME)
            retcode = await pull_proc.wait()
            if retcode != 0:
                raise RuntimeError(f"Failed to pull {MODEL_NAME}")
            print(f"{MODEL_NAME} pulled successfully.")
            models_pulled = True
        else:
            print(f"{MODEL_NAME} already cached.")

        # Judge model is same as inference model, no extra pull needed
        print(f"Judge model: {JUDGE_MODEL} (same as inference, no swap needed)")

        if models_pulled:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, model_volume.commit)
            print("Model volume committed.")

        print("Ollama setup complete.")

    @modal.exit()
    def stop_ollama(self):
        """Terminate Ollama on shutdown."""
        if self.ollama_process and self.ollama_process.poll() is None:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
                self.ollama_process.wait()
        print("Ollama stopped.")

    @modal.method()
    def run(self, small: bool = False):
        """Run the full HotpotQA paragraph retrieval pipeline."""
        import sys

        os.chdir("/root/project")
        sys.path.insert(0, "/root/project")

        print("\n" + "=" * 70)
        print("Starting HotpotQA Paragraph Retrieval Pipeline...")
        if small:
            print("*** SMALL MODE ***")
        print("=" * 70 + "\n")

        from hotpot_pipeline import main
        main(small=small)

        # Collect results
        results = {}
        text_files = [
            "results/comparison.json",
            "results/training_results.json",
            "results/trajectories.json",
            "results/report.txt",
        ]
        for f in text_files:
            if os.path.exists(f):
                with open(f) as fh:
                    results[f] = fh.read()
                print(f"\n--- {f} ---")
                print(results[f][:2000])

        # PT file (binary -> base64)
        import base64
        pt_path = "results/hotpot_tool_selector.pt"
        if os.path.exists(pt_path):
            with open(pt_path, "rb") as fh:
                results[pt_path] = base64.b64encode(fh.read()).decode()
            print(f"\n--- {pt_path} ({os.path.getsize(pt_path)} bytes) ---")

        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(small: bool = False):
    """modal run run_modal.py [--small]"""
    print("Launching pipeline on Modal (GPU: A10G)...")
    if small:
        print("*** SMALL MODE: quick test run ***")
    print(f"Ollama {OLLAMA_VERSION} + {MODEL_NAME}\n")

    results = PipelineRunner().run.remote(small=small)

    if results:
        os.makedirs("results", exist_ok=True)
        import base64
        for filepath, content in results.items():
            if filepath.endswith(".pt"):
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(content))
            else:
                with open(filepath, "w") as f:
                    f.write(content)
            print(f"Saved: {filepath}")

    print("\nDone! Check results/ for outputs.")
