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
        "sentence-transformers>=2.2.0",
    )
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/project",
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/node_modules/**",
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
    def run(self, small: bool = False, dataset: str = "2wiki",
            resume: bool = False, blind: bool = False, ppo_only: bool = False,
            bc_oracle: bool = False, run_name: str = ""):
        """Run the paragraph retrieval pipeline (HotpotQA or 2WikiMultiHopQA)."""
        import sys, glob

        os.chdir("/root/project")
        sys.path.insert(0, "/root/project")

        output_dir = f"results_{run_name}" if run_name else "results"
        ckpt_dir = f"checkpoints_blind_{run_name}" if run_name else ("checkpoints_blind" if blind else "checkpoints")
        vol_ckpt_key = f"cs234_checkpoints_blind_{run_name}" if run_name else ("cs234_checkpoints_blind" if blind else "cs234_checkpoints")
        vol_results_key = f"cs234_results_{run_name}" if run_name else ("cs234_results_blind" if blind else "cs234_results")

        ds_label = "2WikiMultiHopQA" if dataset == "2wiki" else "HotpotQA"
        print("\n" + "=" * 70)
        print(f"Starting {ds_label} Paragraph Retrieval Pipeline...")
        if small:
            print("*** SMALL MODE ***")
        if blind:
            print("*** BLIND TITLES MODE ***")
        if ppo_only:
            print("*** PPO-ONLY MODE (skip prefilter + baselines) ***")
        if bc_oracle:
            print("*** BC-ORACLE MODE (ground-truth BC, PPO 3 iters, *_oracle_bc.*) ***")
        if run_name:
            print(f"*** RUN NAME: {run_name} → {output_dir}/ and {ckpt_dir}/ ***")
        print("=" * 70 + "\n")

        resume_path = None
        if (resume or bc_oracle) and not run_name:
            vol_ckpts = os.path.join(MODEL_DIR, "cs234_checkpoints_blind" if blind else "cs234_checkpoints")
            if os.path.isdir(vol_ckpts):
                os.makedirs(ckpt_dir, exist_ok=True)
                subprocess.run(["cp", "-r", vol_ckpts + "/.", ckpt_dir + "/"],
                               check=False)
                if resume:
                    ckpt_files = sorted(glob.glob(f"{ckpt_dir}/ckpt_iter_*.pt"))
                    if ckpt_files:
                        resume_path = ckpt_files[-1]
                        print(f"Resuming from volume checkpoint: {resume_path}")
                elif bc_oracle:
                    print(f"Loaded split/checkpoints from volume for BC-oracle run")

        from hotpot_pipeline import main
        main(small=small, dataset=dataset, resume=resume_path, blind=blind,
             ppo_only=ppo_only, bc_oracle=bc_oracle, run_name=run_name)

        # Collect results: from output_dir and optionally all checkpoint files
        import base64
        results = {}
        # All files in output_dir (report, comparison, trajectories, training_results, .pt)
        if os.path.isdir(output_dir):
            for name in sorted(os.listdir(output_dir)):
                path = os.path.join(output_dir, name)
                if os.path.isfile(path):
                    rel = f"{output_dir}/{name}"
                    if name.endswith((".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")):
                        with open(path, "rb") as fh:
                            results[rel] = base64.b64encode(fh.read()).decode()
                    else:
                        with open(path) as fh:
                            results[rel] = fh.read()
                    print(f"\n--- {rel} ---")
                    if not name.endswith((".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")):
                        print((results[rel])[:2000])

        # Fallback: default paths when run_name is empty (same as before)
        if not results and os.path.isdir("results"):
            for f in ["results/comparison.json", "results/training_results.json", "results/trajectories.json", "results/report.txt",
                      "results/comparison_oracle_bc.json", "results/training_results_oracle_bc.json", "results/trajectories_oracle_bc.json", "results/report_oracle_bc.txt"]:
                if os.path.exists(f):
                    with open(f) as fh:
                        results[f] = fh.read()
                    print(f"\n--- {f} ---")
                    print(results[f][:2000])
            for p in ["results/hotpot_tool_selector.pt", "results/hotpot_tool_selector_oracle_bc.pt"]:
                if os.path.exists(p):
                    with open(p, "rb") as fh:
                        results[p] = base64.b64encode(fh.read()).decode()
                    print(f"\n--- {p} ({os.path.getsize(p)} bytes) ---")

        # All checkpoint files in ckpt_dir (split.json, baselines.json, ckpt_iter_*.pt)
        if os.path.isdir(ckpt_dir):
            for name in sorted(os.listdir(ckpt_dir)):
                path = os.path.join(ckpt_dir, name)
                if os.path.isfile(path):
                    rel = f"{ckpt_dir}/{name}"
                    if name.endswith((".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")):
                        with open(path, "rb") as fh:
                            results[rel] = base64.b64encode(fh.read()).decode()
                    else:
                        with open(path) as fh:
                            results[rel] = fh.read()
                    print(f"\n--- {rel} ---")
                    if not name.endswith((".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")):
                        print((results[rel])[:500])

        # Persist to volume (results + checkpoints under run-specific keys)
        try:
            if os.path.isdir(output_dir):
                vol_results = os.path.join(MODEL_DIR, vol_results_key)
                subprocess.run(["rm", "-rf", vol_results], check=False)
                subprocess.run(["cp", "-r", output_dir, vol_results], check=False)
                print(f"Persisted results to volume at {vol_results}")
            if os.path.isdir(ckpt_dir):
                dest_ckpt = os.path.join(MODEL_DIR, vol_ckpt_key)
                subprocess.run(["rm", "-rf", dest_ckpt], check=False)
                subprocess.run(["cp", "-r", ckpt_dir, dest_ckpt], check=False)
                print(f"Persisted checkpoints to volume at {dest_ckpt}")
            model_volume.commit()
        except Exception as e:
            print(f"Warning: failed to persist to volume: {e}")

        return results

    @modal.method()
    def run_eval(self, small: bool = False, blind: bool = True,
                 prefilter: bool = True):
        """Run LLM-based evaluation (Command 2: eval_llm.py)."""
        import sys, base64

        os.chdir("/root/project")
        sys.path.insert(0, "/root/project")

        ckpt_dir = "checkpoints_blind" if blind else "checkpoints"
        output_dir = "results"

        # Restore checkpoints from volume (bc_model.pt, ppo_best.pt, split.json, etc.)
        vol_ckpt = os.path.join(MODEL_DIR, f"cs234_{ckpt_dir}")
        if os.path.isdir(vol_ckpt):
            os.makedirs(ckpt_dir, exist_ok=True)
            subprocess.run(["cp", "-r", vol_ckpt + "/.", ckpt_dir + "/"],
                           check=False)
            print(f"Restored checkpoints from volume: {vol_ckpt}")
        else:
            print(f"WARNING: {vol_ckpt} not found on volume. "
                  f"Run train_only.py (Command 1) first and upload checkpoints.")

        from eval_llm import main as eval_main
        eval_main(small=small, blind=blind, prefilter=prefilter)

        # Collect results
        results = {}
        for search_dir in [output_dir, ckpt_dir]:
            if os.path.isdir(search_dir):
                for name in sorted(os.listdir(search_dir)):
                    path = os.path.join(search_dir, name)
                    if os.path.isfile(path):
                        rel = f"{search_dir}/{name}"
                        if name.endswith((".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")):
                            with open(path, "rb") as fh:
                                results[rel] = base64.b64encode(fh.read()).decode()
                        else:
                            with open(path) as fh:
                                results[rel] = fh.read()
                        print(f"\n--- {rel} ---")
                        if not name.endswith((".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")):
                            print((results[rel])[:2000])

        # Persist results to volume
        try:
            vol_results = os.path.join(MODEL_DIR, "cs234_results_eval")
            subprocess.run(["rm", "-rf", vol_results], check=False)
            subprocess.run(["cp", "-r", output_dir, vol_results], check=False)
            model_volume.commit()
            print(f"Persisted eval results to volume at {vol_results}")
        except Exception as e:
            print(f"Warning: failed to persist eval results: {e}")

        return results


# ---------------------------------------------------------------------------
# Download results from volume (no GPU/Ollama; use after a run to get files locally)
# ---------------------------------------------------------------------------
VOLUME_RESULTS_DIR = "cs234_results"
VOLUME_RESULTS_DIR_BLIND = "cs234_results_blind"


@app.function(image=ollama_image, volumes={MODEL_DIR: model_volume})
def download_results_from_volume(blind: bool = False, run_name: str = ""):
    """Read results/ (and optionally checkpoints) from the persisted volume and return as {path: content}."""
    import base64
    out = {}
    if run_name:
        results_dir = os.path.join(MODEL_DIR, f"cs234_results_{run_name}")
        ckpt_dir = os.path.join(MODEL_DIR, f"cs234_checkpoints_blind_{run_name}")
        for vol_dir, prefix in [(results_dir, f"results_{run_name}"), (ckpt_dir, f"checkpoints_blind_{run_name}")]:
            if os.path.isdir(vol_dir):
                for name in sorted(os.listdir(vol_dir)):
                    path = os.path.join(vol_dir, name)
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            data = f.read()
                        key = f"{prefix}/{name}"
                        if name.endswith(".pt"):
                            out[key] = base64.b64encode(data).decode()
                        else:
                            try:
                                out[key] = data.decode("utf-8")
                            except Exception:
                                out[key] = base64.b64encode(data).decode()
        print(f"Found {len(out)} files on volume for run_name={run_name}")
        return out
    vol_dir = os.path.join(MODEL_DIR, VOLUME_RESULTS_DIR_BLIND if blind else VOLUME_RESULTS_DIR)
    if not os.path.isdir(vol_dir):
        print(f"No results dir on volume: {vol_dir}")
        return {}
    for name in sorted(os.listdir(vol_dir)):
        path = os.path.join(vol_dir, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            key = f"results/{name}"
            if name.endswith(".pt"):
                out[key] = base64.b64encode(data).decode()
            else:
                try:
                    out[key] = data.decode("utf-8")
                except Exception:
                    out[key] = base64.b64encode(data).decode()
    print(f"Found {len(out)} files on volume")
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(small: bool = False, dataset: str = "2wiki",
         resume: bool = False, blind: bool = False, ppo_only: bool = False,
         bc_oracle: bool = False, run_name: str = ""):
    """modal run run_modal.py [--small] [--dataset {2wiki,hotpot}] [--resume] [--blind] [--ppo-only] [--bc-oracle] [--run-name NAME]"""
    ds_label = "2WikiMultiHopQA" if dataset == "2wiki" else "HotpotQA"
    print(f"Launching {ds_label} pipeline on Modal (GPU: A10G)...")
    if small:
        print("*** SMALL MODE: quick test run ***")
    if resume:
        print("*** RESUME: will load latest checkpoint from volume ***")
    if blind:
        print("*** BLIND TITLES: paragraph titles anonymised ***")
    if ppo_only:
        print("*** PPO-ONLY: skip prefilter and baselines, go straight to BC+PPO ***")
    if bc_oracle:
        print("*** BC-ORACLE: ground-truth BC, PPO 3 iters, save to *_oracle_bc.* ***")
    if run_name:
        print(f"*** RUN NAME: {run_name} (from-scratch, prefilter+baselines, oracle BC+PPO, all in results_{run_name}/ and checkpoints_blind_{run_name}/) ***")
    print(f"Ollama {OLLAMA_VERSION} + {MODEL_NAME}\n")

    results = PipelineRunner().run.remote(
        small=small, dataset=dataset, resume=resume, blind=blind, ppo_only=ppo_only,
        bc_oracle=bc_oracle, run_name=run_name)

    if results:
        import base64
        for filepath, content in results.items():
            d = os.path.dirname(filepath)
            if d:
                os.makedirs(d, exist_ok=True)
            if filepath.endswith(".pt"):
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(content))
            else:
                with open(filepath, "w") as f:
                    f.write(content)
            print(f"Saved: {filepath}")

    print("\nDone! Check results/ and checkpoints_* for outputs.")


@app.local_entrypoint()
def download_results(blind: bool = False, run_name: str = ""):
    """Download results/ (and checkpoints if run_name set) from Modal volume into local dirs.
    Usage: modal run run_modal.py::download_results [--blind] [--run-name NAME]
    Run from project root so files land in ./results/ or ./results_NAME/ and ./checkpoints_blind_NAME/."""
    print("Fetching results from Modal volume...")
    results = download_results_from_volume.remote(blind=blind, run_name=run_name)
    if not results:
        print("No results found on volume. Run the pipeline first (e.g. modal run run_modal.py --blind).")
        return
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


@app.local_entrypoint()
def run_eval(small: bool = False, blind: bool = True, no_prefilter: bool = False):
    """Run LLM evaluation on Modal (Command 2).

    Requires trained BC/PPO checkpoints on the Modal volume (from train_only.py).
    Uploads local checkpoints_blind/ if present, then runs eval_llm on GPU with Ollama.

    Usage:
        modal run run_modal.py::run_eval [--small] [--blind] [--no-prefilter]
    """
    import base64

    ckpt_dir = "checkpoints_blind" if blind else "checkpoints"

    # Upload local checkpoints to volume if they exist
    if os.path.isdir(ckpt_dir):
        print(f"Uploading local {ckpt_dir}/ to Modal volume...")
        upload_checkpoints_to_volume.remote(blind=blind)

    print(f"Launching LLM evaluation on Modal (GPU: A10G)...")
    if small:
        print("*** SMALL MODE ***")
    print(f"Ollama {OLLAMA_VERSION} + {MODEL_NAME}\n")

    results = PipelineRunner().run_eval.remote(
        small=small, blind=blind, prefilter=not no_prefilter)

    if results:
        binary_exts = (".pt", ".png", ".pdf", ".jpg", ".jpeg", ".gif")
        for filepath, content in results.items():
            d = os.path.dirname(filepath)
            if d:
                os.makedirs(d, exist_ok=True)
            if filepath.endswith(binary_exts):
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(content))
            else:
                with open(filepath, "w") as f:
                    f.write(content)
            print(f"Saved: {filepath}")

    print("\nDone! Check results/ for LLM evaluation outputs.")


@app.function(image=ollama_image, volumes={MODEL_DIR: model_volume})
def upload_checkpoints_to_volume(blind: bool = True):
    """Upload local checkpoints to the Modal volume so run_eval can use them.

    This is called automatically by run_eval if local checkpoints exist.
    The actual upload happens via add_local_dir in the image build, but
    we also copy them to the volume for persistence across runs.
    """
    ckpt_dir = "checkpoints_blind" if blind else "checkpoints"
    src = f"/root/project/{ckpt_dir}"
    dest = os.path.join(MODEL_DIR, f"cs234_{ckpt_dir}")
    if os.path.isdir(src):
        import subprocess
        subprocess.run(["rm", "-rf", dest], check=False)
        subprocess.run(["cp", "-r", src, dest], check=False)
        model_volume.commit()
        print(f"Uploaded {src} → {dest} on volume")
    else:
        print(f"No {src} found in container image")
