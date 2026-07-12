"""Modal harness for running ACN simulations and training remotely.

The laptop stays the control surface; compute runs on Modal. The local working
tree (including uncommitted changes) is shipped with each invocation, deps are
built from ``pyproject.toml``/``uv.lock``, and all artifacts land in the
persistent ``acn-results`` volume mounted over the repo's ``results/`` dir, so
run.py writes its normal run folders without any config changes.

One-time setup (after creating an account at https://modal.com):

    uvx modal token new          # opens browser, click authorize

Usage (from the repo root):

    # CPU smoke test: runs the 8v8 avoidant-attractor preview end to end
    uvx modal run infra/modal_train.py::smoke

    # Training run (current run.py train path; TorchRL trainer slots in later)
    uvx modal run infra/modal_train.py::train --config config/aggressive_config.yaml

    # Inspect / pull artifacts from the volume
    uvx modal volume ls acn-results
    uvx modal volume get acn-results <remote-path> ./results/

GPU selection for ``train`` is read at invocation time from ACN_MODAL_GPU
(default "T4"; set to "none" for CPU-only):

    ACN_MODAL_GPU=none uvx modal run infra/modal_train.py::train ...

Approximate pricing at time of writing (verify at modal.com/pricing): T4 GPU
~$0.59/hr, CPU ~$0.135/core-hr; the starter plan includes monthly free credits.
The ACN bottleneck is CPU env-stepping, so ``train`` requests more cores than
GPU horsepower.
"""

import os
import subprocess
import sys

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_REPO = "/root/acn"

# GPU for the train function, resolved when `modal run` loads this file locally.
_gpu_env = os.environ.get("ACN_MODAL_GPU", "T4")
TRAIN_GPU = None if _gpu_env.lower() in ("", "none", "cpu") else _gpu_env

app = modal.App("acn")

results_volume = modal.Volume.from_name("acn-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Dependencies from the lockfile; rebuilds only when pyproject/uv.lock change.
    .uv_sync(REPO_ROOT)
    .env(
        {
            "SDL_VIDEODRIVER": "dummy",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # The working tree is attached at call time (not baked), so code edits do
    # not trigger image rebuilds. results/ is excluded: the volume mounts there.
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO,
        ignore=[
            "**/.git/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "results/**",
            "logs/**",
            ".env",
            "**/*.gif",
        ],
    )
)


def _run_acn(mode: str, config: str, extra_args: list) -> None:
    """Invoke run.py exactly as on the laptop, from the shipped working tree."""
    cmd = [sys.executable, "run.py", "--mode", mode, "--config", config, *extra_args]
    print(f"[modal] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)

    # Persist artifacts and show what the run produced.
    results_volume.commit()
    results_root = os.path.join(REMOTE_REPO, "results")
    for dirpath, _dirnames, filenames in os.walk(results_root):
        for filename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, filename), results_root)
            print(f"[modal] artifact: {rel}", flush=True)


@app.function(
    image=image,
    volumes={f"{REMOTE_REPO}/results": results_volume},
    cpu=4,
    timeout=30 * 60,
)
def smoke(config: str = "config/avoidant_physics_attractor_preview_config.yaml") -> None:
    """CPU-only end-to-end check: parallel sim, traces and GIF into the volume."""
    _run_acn("parallel", config, [])


@app.function(
    image=image,
    volumes={f"{REMOTE_REPO}/results": results_volume},
    cpu=8,
    gpu=TRAIN_GPU,
    timeout=12 * 60 * 60,
)
def train(config: str = "config/aggressive_config.yaml", persist: bool = False) -> None:
    """Remote training run. Uses the current run.py train path; the TorchRL

    benchmark trainer will use this same entrypoint once implemented."""
    extra = ["--persist"] if persist else []
    _run_acn("train", config, extra)
