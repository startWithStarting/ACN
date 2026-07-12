# Remote training infrastructure

ACN trains remotely on [Modal](https://modal.com); the laptop is only the
control surface. Design rationale and the platform decision are recorded in
[the decision log](../docs/communication_decision_log.md).

## One-time setup (~2 minutes)

1. Create an account at <https://modal.com> (GitHub or Google sign-in works).
1. From any terminal on this machine run:

```bash
uvx modal token new
```

A browser window opens; click **Authorize**. The token is stored under
`~/.modal.toml`. That is the entire manual setup — no quotas, images, or
instance management.

## Commands (from the repo root)

```bash
# End-to-end CPU smoke test (8v8 preview scenario, ~minutes, costs cents)
uvx modal run infra/modal_train.py::smoke

# Remote training with the current run.py train path (T4 GPU by default)
uvx modal run infra/modal_train.py::train --config config/aggressive_config.yaml

# CPU-only training run
ACN_MODAL_GPU=none uvx modal run infra/modal_train.py::train --config config/aggressive_config.yaml

# Browse and retrieve artifacts (traces, GIFs, checkpoints)
uvx modal volume ls acn-results
# NOTE: for directories, cd into the destination and omit the local path —
# the CLI's explicit-dest form errors on directory downloads (verified v1.5.2).
cd results/ && uvx modal volume get acn-results <experiment>/<run-dir>
```

Logs stream live to the terminal; Ctrl+C detaches without killing the remote
run (`modal app list` / `modal app logs acn` to reattach).

## How it works

- **Code**: the local working tree — including uncommitted changes — is
  attached to every invocation. No commit/push needed to iterate.
- **Deps**: the image is built from `pyproject.toml` + `uv.lock` and cached;
  it rebuilds only when the lockfile changes.
- **Artifacts**: the persistent `acn-results` volume is mounted over the
  repo's `results/` directory remotely, so `run.py` writes its normal
  timestamped run folders unchanged. Pull them back with `modal volume get`,
  then ingest locally (`src/storage/ingest.py` or the API `POST /ingest`) —
  the recorder-factory boundary keeps DB infrastructure local.
- **Headless**: the image sets `SDL_VIDEODRIVER=dummy` and `MPLBACKEND=Agg`.

## Scale-out path (documented, not built)

If runs outgrow Modal's burst model (multi-day sweeps where spot-instance
economics dominate), the same config-driven entrypoint moves onto GCP/AWS spot
via [SkyPilot](https://skypilot.readthedocs.io) without code changes. Do not
build GCP infrastructure before that pressure exists.
