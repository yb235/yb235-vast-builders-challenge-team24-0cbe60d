# Yield-or-proceed

CPU logistic model: Cosmos caption labels, YOLO geometry, tracked in W&B.
1. `python3 -m venv training/.venv && training/.venv/bin/pip install -r training/requirements.txt`
2. `training/.venv/bin/python training/build_dataset.py` (`--relabel` to call Cosmos again)
3. `training/.venv/bin/python training/train.py`
Uses `INGRESS_URL`, `USERNAME`, `PASSWORD`, `WANDB_API_KEY`, `WANDB_TEAM`, and `WANDB_PROJECT`. Writes `training/out/`. A second `train.py` run with the same seed rewrites identical `metrics.json`.

About 240 clips from two scenes, one synthetic. This is a data engine and a baseline, not a robot policy. Labels are Cosmos Reason reading captions, not humans or pixels, so the model learns to agree with Cosmos. YOLO cannot name robots; every non-person box is a machine.
