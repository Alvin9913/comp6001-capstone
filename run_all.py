"""One-command pipeline: data -> EDA -> features -> train -> evaluate.

    python run_all.py                  # everything
    python run_all.py --from features  # resume from a stage
    python run_all.py --only eda
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["data", "eda", "features", "train", "evaluate"]
MODULE = {
    "data": "src.data_prep",
    "eda": "src.eda",
    "features": "src.features",
    "train": "src.train",
    "evaluate": "src.evaluate",
}


def run(stage: str) -> None:
    t0 = time.time()
    print(f"\n===== stage: {stage} ({MODULE[stage]}) =====", flush=True)
    subprocess.run([sys.executable, "-m", MODULE[stage]], cwd=ROOT, check=True)
    print(f"===== {stage} done in {time.time() - t0:.1f}s =====", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STAGES, default="data")
    ap.add_argument("--only", dest="only", choices=STAGES, default=None)
    ap.add_argument("--train-models", nargs="*", default=None)
    args = ap.parse_args()

    if args.only:
        stages = [args.only]
    else:
        stages = STAGES[STAGES.index(args.start):]

    for stage in stages:
        if stage == "train" and args.train_models:
            print(f"\n===== stage: train ({args.train_models}) =====", flush=True)
            subprocess.run([sys.executable, "-m", "src.train", "--models", *args.train_models],
                           cwd=ROOT, check=True)
        else:
            run(stage)


if __name__ == "__main__":
    main()
