"""Rebuild outputs/metrics from the saved checkpoints.

Recovery path for the case where metrics/ or figures/ were deleted but the
checkpoints survived. Evaluation is deterministic (eval mode, fixed test split,
same vision standardisation as training), so the rebuilt numbers are identical
to the originals rather than a fresh training run.

Run:
    python scripts/rebuild_metrics.py
Then:
    python -m src.evaluate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import CachedFeatures, evaluate, load_data, make_loader  # noqa: E402
from src.utils import compute_metrics, count_params, get_logger, save_json  # noqa: E402

log = get_logger("rebuild", C.LOGS / "rebuild.log")

SUPERVISED = ["text_only", "vision_only", "early_fusion", "cmgan", "cmgan_nogate", "cmgan_nocross"]


def load_model(name: str, ckpt: Path):
    model = build_model(name, C.PROJ_DIM, C.N_HEADS, C.DROPOUT).to("cuda")
    state = torch.load(ckpt, map_location="cuda", weights_only=True)["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def predict(name: str, ckpt: Path, rows: np.ndarray, txt, vis, txt_len, labels):
    model = load_model(name, ckpt)
    loader = make_loader(CachedFeatures(rows, txt, vis, txt_len, labels), shuffle=False)
    metrics, probs, ys = evaluate(model, loader)
    return metrics, probs, ys


def main() -> None:
    index, txt, vis, txt_len, labels = load_data()
    dev_rows = np.where(index["split"].isin(["train", "validation"]).to_numpy())[0]
    test_rows = np.where((index["split"] == "test").to_numpy())[0]
    log.info("dev %d | test %d", len(dev_rows), len(test_rows))

    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    splits = list(skf.split(dev_rows, labels[dev_rows]))

    for name in SUPERVISED:
        result = {"model": name, "recovered_from": "checkpoints", "history": None,
                  "n_dev": int(len(dev_rows)), "n_test": int(len(test_rows)), "folds": []}
        for k, (_, _va) in enumerate(splits, start=1):
            ckpt = C.CKPT / f"{name}_fold{k}.pt"
            metrics, probs, ys = predict(name, ckpt, test_rows, txt, vis, txt_len, labels)
            np.save(C.METRICS / f"{name}_fold{k}_test_probs.npy", probs)
            np.save(C.METRICS / f"{name}_fold{k}_test_y.npy", ys)
            result["folds"].append({"fold": k, "test": metrics})
            if k == 1:
                result["n_params"] = int(count_params(load_model(name, ckpt)))
            log.info("%-14s f%d macro-F1 %.4f", name, k, metrics["f1_macro"])

        metrics, probs, ys = predict(name, C.CKPT / f"{name}_final.pt",
                                     test_rows, txt, vis, txt_len, labels)
        np.save(C.METRICS / f"{name}_final_test_probs.npy", probs)
        np.save(C.METRICS / f"{name}_final_test_y.npy", ys)
        result["final"] = {"test": metrics}
        result["fold_test_macro_f1"] = [f["test"]["f1_macro"] for f in result["folds"]]
        save_json(result, C.METRICS / f"{name}.json")
        log.info("%-14s FINAL macro-F1 %.4f", name, metrics["f1_macro"])

    # ---- late fusion: re-derive alpha on each fold's validation part
    grid = np.linspace(0.0, 1.0, 21)
    result = {"model": "late_fusion", "recovered_from": "checkpoints",
              "rule": "alpha search on fold-val, 0.05 grid", "folds": []}
    alphas = []
    for k, (_, va_i) in enumerate(splits, start=1):
        va_rows = dev_rows[va_i]
        y_va = labels[va_rows].astype(int)
        _, p_t_va, _ = predict("text_only", C.CKPT / f"text_only_fold{k}.pt",
                               va_rows, txt, vis, txt_len, labels)
        _, p_v_va, _ = predict("vision_only", C.CKPT / f"vision_only_fold{k}.pt",
                               va_rows, txt, vis, txt_len, labels)
        p_t_te = np.load(C.METRICS / f"text_only_fold{k}_test_probs.npy")
        p_v_te = np.load(C.METRICS / f"vision_only_fold{k}_test_probs.npy")

        def bce(a: float) -> float:
            p = np.clip(a * p_t_va + (1 - a) * p_v_va, 1e-6, 1 - 1e-6)
            return float(-np.mean(y_va * np.log(p) + (1 - y_va) * np.log(1 - p)))

        alpha = float(grid[int(np.argmin([bce(a) for a in grid]))])
        alphas.append(alpha)
        p_te = alpha * p_t_te + (1 - alpha) * p_v_te
        np.save(C.METRICS / f"late_fusion_fold{k}_test_probs.npy", p_te)
        y_test = np.load(C.METRICS / f"text_only_fold{k}_test_y.npy")
        metrics = compute_metrics(y_test, p_te)
        result["folds"].append({"fold": k, "alpha": alpha, "test": metrics})
        log.info("late_fusion f%d alpha %.2f macro-F1 %.4f", k, alpha, metrics["f1_macro"])

    alpha = float(np.median(alphas))
    p_t = np.load(C.METRICS / "text_only_final_test_probs.npy")
    p_v = np.load(C.METRICS / "vision_only_final_test_probs.npy")
    y_test = np.load(C.METRICS / "text_only_final_test_y.npy")
    p_te = alpha * p_t + (1 - alpha) * p_v
    np.save(C.METRICS / "late_fusion_final_test_probs.npy", p_te)
    np.save(C.METRICS / "late_fusion_final_test_y.npy", y_test)
    result["final"] = {"alpha": alpha, "test": compute_metrics(y_test, p_te)}
    result["fold_test_macro_f1"] = [f["test"]["f1_macro"] for f in result["folds"]]
    save_json(result, C.METRICS / "late_fusion.json")
    log.info("late_fusion FINAL alpha %.2f macro-F1 %.4f", alpha, result["final"]["test"]["f1_macro"])
    log.info("done - now run: python -m src.evaluate")


if __name__ == "__main__":
    main()
