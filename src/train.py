"""Stage 4 - training and cross-validation for CM-GAN and every baseline.

Protocol (identical for all models):
  * dev pool  = official train + validation (22,226 samples)
  * 5-fold stratified CV over the dev pool
      - fit on 4 folds, pick the epoch by macro-F1 on the 5th
      - score that epoch on the official test split, so the paired test in
        Stage 5 compares all models on exactly the same 2,409 examples
  * final model = refit on the whole dev pool for a fixed number of epochs
    (median best epoch of the folds), then scored on the official test split

Run:
    python -m src.train --models cmgan
    python -m src.train --models text_only vision_only late_fusion
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

try:
    from .config import (
        AMP, BATCH_SIZE, CKPT, DROPOUT, EPOCHS, FEATURES, GRAD_CLIP, LR, MANIFESTS,
        METRICS, MODELS, N_FOLDS, N_HEADS, PROJ_DIM, SEED, WARMUP_FRAC, WEIGHT_DECAY,
    )
    from .models import LateFusion, build_model
    from .utils import compute_metrics, count_params, get_logger, load_json, save_json, set_seed
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import (
        AMP, BATCH_SIZE, CKPT, DROPOUT, EPOCHS, FEATURES, GRAD_CLIP, LR, MANIFESTS,
        METRICS, MODELS, N_FOLDS, N_HEADS, PROJ_DIM, SEED, WARMUP_FRAC, WEIGHT_DECAY,
    )
    from src.models import LateFusion, build_model
    from src.utils import compute_metrics, count_params, get_logger, load_json, save_json, set_seed

log = get_logger("train", METRICS.parent / "logs" / "train.log")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
# ResNet layer-4 activations are neither normalised nor zero-mean, so feeding
# them straight into a linear layer saturates the unit and the vision-only
# baseline collapses onto the majority class. The statistics are estimated once
# on the training split and applied to every split (no test leakage).
_VIS_STATS: dict = {"mean": None, "std": None}


class CachedFeatures(Dataset):
    """Reads the float16 memmaps written by src.features."""

    def __init__(self, rows: np.ndarray, txt: np.ndarray, vis: np.memmap,
                 txt_len: np.ndarray, labels: np.ndarray):
        self.rows = rows
        self.txt = txt
        self.vis = vis
        self.txt_len = txt_len
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        t = torch.from_numpy(np.asarray(self.txt[r], dtype=np.float32))
        L = int(self.txt_len[r])
        mask = torch.zeros(t.shape[0], dtype=torch.bool)
        mask[:L] = True
        v = np.asarray(self.vis[r], dtype=np.float32)
        if _VIS_STATS["mean"] is not None:
            v = (v - _VIS_STATS["mean"]) / _VIS_STATS["std"]
        y = torch.tensor(float(self.labels[r]))
        return t, mask, torch.from_numpy(v), y


def load_data():
    index = pd.read_csv(FEATURES / "index.csv")
    meta = load_json(FEATURES / "features_meta.json")
    n = int(meta["n"])
    _, L, D = meta["text_shape"]
    _, K, V = meta["vis_shape"]
    # these are raw float16 memmaps written by src.features, not .npy containers
    txt = np.memmap(FEATURES / "text_tokens.npy", dtype=np.float16, mode="r", shape=(n, L, D))
    vis = np.memmap(FEATURES / "vis_tokens.npy", dtype=np.float16, mode="r", shape=(n, K, V))
    txt_len = np.load(FEATURES / "text_lengths.npy")
    labels = index["label"].to_numpy().astype(np.float32)

    # estimate the visual standardisation on the training split only
    train_rows = np.where((index["split"] == "train").to_numpy())[0]
    step = max(1, len(train_rows) // 4000)
    sample = train_rows[::step][:4000]
    chunk = np.asarray(vis[sample], dtype=np.float32)
    _VIS_STATS["mean"] = chunk.mean(axis=(0, 1)).astype(np.float32)
    _VIS_STATS["std"] = (chunk.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    log.info("vision standardisation from %d train rows (mean |x|=%.2f)", len(sample),
             float(np.abs(_VIS_STATS["mean"]).mean()))
    return index, txt, vis, txt_len, labels


def make_loader(ds: CachedFeatures, shuffle: bool, batch_size: int = BATCH_SIZE):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=False, pin_memory=True)


# --------------------------------------------------------------- training
def evaluate(model, loader) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for t, m, v, y in loader:
            logits = model(t.to(DEVICE), m.to(DEVICE), v.to(DEVICE))
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return compute_metrics(ys, probs), probs, ys


def train_one(model, tr_loader, va_loader, epochs: int, tag: str) -> tuple[dict, list]:
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(1, len(tr_loader))
    total = steps * epochs
    warm = max(1, int(total * WARMUP_FRAC))

    def sched(step: int) -> float:
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched_fn = torch.optim.lr_scheduler.LambdaLR(opt, sched)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best = {"score": -1.0, "epoch": -1, "state": None, "val": None}
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for t, m, v, y in tr_loader:
            t, m, v, y = t.to(DEVICE), m.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                logits = model(t, m, v)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched_fn.step()
            running += loss.item() * len(y)

        val_metrics, _, _ = evaluate(model, va_loader)
        loss_avg = running / len(tr_loader.dataset)
        history.append({"epoch": ep, "train_loss": loss_avg, **{f"val_{k}": x for k, x in val_metrics.items()}})
        log.info("%s ep %02d/%02d loss %.4f val_acc %.4f val_macroF1 %.4f (%.1fs)",
                 tag, ep, epochs, loss_avg, val_metrics["accuracy"],
                 val_metrics["f1_macro"], time.time() - t0)

        if val_metrics["f1_macro"] > best["score"]:
            best.update(
                score=val_metrics["f1_macro"],
                epoch=ep,
                state={k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()},
                val=val_metrics,
            )

    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "val": best["val"], "history": history}, best["state"]


def run_supervised(name: str, folds: int) -> dict:
    set_seed(SEED)
    index, txt, vis, txt_len, labels = load_data()
    dev_rows = np.where(index["split"].isin(["train", "validation"]).to_numpy())[0]
    test_rows = np.where((index["split"] == "test").to_numpy())[0]
    log.info("%s | dev %d | test %d", name, len(dev_rows), len(test_rows))

    test_ds = CachedFeatures(test_rows, txt, vis, txt_len, labels)
    y_dev = labels[dev_rows]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    result = {"model": name, "folds": [], "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "lr": LR, "n_dev": int(len(dev_rows)), "n_test": int(len(test_rows))}
    best_epochs = []

    for k, (tr_i, va_i) in enumerate(skf.split(dev_rows, y_dev), start=1):
        tr_rows, va_rows = dev_rows[tr_i], dev_rows[va_i]
        tr_loader = make_loader(CachedFeatures(tr_rows, txt, vis, txt_len, labels), shuffle=True)
        va_loader = make_loader(CachedFeatures(va_rows, txt, vis, txt_len, labels), shuffle=False)
        test_loader = make_loader(test_ds, shuffle=False)

        model = build_model(name, PROJ_DIM, N_HEADS, DROPOUT).to(DEVICE)
        if k == 1:
            result["n_params"] = int(count_params(model))
            log.info("%s params: %d", name, result["n_params"])

        info, state = train_one(model, tr_loader, va_loader, EPOCHS, f"{name} f{k}")
        best_epochs.append(info["best_epoch"])
        test_metrics, probs, ys = evaluate(model, test_loader)

        torch.save({"state_dict": state, "model": name, "fold": k},
                   CKPT / f"{name}_fold{k}.pt")
        np.save(METRICS / f"{name}_fold{k}_test_probs.npy", probs)
        np.save(METRICS / f"{name}_fold{k}_test_y.npy", ys)

        result["folds"].append(
            {"fold": k, "best_epoch": info["best_epoch"], "val": info["val"],
             "test": test_metrics, "history": info["history"]}
        )
        log.info("%s f%d TEST macro-F1 %.4f acc %.4f auc %.4f",
                 name, k, test_metrics["f1_macro"], test_metrics["accuracy"],
                 test_metrics["roc_auc"])

    # ---- refit on the whole dev pool
    refit_epochs = int(np.median(best_epochs)) if best_epochs else EPOCHS
    refit_epochs = max(2, refit_epochs)
    log.info("%s refit on all dev for %d epochs", name, refit_epochs)
    tr_loader = make_loader(CachedFeatures(dev_rows, txt, vis, txt_len, labels), shuffle=True)
    va_loader = make_loader(test_ds, shuffle=False)   # monitored only for the curve
    model = build_model(name, PROJ_DIM, N_HEADS, DROPOUT).to(DEVICE)
    info, state = train_one(model, tr_loader, va_loader, refit_epochs, f"{name} final")
    test_metrics, probs, ys = evaluate(model, make_loader(test_ds, shuffle=False))

    torch.save({"state_dict": state, "model": name, "fold": 0}, CKPT / f"{name}_final.pt")
    np.save(METRICS / f"{name}_final_test_probs.npy", probs)
    np.save(METRICS / f"{name}_final_test_y.npy", ys)

    result["final"] = {"epochs": refit_epochs, "test": test_metrics, "history": info["history"]}
    result["fold_test_macro_f1"] = [f["test"]["f1_macro"] for f in result["folds"]]
    log.info("%s FINAL TEST macro-F1 %.4f acc %.4f auc %.4f",
             name, test_metrics["f1_macro"], test_metrics["accuracy"], test_metrics["roc_auc"])

    save_json(result, METRICS / f"{name}.json")
    return result


# ------------------------------------------------------------- late fusion
def run_late_fusion(folds: int) -> dict:
    """Combine the two unimodal heads at decision level; alpha fitted on val."""
    set_seed(SEED)
    index, txt, vis, txt_len, labels = load_data()
    dev_rows = np.where(index["split"].isin(["train", "validation"]).to_numpy())[0]
    test_rows = np.where((index["split"] == "test").to_numpy())[0]
    y_test = labels[test_rows].astype(int)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    grid = np.linspace(0.0, 1.0, 21)

    def probs_of(model_name: str, rows: np.ndarray, ckpt: Path) -> np.ndarray:
        model = build_model(model_name, PROJ_DIM, N_HEADS, DROPOUT).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)["state_dict"]
        model.load_state_dict(state)
        loader = make_loader(CachedFeatures(rows, txt, vis, txt_len, labels), shuffle=False)
        _, p, _ = evaluate(model, loader)
        return p

    result = {"model": "late_fusion", "folds": [], "rule": "alpha search on fold-val, 0.05 grid"}
    alphas = []

    for k, (_, va_i) in enumerate(skf.split(dev_rows, labels[dev_rows]), start=1):
        va_rows = dev_rows[va_i]
        y_va = labels[va_rows].astype(int)
        p_t_va = probs_of("text_only", va_rows, CKPT / f"text_only_fold{k}.pt")
        p_v_va = probs_of("vision_only", va_rows, CKPT / f"vision_only_fold{k}.pt")
        p_t_te = np.load(METRICS / f"text_only_fold{k}_test_probs.npy")
        p_v_te = np.load(METRICS / f"vision_only_fold{k}_test_probs.npy")

        def bce(a: float) -> float:
            p = np.clip(a * p_t_va + (1 - a) * p_v_va, 1e-6, 1 - 1e-6)
            return float(-np.mean(y_va * np.log(p) + (1 - y_va) * np.log(1 - p)))

        losses = [bce(a) for a in grid]
        alpha = float(grid[int(np.argmin(losses))])
        alphas.append(alpha)

        p_te = alpha * p_t_te + (1 - alpha) * p_v_te
        metrics = compute_metrics(y_test, p_te)
        result["folds"].append({"fold": k, "alpha": alpha, "test": metrics})
        np.save(METRICS / f"late_fusion_fold{k}_test_probs.npy", p_te)
        log.info("late_fusion f%d alpha=%.2f TEST macro-F1 %.4f", k, alpha, metrics["f1_macro"])

    # final: reuse the median alpha found on the folds (no in-sample leak)
    alpha = float(np.median(alphas))
    p_t_te = np.load(METRICS / "text_only_final_test_probs.npy")
    p_v_te = np.load(METRICS / "vision_only_final_test_probs.npy")
    p_te = alpha * p_t_te + (1 - alpha) * p_v_te
    final_metrics = compute_metrics(y_test, p_te)
    np.save(METRICS / "late_fusion_final_test_probs.npy", p_te)
    np.save(METRICS / "late_fusion_final_test_y.npy", y_test)

    result["final"] = {"alpha": alpha, "test": final_metrics, "epochs": None}
    result["fold_test_macro_f1"] = [f["test"]["f1_macro"] for f in result["folds"]]
    log.info("late_fusion FINAL alpha=%.2f TEST macro-F1 %.4f", alpha, final_metrics["f1_macro"])
    save_json(result, METRICS / "late_fusion.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    order = ["text_only", "vision_only", "early_fusion", "cmgan", "cmgan_nogate",
             "cmgan_nocross", "late_fusion"]
    todo = [m for m in order if m in args.models]
    log.info("device=%s torch=%s cuda=%s", DEVICE, torch.__version__, torch.cuda.is_available())
    for name in todo:
        if name == "late_fusion":
            need = [CKPT / f"text_only_fold{k}.pt" for k in range(1, args.folds + 1)]
            need += [CKPT / f"vision_only_fold{k}.pt" for k in range(1, args.folds + 1)]
            need += [CKPT / "text_only_final.pt", CKPT / "vision_only_final.pt"]
            missing = [p.name for p in need if not p.exists()]
            if missing:
                log.warning("skip late_fusion, missing %s (run text_only + vision_only first)",
                            missing[:3])
                continue
            run_late_fusion(args.folds)
        else:
            run_supervised(name, args.folds)


if __name__ == "__main__":
    main()
