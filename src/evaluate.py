"""Stage 5 - analysis of results (Criterion 6) and figures for the report.

Reads every outputs/metrics/<model>.json written by Stage 4 and produces
  * tables.md            - mean +- std over the 5 folds, plus the refit model
  * paired significance tests (t-test + Wilcoxon) against CM-GAN
  * fig10 .. fig16      - metric bars with CI, confusion matrices, ROC,
                          training curves, ablation, incongruity magnitude,
                          cross-modal attention maps

Run:
    python -m src.evaluate
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader

try:
    from .config import (
        CKPT, DROPOUT, FEATURES, FIG, MANIFESTS, METRICS, N_HEADS, PROJ_DIM,
    )
    from .models import build_model
    from .utils import compute_metrics, get_logger, load_json, save_json
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import CKPT, DROPOUT, FEATURES, FIG, MANIFESTS, METRICS, N_HEADS, PROJ_DIM
    from src.models import build_model
    from src.utils import compute_metrics, get_logger, load_json, save_json

log = get_logger("evaluate")

plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.25})

ORDER = ["text_only", "vision_only", "early_fusion", "late_fusion",
         "cmgan_nocross", "cmgan_nogate", "cmgan"]
PRETTY = {
    "text_only": "BERT text-only",
    "vision_only": "ResNet-50 vision-only",
    "early_fusion": "Early fusion (concat)",
    "late_fusion": "Late fusion (decision)",
    "cmgan_nocross": "CM-GAN w/o cross-attn",
    "cmgan_nogate": "CM-GAN w/o gate",
    "cmgan": "CM-GAN (full)",
}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1_binary", "f1_macro", "roc_auc"]


def load_all() -> dict:
    out = {}
    for name in ORDER:
        p = METRICS / f"{name}.json"
        if p.exists():
            out[name] = load_json(p)
        else:
            log.warning("missing %s", p.name)
    return out


# ------------------------------------------------------------------ tables
def table_markdown(results: dict) -> str:
    lines = ["| Model | Accuracy | Precision | Recall | Binary F1 | Macro F1 | ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in results:
            continue
        r = results[name]
        cells = []
        for k in METRIC_KEYS:
            vals = [f["test"][k] for f in r["folds"]]
            cells.append(f"{np.mean(vals):.4f} 卤 {np.std(vals):.4f}")
        lines.append(f"| {PRETTY[name]} | " + " | ".join(cells) + " |")

    lines += ["", "Refit on the full dev pool, evaluated once on the official test split:", "",
              "| Model | Accuracy | Precision | Recall | Binary F1 | Macro F1 | ROC-AUC |",
              "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in results:
            continue
        f = results[name]["final"]["test"]
        lines.append(f"| {PRETTY[name]} | " + " | ".join(f"{f[k]:.4f}" for k in METRIC_KEYS) + " |")
    return "\n".join(lines) + "\n"


def paired_tests(results: dict) -> list[dict]:
    if "cmgan" not in results:
        return []
    base = np.array(results["cmgan"]["fold_test_macro_f1"], dtype=float)
    rows = []
    for name in ORDER:
        if name == "cmgan" or name not in results:
            continue
        other = np.array(results[name]["fold_test_macro_f1"], dtype=float)
        diff = base - other
        t_p = float(stats.ttest_rel(base, other).pvalue)
        try:
            w_p = float(stats.wilcoxon(base, other).pvalue)
        except Exception:  # noqa: BLE001  (all-zero differences)
            w_p = float("nan")
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        rows.append({
            "comparison": f"CM-GAN vs {PRETTY[name]}",
            "mean_macro_f1_diff": float(diff.mean()),
            "std_diff": sd,
            "t_stat": float(stats.ttest_rel(base, other).statistic),
            "t_p": t_p,
            "wilcoxon_p": w_p,
            "cohens_d": float(diff.mean() / sd) if sd > 0 else float("nan"),
            "significant_0.05": bool(t_p < 0.05),
        })
    return rows


# ----------------------------------------------------------------- figures
def fig_metric_bars(results: dict) -> str:
    names = [n for n in ORDER if n in results]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    for ax, key, title in ((axes[0], "f1_macro", "Macro-F1"), (axes[1], "accuracy", "Accuracy")):
        means, errs = [], []
        for n in names:
            vals = [f["test"][key] for f in results[n]["folds"]]
            means.append(np.mean(vals))
            errs.append(np.std(vals))
        colors = ["#C44E52" if n == "cmgan" else "#4C72B0" for n in names]
        ax.bar(x, means, yerr=errs, capsize=2, color=colors, alpha=0.9,
               error_kw={"elinewidth": 0.7})
        ax.set_xticks(x, [PRETTY[n] for n in names], rotation=32, ha="right", fontsize=5.6)
        ax.set_ylim(min(means) - 0.05, max(means) + 0.035)
        ax.set_ylabel(title, fontsize=7)
        ax.set_title(f"{title} on the official test split", fontsize=8)
        ax.tick_params(axis="y", labelsize=6)
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.text(i, m + e + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=5.0)
    out = FIG / "fig10_metric_bars.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def _test_labels(name: str) -> np.ndarray:
    """Official test labels, shared by every model.

    late_fusion reuses the two unimodal checkpoints, so it never writes its own
    *_test_y.npy; fall back to any model that did.
    """
    p = METRICS / f"{name}_final_test_y.npy"
    if p.exists():
        return np.load(p).astype(int)
    for alt in ("text_only", "cmgan", "vision_only"):
        q = METRICS / f"{alt}_final_test_y.npy"
        if q.exists():
            return np.load(q).astype(int)
    raise FileNotFoundError("no *_final_test_y.npy available")


def fig_confusions(results: dict) -> str:
    names = [n for n in ORDER if n in results and (METRICS / f"{n}_final_test_probs.npy").exists()]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.62 * ncol, 1.9 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        p = np.load(METRICS / f"{name}_final_test_probs.npy")
        y = _test_labels(name)
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(PRETTY[name], fontsize=6.2)
    for ax in axes[len(names):]:
        ax.axis("off")
    out = FIG / "fig11_confusion_matrices.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def fig_roc(results: dict) -> str:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for name in ORDER:
        p_file = METRICS / f"{name}_final_test_probs.npy"
        if name not in results or not p_file.exists():
            continue
        p = np.load(p_file)
        y = _test_labels(name)
        fpr, tpr, _ = roc_curve(y, p)
        auc = results[name]["final"]["test"]["roc_auc"]
        lw = 2.2 if name == "cmgan" else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{PRETTY[name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC on official test (refit models)")
    ax.legend(fontsize=6.0, frameon=False, loc="lower right")
    out = FIG / "fig12_roc.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def fig_curves(results: dict) -> str:
    if "cmgan" not in results:
        return ""
    hist = results["cmgan"]["folds"][0].get("history")
    if not hist:
        # metrics rebuilt from checkpoints carry no training history; the figure
        # already produced by the original run stays on disk.
        log.info("no training history in metrics (rebuilt run) - keeping the existing fig13")
        return ""
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    axes[0].plot(ep, [h["train_loss"] for h in hist], marker="o", ms=3, color="#C44E52")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss (BCE)")
    axes[0].set_title("CM-GAN training loss (fold 1)")
    axes[1].plot(ep, [h["val_f1_macro"] for h in hist], marker="o", ms=3, label="fold-val macro F1",
                 color="#4C72B0")
    axes[1].plot(ep, [h["val_accuracy"] for h in hist], marker="s", ms=3, label="fold-val accuracy",
                 color="#55A868")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Held-out fold performance")
    axes[1].legend(frameon=False, fontsize=8)
    out = FIG / "fig13_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def fig_ablation(results: dict) -> str:
    names = [n for n in ["cmgan_nocross", "cmgan_nogate", "cmgan"] if n in results]
    if len(names) < 2:
        return ""
    means = [np.mean([f["test"]["f1_macro"] for f in results[n]["folds"]]) for n in names]
    errs = [np.std([f["test"]["f1_macro"] for f in results[n]["folds"]]) for n in names]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.bar(range(len(names)), means, yerr=errs, capsize=4,
           color=["#999999", "#8172B2", "#C44E52"])
    ax.set_xticks(range(len(names)), [PRETTY[n] for n in names], fontsize=8)
    ax.set_ylabel("macro F1 (official test)")
    ax.set_ylim(min(means) - 0.04, max(means) + 0.02)
    ax.set_title("Ablation: contribution of cross-attention and the incongruity gate")
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    out = FIG / "fig14_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


# ------------------------------------------------ incongruity + attention
def _load_test_features():
    index = pd.read_csv(FEATURES / "index.csv")
    meta = load_json(FEATURES / "features_meta.json")
    n = int(meta["n"])
    _, L, D = meta["text_shape"]
    _, K, V = meta["vis_shape"]
    txt = np.memmap(FEATURES / "text_tokens.npy", dtype=np.float16, mode="r", shape=(n, L, D))
    txt_len = np.load(FEATURES / "text_lengths.npy")
    vis = np.memmap(FEATURES / "vis_tokens.npy", dtype=np.float16, mode="r", shape=(n, K, V))
    rows = np.where((index["split"] == "test").to_numpy())[0]
    return index, txt, txt_len, vis, rows


def fig_delta(results: dict) -> tuple[str, dict]:
    ck = CKPT / "cmgan_final.pt"
    if not ck.exists():
        return "", {}
    index, txt, txt_len, vis, rows = _load_test_features()
    model = build_model("cmgan", PROJ_DIM, N_HEADS, DROPOUT).cuda()
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True)["state_dict"])
    model.eval()

    delta_norms, labels, probs = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 128):
            batch = rows[i:i + 128]
            t = torch.from_numpy(np.asarray(txt[batch], dtype=np.float32)).cuda()
            L = txt_len[batch]
            mask = torch.zeros(t.shape[0], t.shape[1], dtype=torch.bool)
            for j, l in enumerate(L):
                mask[j, :int(l)] = True
            v = torch.from_numpy(np.asarray(vis[batch], dtype=np.float32)).cuda()
            logits, parts = model(t, mask.cuda(), v, return_parts=True)
            delta_norms.append(parts["delta"].norm(dim=-1).float().cpu().numpy())
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(index["label"].to_numpy()[batch])
    d = np.concatenate(delta_norms)
    y = np.concatenate(labels).astype(int)
    p = np.concatenate(probs)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    axes[0].hist(d[y == 0], bins=50, alpha=0.6, label="non-sarcastic", color="#4C72B0", density=True)
    axes[0].hist(d[y == 1], bins=50, alpha=0.6, label="sarcastic", color="#C44E52", density=True)
    axes[0].set_xlabel("||Delta||_2 (gated incongruity magnitude)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Where the model sees conflict")
    axes[0].legend(frameon=False)

    order = np.argsort(d)
    nb = 9
    chunks = np.array_split(order, nb)
    mean_d = [d[c].mean() for c in chunks]
    acc = [float(((p[c] >= 0.5).astype(int) == y[c]).mean()) for c in chunks]
    axes[1].plot(mean_d, acc, marker="o", color="#55A868")
    axes[1].set_xlabel("mean ||Delta|| within bin")
    axes[1].set_ylabel("accuracy within bin")
    axes[1].set_title("Accuracy vs incongruity magnitude")

    out = FIG / "fig15_incongruity_magnitude.png"
    fig.savefig(out)
    plt.close(fig)

    t, pval = stats.ttest_ind(d[y == 1], d[y == 0], equal_var=False)
    info = {
        "delta_mean_sarcastic": float(d[y == 1].mean()),
        "delta_mean_non_sarcastic": float(d[y == 0].mean()),
        "welch_t": float(t), "welch_p": float(pval),
        "corr_delta_correct": float(np.corrcoef(d, ((p >= 0.5).astype(int) == y).astype(float))[0, 1]),
    }
    return out.name, info


def fig_attention(results: dict, n_show: int = 3) -> str:
    ck = CKPT / "cmgan_final.pt"
    if not ck.exists():
        return ""
    index, txt, txt_len, vis, rows = _load_test_features()
    model = build_model("cmgan", PROJ_DIM, N_HEADS, DROPOUT).cuda()
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True)["state_dict"])
    model.eval()

    # pick sarcastic test posts the model gets right with high confidence
    picks = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            batch = rows[i:i + 256]
            t = torch.from_numpy(np.asarray(txt[batch], dtype=np.float32)).cuda()
            L = txt_len[batch]
            mask = torch.zeros(t.shape[0], t.shape[1], dtype=torch.bool)
            for j, l in enumerate(L):
                mask[j, :int(l)] = True
            v = torch.from_numpy(np.asarray(vis[batch], dtype=np.float32)).cuda()
            logits = model(t, mask.cuda(), v)
            prob = torch.sigmoid(logits).float().cpu().numpy()
            y = index["label"].to_numpy()[batch]
            for k, r in enumerate(batch):
                if y[k] == 1 and prob[k] > 0.9:
                    picks.append((r, prob[k]))
            if len(picks) >= 60:
                break
    picks = picks[:n_show]
    if not picks:
        return ""

    root = MANIFESTS.parent.parent
    fig, axes = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.9),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for col, (r, prob) in enumerate(picks):
        batch = np.array([r])
        t = torch.from_numpy(np.asarray(txt[batch], dtype=np.float32)).cuda()
        L = int(txt_len[r])
        mask = torch.zeros(t.shape[0], t.shape[1], dtype=torch.bool)
        mask[0, :L] = True
        v = torch.from_numpy(np.asarray(vis[batch], dtype=np.float32)).cuda()
        with torch.no_grad():
            h_t = model.txt_proj(t)
            h_v = model.vis_proj(v)
            _, w = model.block.vis_attn(query=h_v, key=h_t, value=h_t,
                                        key_padding_mask=~mask.cuda(),
                                        need_weights=True, average_attn_weights=True)
        w = w[0].float().cpu().numpy()[:, :L].mean(-1)   # (49,)
        attn = w.reshape(7, 7)
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        attn_img = np.asarray(Image.fromarray((attn * 255).astype(np.uint8))
                              .resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0

        img = Image.open(root / index.loc[r, "img_path"]).convert("RGB").resize((224, 224))
        clean = re.sub(r"[^\x00-\x7f]+", "", str(index.loc[r, "text"])).strip()
        axes[0, col].imshow(img)
        axes[0, col].axis("off")
        axes[0, col].set_title(clean[:38] + ("..." if len(clean) > 38 else ""),
                               fontsize=4.8, wrap=True)
        axes[1, col].imshow(img)
        axes[1, col].imshow(attn_img, cmap="jet", alpha=0.45, interpolation="bilinear")
        axes[1, col].axis("off")
        axes[1, col].set_title(f"p(sarcastic) = {prob:.2f}", fontsize=5.6)
    out = FIG / "fig16_attention_maps.png"
    fig.savefig(out)
    plt.close(fig)
    return out.name


def main() -> None:
    results = load_all()
    if not results:
        raise SystemExit("no metrics found - run `python -m src.train` first")

    md = table_markdown(results)
    (METRICS / "tables.md").write_text(md, encoding="utf-8")
    log.info("tables.md written")

    tests = paired_tests(results)
    figs = {}
    figs["fig10"] = fig_metric_bars(results)
    figs["fig11"] = fig_confusions(results)
    figs["fig12"] = fig_roc(results)
    figs["fig13"] = fig_curves(results)
    figs["fig14"] = fig_ablation(results)
    figs["fig15"], delta_info = fig_delta(results)
    figs["fig16"] = fig_attention(results)

    summary = {
        "models": {n: {"final": r["final"]["test"],
                       "fold_mean_macro_f1": float(np.mean(r["fold_test_macro_f1"])),
                       "fold_std_macro_f1": float(np.std(r["fold_test_macro_f1"])),
                       "n_params": r.get("n_params")} for n, r in results.items()},
        "paired_tests_vs_cmgan": tests,
        "incongruity": delta_info,
        "figures": figs,
    }
    save_json(summary, METRICS / "results_summary.json")

    for t in tests:
        log.info("%-40s diff=%+.4f  t-p=%.4g  wilcoxon-p=%.4g  d=%.2f",
                 t["comparison"], t["mean_macro_f1_diff"], t["t_p"], t["wilcoxon_p"], t["cohens_d"])
    log.info("figures -> %s", FIG)


if __name__ == "__main__":
    main()


