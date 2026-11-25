"""Stage 2 - dataset EDA (Criterion 1).

Produces the figures and the JSON summary the report cites:
  fig01 split / class balance
  fig02 tweet length distribution by class
  fig03 image geometry (width vs height, aspect ratio)
  fig04 image appearance (brightness / contrast / colourfulness) by class
  fig05 example posts (sarcastic vs non-sarcastic)
  fig06 most frequent terms per class
  fig07 length vs label

Run:
    python -m src.eda
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

try:
    from .config import FIG, MANIFESTS, METRICS
    from .utils import get_logger, save_json
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import FIG, MANIFESTS, METRICS
    from src.utils import get_logger, save_json

log = get_logger("eda")

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
})
PALETTE = {0: "#4C72B0", 1: "#C44E52"}
CLASS_NAMES = {0: "non-sarcastic", 1: "sarcastic"}

STOP = set("""a an the is are was were be been being to of and or in on at for with from by
this that these those it its i you he she they we my your our their his her as but not no so
if then than too very can could would should will just about into over after before out up down
do does did done have has had me him them us who whom which what when where why how all any
one two get got go going like im ive dont doesnt really much more most some such only also
""".split())

TOKEN_RE = re.compile(r"[a-z][a-z']+")


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for split in ["train", "validation", "test"]:
        p = MANIFESTS / f"{split}.csv"
        if p.exists():
            out[split] = pd.read_csv(p)
    return out


# ------------------------------------------------------------------ figures
def fig_split_balance(dfs: dict[str, pd.DataFrame]) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    splits = list(dfs)
    counts = np.array([[int((dfs[s]["label"] == c).sum()) for c in (0, 1)] for s in splits])

    x = np.arange(len(splits))
    axes[0].bar(x - 0.2, counts[:, 0], 0.4, label=CLASS_NAMES[0], color=PALETTE[0])
    axes[0].bar(x + 0.2, counts[:, 1], 0.4, label=CLASS_NAMES[1], color=PALETTE[1])
    axes[0].set_xticks(x, splits)
    axes[0].set_ylabel("samples")
    axes[0].set_title("Class balance per split")
    axes[0].legend(frameon=False)
    for i in range(len(splits)):
        axes[0].text(i - 0.2, counts[i, 0], str(counts[i, 0]), ha="center", va="bottom", fontsize=7)
        axes[0].text(i + 0.2, counts[i, 1], str(counts[i, 1]), ha="center", va="bottom", fontsize=7)

    ratios = counts[:, 1] / counts.sum(1)
    axes[1].bar(splits, ratios, color="#8172B2")
    axes[1].axhline(0.5, ls="--", c="k", lw=0.8, label="balanced (0.5)")
    axes[1].set_ylim(0, 0.6)
    axes[1].set_ylabel("sarcastic ratio")
    axes[1].set_title("Positive-class ratio")
    axes[1].legend(frameon=False)
    for i, r in enumerate(ratios):
        axes[1].text(i, r + 0.005, f"{r:.3f}", ha="center", fontsize=8)

    out = FIG / "fig01_split_class_balance.png"
    fig.savefig(out)
    plt.close(fig)
    return {"path": out.name, "counts": counts.tolist(), "ratios": ratios.tolist()}


def fig_text_length(dfs: dict[str, pd.DataFrame]) -> dict:
    df = dfs["train"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for c in (0, 1):
        sns.histplot(df.loc[df.label == c, "text_len"], bins=45, stat="density",
                     ax=axes[0], color=PALETTE[c], alpha=0.45, label=CLASS_NAMES[c],
                     element="step", fill=True)
    axes[0].set_xlabel("characters per tweet")
    axes[0].set_ylabel("density")
    axes[0].set_title("Tweet length by class (train)")
    axes[0].legend(frameon=False)

    sns.histplot(df["text_len"], bins=60, ax=axes[1], color="0.4")
    axes[1].axvline(df["text_len"].median(), ls="--", c="tab:red",
                    label=f"median {df['text_len'].median():.0f}")
    axes[1].axvline(df["text_len"].quantile(0.95), ls=":", c="tab:blue",
                    label=f"p95 {df['text_len'].quantile(0.95):.0f}")
    axes[1].set_xlabel("characters per tweet")
    axes[1].set_ylabel("count")
    axes[1].set_title("Overall length distribution")
    axes[1].legend(frameon=False)

    out = FIG / "fig02_text_length.png"
    fig.savefig(out)
    plt.close(fig)
    return {
        "path": out.name,
        "mean_by_class": {str(c): float(df.loc[df.label == c, "text_len"].mean()) for c in (0, 1)},
        "p95": float(df["text_len"].quantile(0.95)),
        "max": int(df["text_len"].max()),
        "truncated_frac": float((df["text_len"] > 0).mean()),
    }


def fig_image_geometry(dfs: dict[str, pd.DataFrame]) -> dict:
    df = pd.concat(dfs.values())
    aspect = df["img_w"] / df["img_h"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].scatter(df["img_w"], df["img_h"], s=3, alpha=0.15, c="#4C72B0", linewidths=0)
    axes[0].set_xlabel("width (px)")
    axes[0].set_ylabel("height (px)")
    axes[0].set_title(f"Image geometry (n={len(df)})")
    axes[0].set_xlim(0, 540)
    axes[0].set_ylim(0, 540)

    sns.histplot(aspect, bins=50, ax=axes[1], color="#55A868")
    axes[1].axvline(1.0, ls="--", c="k", lw=0.8)
    axes[1].set_xlabel("aspect ratio (w/h)")
    axes[1].set_ylabel("count")
    axes[1].set_xlim(0, 4)
    axes[1].set_title(f"median aspect {aspect.median():.2f}")

    out = FIG / "fig03_image_geometry.png"
    fig.savefig(out)
    plt.close(fig)
    return {
        "path": out.name,
        "w_mean": float(df["img_w"].mean()), "h_mean": float(df["img_h"].mean()),
        "w_min": int(df["img_w"].min()), "w_max": int(df["img_w"].max()),
        "aspect_median": float(aspect.median()),
        "square_frac": float((abs(aspect - 1) < 0.05).mean()),
        "kb_mean": float(df["img_bytes"].mean() / 1024),
    }


def fig_image_appearance(dfs: dict[str, pd.DataFrame]) -> dict:
    df = pd.concat(dfs.values(), ignore_index=True)
    metrics = [("brightness", "brightness (0-255)"),
               ("contrast", "contrast (gray std)"),
               ("colorfulness", "colourfulness (Hasler-Suesstrunk)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3))
    info = {}
    for ax, (col, label) in zip(axes, metrics):
        sns.violinplot(data=df, x="label", y=col, hue="label", ax=ax,
                       palette=PALETTE, cut=0, inner="quartile", legend=False)
        ax.set_xticks([0, 1], [CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=6.5)
        ax.set_title(col, fontsize=8)
        for c in (0, 1):
            info[f"{col}_class{c}"] = float(df.loc[df.label == c, col].mean())
    fig.tight_layout()
    out = FIG / "fig04_image_appearance.png"
    fig.savefig(out)
    plt.close(fig)
    info["path"] = out.name
    return info


def fig_examples(dfs: dict[str, pd.DataFrame], n: int = 8) -> dict:
    df = dfs["train"]
    root = MANIFESTS.parent.parent
    rows = []
    for c in (1, 0):
        sub = df[df.label == c].sample(n, random_state=42)
        rows.append(sub)
    sub = pd.concat(rows).reset_index(drop=True)

    ncol = 4
    nrow = int(np.ceil(len(sub) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 4.0))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= len(sub):
            continue
        r = sub.iloc[i]
        img = Image.open(root / r["img_path"]).convert("RGB")
        ax.imshow(img)
        # strip emoji / non-latin glyphs the serif face cannot render
        clean = re.sub(r"[^\x00-\x7f]+", "", str(r["text"])).replace("  ", " ").strip()
        title = f"[{CLASS_NAMES[int(r['label'])]}] {clean[:46]}"
        color = PALETTE[int(r["label"])]
        ax.set_title(title + ("..." if len(clean) > 46 else ""), fontsize=4.6, color=color)
    fig.tight_layout()
    out = FIG / "fig05_examples.png"
    fig.savefig(out)
    plt.close(fig)
    return {"path": out.name}


def fig_top_terms(dfs: dict[str, pd.DataFrame], k: int = 15) -> dict:
    df = dfs["train"]
    counters = {}
    for c in (0, 1):
        cnt = Counter()
        for txt in df.loc[df.label == c, "text"].astype(str):
            for tok in TOKEN_RE.findall(txt.lower()):
                if tok not in STOP and len(tok) > 2:
                    cnt[tok] += 1
        counters[c] = cnt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    info = {}
    for ax, c in zip(axes, (1, 0)):
        top = counters[c].most_common(k)
        words, counts = zip(*top)
        ax.barh(range(len(words))[::-1], counts, color=PALETTE[c])
        ax.set_yticks(range(len(words))[::-1], words, fontsize=8)
        ax.set_xlabel("frequency")
        ax.set_title(f"Top terms - {CLASS_NAMES[c]}")
        info[f"top_{CLASS_NAMES[c]}"] = [(w, int(n)) for w, n in top]
    out = FIG / "fig06_top_terms.png"
    fig.savefig(out)
    plt.close(fig)
    info["path"] = out.name
    return info


def fig_label_length(dfs: dict[str, pd.DataFrame]) -> dict:
    df = dfs["train"].copy()
    df["length_bin"] = pd.cut(df["text_len"], bins=[0, 40, 80, 120, 160, 400],
                              labels=["<40", "40-80", "80-120", "120-160", ">160"])
    rate = df.groupby("length_bin", observed=True)["label"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(rate.index.astype(str), rate["mean"], color="#8172B2")
    ax.axhline(df["label"].mean(), ls="--", c="k", lw=0.8,
               label=f"overall {df['label'].mean():.3f}")
    ax.set_ylabel("sarcastic rate")
    ax.set_xlabel("tweet length (characters)")
    ax.set_title("Sarcasm rate vs text length")
    for i, (m, n) in enumerate(zip(rate["mean"], rate["count"])):
        ax.text(i, m + 0.004, f"n={n}", ha="center", fontsize=6.5)
    ax.legend(frameon=False)
    out = FIG / "fig07_length_vs_label.png"
    fig.savefig(out)
    plt.close(fig)
    return {"path": out.name, "rate_by_bin": {str(i): float(v) for i, v in rate["mean"].items()}}


def main() -> None:
    dfs = load()
    if "train" not in dfs:
        raise FileNotFoundError("run `python -m src.data_prep` first")
    summary = {
        "splits": {k: int(len(v)) for k, v in dfs.items()},
        "class_counts": {
            k: {CLASS_NAMES[c]: int((v["label"] == c).sum()) for c in (0, 1)} for k, v in dfs.items()
        },
    }
    summary["fig01"] = fig_split_balance(dfs)
    summary["fig02"] = fig_text_length(dfs)
    summary["fig03"] = fig_image_geometry(dfs)
    summary["fig04"] = fig_image_appearance(dfs)
    summary["fig05"] = fig_examples(dfs)
    summary["fig06"] = fig_top_terms(dfs)
    summary["fig07"] = fig_label_length(dfs)

    save_json(summary, METRICS / "eda_summary.json")
    log.info("EDA done -> %s", FIG)


if __name__ == "__main__":
    main()
