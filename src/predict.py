"""Load the trained CM-GAN and predict sarcasm for a text+image pair.

Two modes:

    python -m src.predict --examples 6
        draws 6 posts from the official test split, prints the true label,
        the predicted probability and the incongruity magnitude

    python -m src.predict --text "great, another Monday" --image data/images/test/001234.jpg
        predicts one arbitrary post

This is a live demo of the model, not of the pipeline: the encoders run on the
fly, so any image and caption can be used, not only ones in the cache.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from . import config as C
from .features import VIS_MEAN, VIS_STD
from .models import build_model, masked_mean

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_TF = transforms.Compose([
    transforms.Resize(int(C.IMG_SIZE * 1.14)),
    transforms.CenterCrop(C.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(VIS_MEAN, VIS_STD),
])


class Predictor:
    """Wraps the two frozen encoders, the trained head and the preprocessing
    statistics, so that raw text + a JPEG path is all the caller needs."""

    def __init__(self, ckpt: Path | None = None):
        from transformers import BertModel, BertTokenizerFast
        from torchvision.models import ResNet50_Weights, resnet50

        try:  # keep the demo output clean: no weight-loading progress bars
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:  # noqa: BLE001
            pass

        # identical standardisation to training: reuse the train-split statistics
        from .train import _VIS_STATS, load_data
        load_data()
        self.vis_mean = torch.tensor(_VIS_STATS["mean"]).to(DEVICE)
        self.vis_std = torch.tensor(_VIS_STATS["std"]).to(DEVICE)

        self.tok = BertTokenizerFast.from_pretrained(C.TEXT_MODEL)
        self.bert = BertModel.from_pretrained(C.TEXT_MODEL).eval().to(DEVICE)
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = torch.nn.Sequential(*list(backbone.children())[:-2]).eval().to(DEVICE)

        self.model = build_model("cmgan", C.PROJ_DIM, C.N_HEADS, C.DROPOUT).to(DEVICE)
        state = torch.load(ckpt or (C.CKPT / "cmgan_final.pt"),
                           map_location=DEVICE, weights_only=True)["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, text: str, image_path: Path) -> dict:
        enc = self.tok(text, max_length=C.TEXT_MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].bool().to(DEVICE)
        txt = self.bert(input_ids=ids, attention_mask=mask.long()).last_hidden_state

        img = Image.open(image_path).convert("RGB")
        x = IMG_TF(img).unsqueeze(0).to(DEVICE)
        feats = self.cnn(x).flatten(2).transpose(1, 2)          # (1, 49, 2048)
        feats = (feats - self.vis_mean) / self.vis_std

        logits, parts = self.model(txt, mask, feats, return_parts=True)
        prob = float(torch.sigmoid(logits).squeeze().cpu())
        return {
            "p_sarcastic": prob,
            "label": "sarcastic" if prob >= 0.5 else "not sarcastic",
            "confidence": prob if prob >= 0.5 else 1 - prob,
            "delta_norm": float(parts["delta"].norm(dim=-1).squeeze().cpu()),
        }


def main() -> None:
    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001  (older/newer transformers)
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=0,
                    help="sample N posts from the official test split instead")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pred = Predictor()
    root = C.MANIFESTS.parent.parent

    if args.examples:
        test = pd.read_csv(C.MANIFESTS / "test.csv")
        sample = test.sample(args.examples, random_state=args.seed)
        print(f"{'caption':<52} {'true':<14} {'predicted':<14} {'p':>6} {'delta':>6}  ok   image")
        print("-" * 128)
        hits = 0
        rows_out = []
        for _, row in sample.iterrows():
            out = pred(row["text"], root / row["img_path"])
            true = "sarcastic" if int(row["label"]) == 1 else "not sarcastic"
            ok = out["label"] == true
            hits += ok
            caption = (str(row["text"])[:49] + "...") if len(str(row["text"])) > 52 else str(row["text"])
            print(f"{caption:<52} {true:<14} {out['label']:<14} "
                  f"{out['p_sarcastic']:>6.3f} {out['delta_norm']:>6.2f}  {'Y' if ok else 'n'}"
                  f"   {row['img_path']}")
            rows_out.append((row, out, true))
        print("-" * 128)
        print(f"{hits}/{len(sample)} correct on this sample")

        sheet = C.FIG / "predict_examples.png"
        render_sheet(rows_out, root, sheet)
        print(f"contact sheet with the images: {sheet}")
        return

    if not args.text or not args.image:
        ap.error("give --examples N, or both --text and --image")
    out = pred(args.text, args.image)
    print(f"text       : {args.text}")
    print(f"image      : {args.image}")
    print(f"prediction : {out['label']}  (p = {out['p_sarcastic']:.3f}, "
          f"confidence {out['confidence']:.3f})")
    print(f"divergence : |Delta| = {out['delta_norm']:.2f}  "
          f"(higher means the two modalities disagree more)")


def render_sheet(rows, root: Path, out_path: Path) -> None:
    """One row per example: the picture, the caption, true label, prediction."""
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.55 * n), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (row, out, true) in zip(axes, rows):
        ax.axis("off")
        img = plt.imread(root / row["img_path"])
        ax.imshow(img, extent=(0.0, 0.16, 0.05, 0.95), aspect="auto")
        caption = str(row["text"])
        caption = "\n".join(textwrap.wrap(caption, 74)[:2])
        ok = out["label"] == true
        ax.text(0.175, 0.72, caption, fontsize=7.4, va="center", ha="left", family="serif")
        ax.text(0.175, 0.22,
                f"true: {true}    predicted: {out['label']}    p = {out['p_sarcastic']:.3f}"
                f"    |Delta| = {out['delta_norm']:.2f}    {'correct' if ok else 'wrong'}",
                fontsize=7.4, va="center", ha="left", family="serif",
                color="#1b5e20" if ok else "#b71c1c")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
