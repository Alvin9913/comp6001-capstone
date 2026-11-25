"""Stage 3 - frozen feature extraction.

BERT-base (frozen) -> token sequence  (N, 48, 768)
ResNet-50  (frozen) -> 7x7 layer4 grid (N, 49, 2048)

Both are cached to disk as float16 memmaps so that every downstream model
(CM-GAN and all baselines) trains on identical features, which is what makes
the benchmark comparison fair and the 5-fold protocol affordable.

Run:
    python -m src.features
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from .config import (
        FEATURES, HF_CACHE, IMG_SIZE, MANIFESTS, TEXT_MAX_LEN, TEXT_MODEL,
        TXT_DIM, VIS_DIM, VIS_TOKENS,
    )
    from .utils import get_logger, save_json
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import (
        FEATURES, HF_CACHE, IMG_SIZE, MANIFESTS, TEXT_MAX_LEN, TEXT_MODEL,
        TXT_DIM, VIS_DIM, VIS_TOKENS,
    )
    from src.utils import get_logger, save_json

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

log = get_logger("features")

VIS_MEAN = (0.485, 0.456, 0.406)
VIS_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ text
class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        enc = self.tok(
            self.texts[i],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        return (
            torch.tensor(enc["input_ids"], dtype=torch.long),
            torch.tensor(enc["attention_mask"], dtype=torch.long),
        )


@torch.no_grad()
def extract_text(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from transformers import BertModel, BertTokenizerFast

    log.info("loading %s", TEXT_MODEL)
    tok = BertTokenizerFast.from_pretrained(TEXT_MODEL)
    bert = BertModel.from_pretrained(TEXT_MODEL).eval().cuda().half()

    n = len(df)
    path = FEATURES / "text_tokens.npy"
    arr = np.memmap(path, dtype=np.float16, mode="w+", shape=(n, TEXT_MAX_LEN, TXT_DIM))
    lengths = np.zeros(n, dtype=np.int32)

    dl = DataLoader(TextDataset(df["text"].tolist(), tok, TEXT_MAX_LEN),
                    batch_size=128, shuffle=False, num_workers=0)
    pos = 0
    for ids, mask in tqdm(dl, desc="BERT tokens"):
        ids, mask = ids.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out = bert(input_ids=ids, attention_mask=mask).last_hidden_state
        b = out.shape[0]
        arr[pos:pos + b] = out.float().cpu().numpy()
        lengths[pos:pos + b] = mask.sum(dim=1).cpu().numpy()
        pos += b
    arr.flush()
    np.save(FEATURES / "text_lengths.npy", lengths)
    log.info("text tokens -> %s shape=%s", path.name, (n, TEXT_MAX_LEN, TXT_DIM))
    return np.asarray(path), lengths


# ---------------------------------------------------------------- vision
class ImageDataset(Dataset):
    def __init__(self, paths: list[str], root: Path, size: int):
        self.paths = paths
        self.root = root
        self.tf = transforms.Compose(
            [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(VIS_MEAN, VIS_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.paths[i]).convert("RGB")
        return self.tf(img)


@torch.no_grad()
def extract_vision(df: pd.DataFrame) -> np.ndarray:
    from torchvision.models import ResNet50_Weights, resnet50

    log.info("loading resnet50 (IMAGENET1K_V2)")
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # keep conv1..layer4 only: torchvision's _forward_impl calls torch.flatten(x, 1)
    # right after avgpool, so replacing avgpool/fc with Identity is not enough.
    net = torch.nn.Sequential(*list(backbone.children())[:-2])
    net = net.eval().cuda().half()

    n = len(df)
    path = FEATURES / "vis_tokens.npy"
    arr = np.memmap(path, dtype=np.float16, mode="w+", shape=(n, VIS_TOKENS, VIS_DIM))

    root = MANIFESTS.parent.parent
    dl = DataLoader(ImageDataset(df["img_path"].tolist(), root, IMG_SIZE),
                    batch_size=64, shuffle=False, num_workers=0)
    pos = 0
    for x in tqdm(dl, desc="ResNet tokens"):
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            feats = net(x)                      # (B, 2048, 7, 7)
        feats = feats.flatten(2).transpose(1, 2)  # (B, 49, 2048)
        b = feats.shape[0]
        arr[pos:pos + b] = feats.float().cpu().numpy()
        pos += b
    arr.flush()
    log.info("vision tokens -> %s shape=%s", path.name, (n, VIS_TOKENS, VIS_DIM))
    return np.asarray(path)


def _cache_ok(path: Path, shape: tuple[int, ...], min_abs: float = 1e-3) -> bool:
    """A crashed extraction leaves a correctly sized but all-zero memmap behind.

    Checking existence alone therefore silently reuses garbage, so sample the
    file and require it to carry actual signal.
    """
    if not path.exists():
        return False
    try:
        arr = np.memmap(path, dtype=np.float16, mode="r", shape=shape)
        step = max(1, shape[0] // 200)
        sample = np.asarray(arr[::step][:200], dtype=np.float32)
        return bool(np.abs(sample).mean() > min_abs)
    except Exception:  # noqa: BLE001  (wrong size / unreadable -> regenerate)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = ap.parse_args()

    frames = []
    # concatenate splits in a fixed order: train, validation, test
    order = [s for s in ["train", "validation", "test"] if s in args.splits]
    for split in order:
        df = pd.read_csv(MANIFESTS / f"{split}.csv")
        df["split"] = split
        frames.append(df)
        log.info("%s: %d rows", split, len(df))
    df = pd.concat(frames, ignore_index=True)
    df["row"] = np.arange(len(df))
    df.to_csv(FEATURES / "index.csv", index=False)

    txt_shape = (len(df), TEXT_MAX_LEN, TXT_DIM)
    vis_shape = (len(df), VIS_TOKENS, VIS_DIM)

    if not _cache_ok(FEATURES / "text_tokens.npy", txt_shape):
        extract_text(df)
    else:
        log.info("text features exist and look valid, skipping")
    if not _cache_ok(FEATURES / "vis_tokens.npy", vis_shape):
        extract_vision(df)
    else:
        log.info("vision features exist and look valid, skipping")

    save_json(
        {
            "n": len(df),
            "order": order,
            "text_shape": [len(df), TEXT_MAX_LEN, TXT_DIM],
            "vis_shape": [len(df), VIS_TOKENS, VIS_DIM],
            "dtype": "float16",
            "text_model": TEXT_MODEL,
            "vision_model": "resnet50 IMAGENET1K_V2 (layer4, 7x7)",
        },
        FEATURES / "features_meta.json",
    )
    log.info("done")


if __name__ == "__main__":
    main()
