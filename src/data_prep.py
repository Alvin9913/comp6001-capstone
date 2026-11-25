"""Stage 1 - turn the HuggingFace MMSD parquet shards into a local dataset.

Downloads the `mmsd-v1` config of `coderchen01/MMSD2.0` (the original
Cai, Cai & Wan 2019 split: 19,816 / 2,410 / 2,409), decodes the inline
image bytes into JPEG files, and writes one manifest CSV per split with
per-image statistics that the EDA stage reuses.

Run:
    python -m src.data_prep            # download + decode (all splits)
    python -m src.data_prep --skip-download
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

try:  # package imports when run as `python -m src.data_prep`
    from .config import HF_BASE, IMAGES, LOGS, MANIFESTS, RAW, SPLIT_FILES
    from .utils import get_logger, save_json
except ImportError:  # direct script run
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import HF_BASE, IMAGES, LOGS, MANIFESTS, RAW, SPLIT_FILES
    from src.utils import get_logger, save_json

log = get_logger("data_prep", LOGS / "data_prep.log")

CSV_FIELDS = [
    "idx", "id", "split", "label", "text", "text_len",
    "img_path", "img_w", "img_h", "img_bytes",
    "brightness", "contrast", "colorfulness",
]


# ------------------------------------------------------------------ download
def download(url: str, dest: Path, retries: int = 3) -> Path:
    """Download with a .part file so an interrupted run can resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip (exists) %s", dest.name)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(part, "ab") as f:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if total and have % (200 << 20) < (1 << 22):
                        log.info("  %s %.0f/%.0f MB", dest.name, have / 2**20, total / 2**20)
            if part.stat().st_size == 0:
                raise RuntimeError("empty download")
            part.replace(dest)
            log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 2**20)
            return dest
        except Exception as exc:  # noqa: BLE001
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, dest.name, exc)
    raise RuntimeError(f"could not download {url}")


# ------------------------------------------------------------- image stats
def image_stats(img: Image.Image) -> tuple[float, float, float]:
    """brightness, contrast (std) and Hasler-Suesstrunk colorfulness."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    return brightness, contrast, colorfulness


def _iter_rows(parquet_path: Path, batch_size: int = 128):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        tbl = batch.to_pydict()
        yield tbl


def decode_split(split: str, paths: list[Path], writer: csv.DictWriter) -> int:
    """Decode one split, writing images + manifest rows. Returns rows written."""
    imgs_dir = IMAGES / split
    imgs_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in paths:
        log.info("decoding %s", p.name)
        for tbl in _iter_rows(p):
            for i in range(len(tbl["text"])):
                image_field = tbl["image"][i]
                raw = image_field["bytes"] if isinstance(image_field, dict) else image_field
                text = tbl["text"][i] or ""
                label = int(tbl["label"][i])
                tid = str(tbl["id"][i])

                try:
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    log.warning("bad image at row %d (%s): %s", n, tid, exc)
                    continue

                # keep originals but cap the long side to save disk
                if max(img.size) > 512:
                    scale = 512 / max(img.size)
                    img = img.resize(
                        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                        Image.BILINEAR,
                    )

                name = f"{n:06d}.jpg"
                out = imgs_dir / name
                img.save(out, "JPEG", quality=92)
                brightness, contrast, colorfulness = image_stats(img)

                writer.writerow(
                    {
                        "idx": n,
                        "id": tid,
                        "split": split,
                        "label": label,
                        "text": text.replace("\n", " ").strip(),
                        "text_len": len(text),
                        "img_path": str(out.relative_to(MANIFESTS.parent.parent)),
                        "img_w": img.width,
                        "img_h": img.height,
                        "img_bytes": out.stat().st_size,
                        "brightness": round(brightness, 3),
                        "contrast": round(contrast, 3),
                        "colorfulness": round(colorfulness, 3),
                    }
                )
                n += 1
                if n % 1000 == 0:
                    log.info("  %s: %d rows", split, n)
    return n


def summarise(rows: dict[str, list[dict]]) -> dict:
    stats: dict = {"splits": {}, "total": 0}
    for split, rs in rows.items():
        labels = np.array([r["label"] for r in rs], dtype=int)
        stats["splits"][split] = {
            "n": len(rs),
            "n_sarcastic": int(labels.sum()),
            "n_non_sarcastic": int((1 - labels).sum()),
            "balance_pos": float(labels.mean()) if len(rs) else float("nan"),
            "text_len_mean": float(np.mean([r["text_len"] for r in rs])),
            "text_len_median": float(np.median([r["text_len"] for r in rs])),
            "text_len_p95": float(np.percentile([r["text_len"] for r in rs], 95)),
            "img_w_mean": float(np.mean([r["img_w"] for r in rs])),
            "img_h_mean": float(np.mean([r["img_h"] for r in rs])),
            "img_bytes_mean": float(np.mean([r["img_bytes"] for r in rs])),
            "brightness_mean": float(np.mean([r["brightness"] for r in rs])),
            "contrast_mean": float(np.mean([r["contrast"] for r in rs])),
            "colorfulness_mean": float(np.mean([r["colorfulness"] for r in rs])),
        }
        stats["total"] += len(rs)

    ids = [r["id"] for rs in rows.values() for r in rs]
    texts = [r["text"] for rs in rows.values() for r in rs]
    stats["unique_id"] = len(set(ids))
    stats["duplicate_text_ratio"] = float(1 - len(set(texts)) / len(texts))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    paths: dict[str, list[Path]] = {}
    for split, files in SPLIT_FILES.items():
        paths[split] = []
        for fname in files:
            dest = RAW / fname
            if not args.skip_download:
                download(f"{HF_BASE}/{fname}", dest)
            if not dest.exists():
                raise FileNotFoundError(dest)
            paths[split].append(dest)

    manifest = MANIFESTS / "manifest.csv"
    per_split: dict[str, list[dict]] = {}
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for split, plist in paths.items():
            log.info("=== %s ===", split)
            decode_split(split, plist, writer)

    # reload manifests for stats + per split csvs
    import pandas as pd

    df = pd.read_csv(manifest)
    for split in SPLIT_FILES:
        sub = df[df["split"] == split].reset_index(drop=True)
        sub.to_csv(MANIFESTS / f"{split}.csv", index=False)
        per_split[split] = sub.to_dict("records")
        log.info("%s: %d rows", split, len(sub))

    stats = summarise(per_split)
    stats["manifest"] = str(manifest)
    save_json(stats, MANIFESTS / "dataset_stats.json")
    log.info("total rows: %d", stats["total"])
    log.info("sarcastic ratio (all): %.3f", df["label"].mean())
    log.info("saved -> %s", manifest)


if __name__ == "__main__":
    main()
