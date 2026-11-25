"""Project-wide paths and hyper-parameters.

Every script resolves paths from this module so the whole pipeline can be
moved to another machine / Colab by editing ROOT only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = DATA / "images"
MANIFESTS = DATA / "manifests"
FEATURES = DATA / "features"
HF_CACHE = DATA / "hf_cache"

OUT = ROOT / "outputs"
FIG = OUT / "figures"
METRICS = OUT / "metrics"
CKPT = OUT / "checkpoints"
LOGS = OUT / "logs"

for _p in (DATA, RAW, IMAGES, MANIFESTS, FEATURES, OUT, FIG, METRICS, CKPT, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- dataset
HF_REPO = "coderchen01/MMSD2.0"
HF_SUBFOLDER = "mmsd-v1"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_SUBFOLDER}"
SPLIT_FILES = {
    "train": [f"train-0000{i}-of-00004.parquet" for i in range(4)],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}

# ------------------------------------------------------------- encoders
TEXT_MODEL = "bert-base-uncased"
VISION_ARCH = "resnet50"
IMG_SIZE = 224
TEXT_MAX_LEN = 48
VIS_TOKENS = 49          # 7 x 7 grid after layer4 on 224x224 input
VIS_DIM = 2048           # ResNet-50 layer4 channels
TXT_DIM = 768            # BERT-base hidden size

# --------------------------------------------------------------- model
PROJ_DIM = 256
N_HEADS = 4
DROPOUT = 0.2

# ------------------------------------------------------------ training
SEED = 42
N_FOLDS = 5
EPOCHS = 10
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.10
GRAD_CLIP = 1.0
AMP = True

MODELS = [
    "cmgan",
    "cmgan_nogate",
    "cmgan_nocross",
    "text_only",
    "vision_only",
    "early_fusion",
    "late_fusion",
]
