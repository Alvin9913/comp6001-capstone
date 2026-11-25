# Cross-Modal Incongruity Modelling for Multimodal Sarcasm Detection (COMP6001 Capstone)

Reference implementation of **CM-GAN** (Cross-Modal Gated Attention Network), the method proposed
in the accompanying capstone proposal, benchmarked against four ablations/baselines on the
MMSD dataset (Cai, Cai & Wan, 2019).

Everything runs on one NVIDIA GPU (tested on an RTX 3080 Ti, 12 GB) and is fully reproducible from
an empty `data/` directory with the five commands in *How to run*.

**Colab:** `CMGAN_MMSD_capstone.ipynb` is the same pipeline as a single notebook — it writes the
`src/` modules to disk, then runs the five stages, and needs no local files. Open it in Colab, pick a
GPU runtime, and Run all (about 2 hours on a T4; a QUICK MODE cell gives a 15-minute smoke test).

---

## 1. What the code does

| Stage | Module | Output |
|---|---|---|
| 1. Data acquisition | `src/data_prep.py` | 24,635 images + manifests + dataset statistics |
| 2. Dataset EDA | `src/eda.py` | `outputs/figures/fig01..fig07` |
| 3. Feature extraction | `src/features.py` | frozen BERT token cache + ResNet-50 token cache |
| 4. Training / 5-fold CV | `src/train.py` | checkpoints, per-fold metrics, training curves |
| 5. Analysis | `src/evaluate.py` | tables, paired tests, `fig10..fig16` |

The encoders are **frozen**: BERT-base and ResNet-50 are used as fixed feature extractors and only
the fusion/classification network is trained. This keeps every model (CM-GAN and all baselines) on
exactly the same input features, which is what makes the comparison in Criterion 4 fair, and it
keeps a full 5-fold × 7-configuration study inside one GPU-hour.

## 2. Method

```
text   BERT-base  -> (L=48, 768) --+
                                   +--> bidirectional cross-attention --> gated incongruity --> MLP
image  ResNet-50  -> (49, 2048)  --+
```

* **Bidirectional cross-attention** – text queries attend to visual tokens and vision queries attend
  to text tokens, so each modality is contextualised by the other before any comparison happens.
* **Gated incongruity unit** – `Delta = |h_t - h_v| * sigmoid(W_g [h_t ; h_v] + b_g)`. The gate `g`
  learns *which dimensions* of the disagreement are informative, so agreeing/redundant dimensions
  are suppressed instead of being averaged into the classifier.
* **Classification head** – `[h_t ; h_v ; Delta]` → 256 → 1 with dropout + LeakyReLU.

The ablation configurations `cmgan_nogate` (no `Delta`) and `cmgan_nocross` (no cross-attention)
isolate the contribution of each component.

## 3. Benchmarks

| Model | Family | Why it is included |
|---|---|---|
| `text_only` | unimodal | textual upper bound; MMSD is text-dominant |
| `vision_only` | unimodal | visual upper bound; tests how little the image alone carries |
| `early_fusion` | feature-level fusion | the naive "concatenate and classify" baseline |
| `late_fusion` | decision-level fusion | ensemble of the two unimoal heads, mixing weight fitted on fold-validation |
| `cmgan_nocross` / `cmgan_nogate` | ablations | isolate cross-attention and the gate |
| `cmgan` | proposed | full model |

## 4. Repository layout

```
COMP6001_Capstone/
├── run_all.py             # stage-by-stage orchestrator
├── requirements.txt
├── src/
│   ├── config.py          # paths + every hyper-parameter
│   ├── data_prep.py       # download / decode / manifests / statistics
│   ├── eda.py             # dataset figures
│   ├── features.py        # frozen encoder caches
│   ├── models.py          # CM-GAN + baselines (from scratch, PyTorch)
│   ├── train.py           # 5-fold CV + refit + checkpointing
│   ├── evaluate.py        # metrics, statistics, result figures
│   └── utils.py           # seeding, logging, metrics
├── data/                  # created at runtime (images, features) - not submitted
└── outputs/               # figures, metrics, checkpoints, logs
```

## 5. Setup

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.7.1+cu118, torchvision 0.22.1, transformers 4.5x.
CPU-only works but Stage 3/4 become slow (hours instead of minutes).

## 6. How to run

```bash
python run_all.py                 # all five stages, in order
```

or stage by stage:

```bash
python -m src.data_prep           # ~2.7 GB download, ~24.6k JPEGs decoded
python -m src.eda                 # dataset figures
python -m src.features            # BERT + ResNet caches (~7 GB of float16 memmaps)
python -m src.train               # all 7 configurations x 5 folds + refit
python -m src.evaluate            # tables, paired tests, figures
```

Useful flags:

```bash
python run_all.py --from features       # resume after a crash
python run_all.py --only eda
python -m src.train --models cmgan      # single model, 5 folds + refit
python -m src.data_prep --skip-download # rebuild manifests from cached parquet
python -m src.predict --examples 6      # live inference on 6 unseen test posts
python -m src.predict --text "great, another Monday" --image data/images/test/000000.jpg
```

`src/predict.py` loads `outputs/checkpoints/cmgan_final.pt`, runs the two frozen encoders
on whatever text and image you give it, and prints the predicted label, its probability and the
incongruity magnitude — that is the per-sample behaviour the report analyses in Section 7.5.

`late_fusion` needs the `text_only` and `vision_only` checkpoints, so it is always run after them.

## 7. Protocol

* Dev pool = official train + validation (22,226 samples); official test (2,409) is never trained on.
* 5-fold stratified CV over the dev pool; the epoch is chosen by macro-F1 on the held-out fold and
  that checkpoint is scored on the official test split — so all models are compared on the same
  2,409 examples, five times, which is what the paired test in Stage 5 consumes.
* Final model = refit on the whole dev pool for the median best epoch of the folds, scored once on
  the official test split.
* Metrics: accuracy, precision, recall, binary F1, macro F1, ROC-AUC (macro F1 is the headline,
  since MMSD is mildly imbalanced).

## 8. Computational and memory complexity

Let `N` be the number of samples, `L = 48` text tokens, `K = 49` visual tokens, `d = 256` the shared
projection width, `D_t = 768`, `D_v = 2048` the encoder widths and `H = 4` attention heads.

| Component | Time per batch | Memory |
|---|---|---|
| Text projection | `O(B·L·D_t·d)` | `O(B·L·d)` |
| Visual projection | `O(B·K·D_v·d)` | `O(B·K·d)` |
| Cross-attention (x2) | `O(B·H·(L·K·d + L²·d + K²·d))` | `O(B·H·L·K)` |
| Gated incongruity | `O(B·d²)` | `O(B·d)` |
| MLP head | `O(B·d²)` | `O(B·d)` |

With `d = 256` the trainable part is ~1.5 M parameters, so one epoch over 22k samples takes a few
seconds on a 3080 Ti. The dominant cost of the whole project is **not** the model but the one-off
frozen feature extraction: BERT-base forward passes (`O(N·L²·768)` per layer, 12 layers) and
ResNet-50 forward passes (≈4 GFLOPs/image), which is why their outputs are cached as float16
memmaps and reused by all 35 training runs.

Disk: ~2.3 GB of JPEGs, ~2.7 GB of parquet, ~6.7 GB of feature caches (text 1.8 GB + vision 4.9 GB).
Peak RAM: text cache is loaded fully (~1.8 GB); visual cache is memory-mapped so it is streamed
from disk, keeping RSS under ~6 GB.

## 9. Known limitations and possible failure modes

* **Frozen encoders.** CM-GAN never fine-tunes BERT or ResNet, so the headline numbers are below
  the 86–88 % F1 reported by fully fine-tuned SOTA systems (Pan et al., 2020). The ablation design
  is still valid because every model shares the same frozen features, but the absolute numbers
  should not be read as a SOTA claim.
* **Image truncation.** Images are re-encoded as JPEG q92 and any side > 512 px is downscaled during
  `data_prep`. Re-running on the original files would change features slightly.
* **Corrupt rows.** `data_prep` skips images that PIL cannot decode and logs a warning; a skipped
  row shifts later indices, so the manifests (not row order) are the source of truth.
* **Windows specifics.** DataLoader workers are set to 0 on purpose: CUDA + fork on Windows is
  fragile and the cached features make parallel loading unnecessary.
* **Late fusion leakage guard.** The mixing weight `alpha` is fitted only on fold-validation
  predictions; the refit model reuses the median `alpha` of the five folds rather than re-fitting on
  data it was trained on.
* **Threshold.** All models use a fixed 0.5 decision threshold; no per-model threshold tuning, so
  precision/recall trade-offs are comparable but not individually optimised.
* **Single seed.** `SEED = 42` governs the split, initialisation and shuffling. The 5-fold spread is
  reported as the uncertainty estimate; run `python -m src.train --models cmgan` with a different
  `config.SEED` for a multi-seed check.

## 10. Results

Five-fold results on the official test split (mean +- standard deviation over the five folds):

| Model | Accuracy | Precision | Recall | Binary F1 | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| BERT text-only | 0.7915 +- 0.0017 | 0.7276 +- 0.0038 | 0.7614 +- 0.0049 | 0.7441 +- 0.0019 | 0.7841 +- 0.0016 | 0.8623 +- 0.0011 |
| ResNet-50 vision-only | 0.7173 +- 0.0036 | 0.6325 +- 0.0042 | 0.6920 +- 0.0061 | 0.6609 +- 0.0045 | 0.7093 +- 0.0037 | 0.7785 +- 0.0047 |
| Early fusion (concat) | 0.8007 +- 0.0036 | 0.7291 +- 0.0099 | 0.7952 +- 0.0121 | 0.7605 +- 0.0027 | 0.7949 +- 0.0030 | 0.8764 +- 0.0016 |
| Late fusion (decision) | 0.8017 +- 0.0027 | 0.7402 +- 0.0040 | 0.7735 +- 0.0070 | 0.7565 +- 0.0038 | 0.7946 +- 0.0029 | 0.8744 +- 0.0016 |
| CM-GAN without cross-attention | 0.8178 +- 0.0059 | 0.7521 +- 0.0164 | 0.8102 +- 0.0197 | 0.7797 +- 0.0054 | 0.8121 +- 0.0053 | 0.8917 +- 0.0024 |
| CM-GAN without gate | 0.8151 +- 0.0036 | 0.7372 +- 0.0078 | 0.8327 +- 0.0139 | 0.7819 +- 0.0046 | 0.8107 +- 0.0036 | 0.8952 +- 0.0008 |
| CM-GAN (full) | 0.8233 +- 0.0042 | 0.7639 +- 0.0123 | 0.8058 +- 0.0149 | 0.7841 +- 0.0038 | 0.8173 +- 0.0038 | 0.8956 +- 0.0013 |

Paired t-tests against CM-GAN over the five fold-level test scores:

| Comparison | Mean delta macro-F1 | t-test p | Wilcoxon p | Cohen's d |
|---|---|---|---|---|
| vs BERT text-only | +0.0332 | 0.00012 | 0.0625 | 6.63 |
| vs ResNet-50 vision-only | +0.1080 | <0.00001 | 0.0625 | 46.90 |
| vs early fusion | +0.0224 | 0.00128 | 0.0625 | 3.61 |
| vs late fusion | +0.0226 | 0.00093 | 0.0625 | 3.92 |
| vs CM-GAN without cross-attention | +0.0052 | 0.1248 | 0.125 | 0.87 |
| vs CM-GAN without gate | +0.0066 | 0.0665 | 0.0625 | 1.12 |

Reading: CM-GAN (0.817 macro-F1) beats every baseline significantly, but neither ablation is
significantly worse than the full model, so the gain is attributable to the shared-space joint
architecture rather than to the gate alone. The incongruity magnitude does not separate sarcastic
from non-sarcastic posts (9.00 vs 9.72, Welch p = 1.4e-25, i.e. the opposite direction), but it does
predict correctness: accuracy rises from 0.63 to 0.95 across magnitude bins (r = 0.21).

Full tables: `outputs/metrics/tables.md`; machine-readable: `outputs/metrics/results_summary.json`;
figures: `outputs/figures/`.

## 11. References

* Cai, Y., Cai, H., & Wan, X. (2019). Multi-modal sarcasm detection in Twitter with hierarchical
  fusion model. ACL 2019, 2506–2515.
* Castro, S. et al. (2019). Towards multimodal sarcasm detection (an Obviously Perfect Paper).
  ACL 2019, 4619–4629.
* Pan, H. et al. (2020). Modeling intra and inter-modality incongruity for multi-modal sarcasm
  detection. Findings of EMNLP 2020, 1383–1392.

## License

MIT — see [LICENSE](LICENSE).
