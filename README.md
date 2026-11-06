# SDNET2018 Cross-Surface Generalisation: Bridge Decks + Walls → Pavements

## Overview
This repository measures **surface (domain) shift** in concrete crack detection with ResNet-18 on SDNET2018. Instead of a random train/test split, models are trained on **bridge decks (D) and walls (W)** and tested on an entirely unseen surface: **pavements (P)**.

Two models are compared:

1. **Frozen baseline (linear probe):** ImageNet-pretrained ResNet-18, whole backbone frozen, only the final 2-class layer trained.
2. **Full fine-tune:** same network, classifier head trained for 3 epochs, then all layers trained for 8 epochs at a lower learning rate.

## Dataset and split
Source: SDNET2018 (Utah State University).

| Split | Surfaces | Cracked | Uncracked | Total | Cracked share |
|---|---|---|---|---|---|
| Train | D + W | 4,994 | 21,998 | 26,992 | 18.5% |
| Val | D + W (15% random hold-out) | 882 | 3,884 | 4,766 | 18.5% |
| Test | P (100% of pavements) | 2,608 | 21,726 | 24,334 | 10.7% |

The model never sees pavement images during training or model selection. Pavements contain fewer cracks (10.7%) than the training surfaces (18.5%), so accuracy is not comparable between validation and test; recall, precision and AUROC are.

## Models
- **Frozen baseline:** all backbone parameters frozen; only the `fc` layer (2 classes) is trained for 5 epochs with Adam and class-weighted cross-entropy (weights 3:1 for cracked:uncracked). Best epoch chosen on validation loss.
- **Full fine-tune** (`train_finetune.py`): class-weighted cross-entropy; best epoch chosen on validation AUPRC; temperature scaling fitted on validation; a decision threshold chosen on validation for at least 95% recall is also reported.
- Input 224 × 224. Hardware: NVIDIA RTX 2060 (about 112 s per epoch).

## Results (pavement hold-out, seed 42)

| Model | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Frozen baseline, threshold 0.5 | 0.65 | 0.27 | 0.38 | not computed |
| Full fine-tune, threshold 0.5 | 0.49 | 0.44 | 0.46 | 0.72 |

Frozen baseline confusion matrix (rows actual, columns predicted; cracked, uncracked): `[[712, 1896], [379, 21347]]`.
Fine-tuned confusion matrix: `[[1137, 1471], [1186, 20540]]`.

**Accuracy is misleading here.** Always predicting "uncracked" scores 89.3% on pavements. The baseline reaches about 90.7% and the fine-tuned model 89.1%.

### The surface gap (fine-tuned model)

| Data | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| In-domain validation (D + W) | 0.72 | 0.82 | 0.77 | 0.94 | 0.87 |
| Pavement test (P) | 0.49 | 0.44 | 0.46 | 0.72 | 0.50 |

- Calibration is acceptable on pavements (ECE about 0.05; fitted temperature 1.08).
- A threshold chosen on validation for 95% recall (0.09) does **not** transfer: on pavements it gives recall 0.67 but precision 0.17.

## Interpretation
- Fine-tuning improved recall on pavements (0.27 → 0.44) and F1 (0.38 → 0.46), but a large gap to in-domain performance remains. This is consistent with surface shift.
- The two runs differ in more than freezing (epochs, learning rates, checkpoint-selection metric), so the improvement should not be attributed to fine-tuning alone.
- The cause of the gap (texture, lighting, background, resolution) has not been tested.

## Limitations
- One training seed so far; results from additional seeds will be added with mean ± standard deviation.
- One held-out surface only; other surface splits (for example leave-one-surface-out) are not yet run.
- The validation split is random, not grouped by source photograph. Near-duplicate crops may fall in both train and validation, which can make in-domain scores optimistic.
- The pavement set was evaluated for both models, and it informed the decision to fine-tune, so it is not a pristine one-shot test set.
- Grad-CAM images are qualitative examples, not evidence of correct localisation.

## Project structure
```
Project/
├── data/
│   ├── raw/                  # SDNET2018 (D, W, P folders)
│   └── processed/sdnet_split/ # train / val / test, each with cracked / uncracked
├── models/                   # frozen-baseline checkpoint
├── results/                  # metrics.json, best.pt, Grad-CAM figures
├── src/
│   ├── dataset.py            # DataLoaders and transforms
│   ├── model.py              # ResNet-18, frozen backbone
│   ├── prepare_data.py       # builds the D+W / P split
│   ├── train.py              # frozen-baseline training
│   ├── evaluate.py           # baseline pavement evaluation
│   ├── train_finetune.py     # two-stage full fine-tune + calibration + metrics
│   ├── gradcam.py            # Grad-CAM grid (TP / FP / FN / TN)
│   └── aggregate.py          # mean ± std across seeds
├── requirements.txt
└── README.md
```

## How to reproduce
```
pip install -r requirements.txt        # use a CUDA build of PyTorch for GPU training
# download SDNET2018 into data/raw/ (D, W, P), then:
cd src
python prepare_data.py
python train.py                        # frozen baseline
python evaluate.py
python train_finetune.py --data ..\data\processed\sdnet_split --out ..\results\finetune_seed42 --workers 0 --seed 42
python gradcam.py --ckpt ..\results\finetune_seed42\best.pt --data ..\data\processed\sdnet_split --out ..\results\finetune_seed42\gradcam.png
python aggregate.py ..\results\finetune_seed42 ..\results\finetune_seed1 ..\results\finetune_seed2
```
`--workers 0` avoids DataLoader worker problems on Windows.

## Reference
Dorafshan, S., Thomas, R. J., & Maguire, M. (2018). SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. *Data in Brief*, 21, 1664–1668.