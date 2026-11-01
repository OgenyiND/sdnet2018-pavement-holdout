# SDNET2018 Cross-Surface Generalization Baseline

## Overview
This repository contains a rigorous evaluation of a baseline Convolutional Neural Network (ResNet-18) on the SDNET2018 dataset. 

The primary goal of this project is to measure **domain shift** (or surface shift) in concrete crack detection. Instead of a standard, optimistic random train/test split, this project evaluates whether a model trained exclusively on Bridge Decks and Walls can generalize to an entirely unseen surface: Pavements.

## Dataset & Methodology
*   **Source Data:** SDNET2018 (Utah State University)
*   **Training Domain:** Bridge Decks (D) + Walls (W)
*   **Validation Domain:** 15% holdout of Bridge Decks (D) + Walls (W) for early stopping.
*   **Test Domain (Holdout):** 100% of Pavements (P). The model never sees pavement images during training or model selection.
*   **Imbalance Handling:** The training set is heavily imbalanced towards "uncracked" images. A weighted Cross-Entropy loss was applied during training to penalize false negatives in the cracked class.

### Data Split Details
| Split | Images (Cracked) | Images (Uncracked) | Total |
| :--- | :--- | :--- | :--- |
| **Train (D + W)** | 5,898 | 21,998 | 27,896 |
| **Val (D + W)** | 1,041 | 3,884 | 4,925 |
| **Test (P Only)** | 2,608 | 21,726 | 24,334 |

## Model Architecture
*   **Base Model:** ResNet-18 (pretrained on ImageNet).
*   **Modification:** The early convolutional layers were frozen to retain generic feature extractors. The final fully connected layer (`fc`) was replaced with a 2-class linear layer (Cracked vs. Uncracked) and trained from scratch.
*   **Input Resolution:** 224x224 (standard ImageNet size).

## Results

### Training & Validation (Bridge Decks + Walls)
The model trained for 5 epochs on the GPU. The best model (Epoch 4) achieved:
*   **Validation Accuracy:** 86.87%
*   **Validation Loss:** 0.3460

### Honest Evaluation (Pavement Holdout)
*The model was evaluated exactly once on the unseen Pavement dataset using the best checkpoint.*

**Overall Accuracy: 91%** 
*(Note: This number is misleading due to class imbalance. See per-class metrics below).*

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Cracked** | 0.65 | 0.27 | 0.38 | 2608 |
| **Uncracked** | 0.92 | 0.98 | 0.95 | 21726 |
| **Macro Avg** | 0.79 | 0.63 | 0.67 | 24334 |

**Confusion Matrix:**
```text
[[  712  1896]  <-- Actual Cracked (Missed 1,896 cracks)
 [  379 21347]] <-- Actual Uncracked


                # Goal: Evaluate cross-surface generalization (Train on Bridge Decks + Walls, Test on Pavements).

Model: ResNet-18 (frozen features, fine-tuned FC layer).

Data Split: 85% Train (D+W), 15% Val (D+W), 100% Test (P).

Validation Accuracy (D+W): ~86.8%

Pavement Holdout Accuracy: 0.91


                    # Key Finding: The Cost of Domain Shift
While the model achieved ~87% accuracy on its training domains, it catastrophically failed to recall pavement cracks (Recall: 27%).

Because the model was trained on Bridge Decks and Walls, it learned features specific to the texture, lighting, and context of those surfaces. When presented with ground-level Pavements, it defaulted to predicting "Uncracked" to minimize training loss (due to the dataset imbalance).

This proves that features learned from elevated bridge structures do not transfer seamlessly to ground-level pavements without domain adaptation, and highlights the danger of relying solely on standard random-split validation metrics.


                # Project Structure
Project/
├── data/
│   ├── raw/                 # Original SDNET2018 dataset (D, W, P folders)
│   └── processed/           # Generated splits (train, val, test)
├── models/                  # Saved model checkpoints (.pth)
├── results/                 # Evaluation logs and JSON outputs
├── src/
│   ├── dataset.py           # PyTorch DataLoaders and Transforms
│   ├── model.py             # ResNet-18 architecture definition
│   ├── prepare_data.py      # Script to split and organize raw data
│   ├── train.py             # Main training loop
│   └── evaluate.py          # Holdout evaluation script
├── requirements.txt
└── README.md


                            # How to Reproduce
Install dependencies: pip install -r requirements.txt

Download SDNET2018 and place the D, W, and P folders into data/raw/.

Run data preparation: cd src && python prepare_data.py

Train the model: python train.py

Evaluate on pavements: python evaluate.py