# One-Shot Foundation Image Classification

ITCS 6169-091: Computer Vision (Assignment 3)

## Overview

This project explores 1-shot image classification across 16 classes using frozen vision-language foundation models. Out of a 2,400-image dataset, strictly 1 training image per class (16 support images total) is used for adaptation, while the remaining 2,384 images are held out for validation.

The implementation progresses through three iterations:

* **Model 0 (Baseline):** Combines a frozen ViT-B-16-SigLIP backbone with zero-shot text prompt prototypes, a visual support cache, and a residual logit adapter.
* **Model 1 (Intermediate):** Upgrades to a frozen SigLIP-2 SO400M backbone at 384x384 resolution with multi-view support augmentation, fine-tuned cache keys, and dynamic test-time adaptation.
* **Model 2 (Final):** Resolves overfitting and noisy test-time caching by collapsing augmented views into 16 clean per-class visual prototypes, training a strongly regularized 16-parameter residual gain, and applying strict confidence-gated test-time adaptation.

## Validation Results

| Model | Val. Accuracy (%) |
| :--- | :---: |
| Model 0 (Baseline) | 89.60 |
| Model 1 (Intermediate) | 88.76 |
| Model 2 (Final) | 90.90 |

## Environment

* **Platform:** Google Colab
* **Hardware:** NVIDIA A100 GPU
* **Frameworks & Libraries:** Python, PyTorch, `open_clip_torch`, `timm`, `torchvision`, `scikit-learn`, `numpy`, `Pillow`

## Setup and Usage

1. Open a Google Colab runtime configured with an A100 GPU.
2. Install the required dependencies:
   ```bash
   pip install open_clip_torch timm
