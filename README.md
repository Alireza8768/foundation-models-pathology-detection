# Investigation of Foundation Models for Dense Prediction Tasks in Digital Pathology

This repository contains the research code developed as part of a Master’s thesis
investigating the applicability of Vision Transformer–based foundation models
to dense prediction tasks, with a focus on **object detection of mitotic figures**
in digital pathology.

The project explores whether large, pretrained foundation models—originally not
designed for object detection—can be adapted to localize small objects in
high-resolution histopathological images, and whether this adaptation provides
practical benefits over classical convolutional backbones.

---

## Table of Contents

- [Project Context](#project-context)
- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Core Contributions](#core-contributions)
- [Investigated Foundation Models](#investigated-foundation-models)
- [Methodology Overview](#methodology-overview)
  - [Feature Space Analysis](#feature-space-analysis)
  - [Dataset & Preprocessing](#dataset--preprocessing)
  - [Adapting ViT Features for Object Detection](#adapting-vit-features-for-object-detection)
- [Key Findings](#key-findings)
- [Additional Baseline Evaluation](#additional-baseline-evaluation-on-an-independent-dataset)
- [Repository Structure](#repository-structure)
- [Configuration Philosophy](#Configuration-Philosophy)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)
- [How to Start (Experimental)](#how-to-start-experimental)
- [Author](#author)

---
  
## Project Context

- **Degree:** Master’s Thesis  
- **Field:** Applied Computer Science  
- **Focus Area:** Advanced Programming  
- **Institution:** Hochschule Flensburg  
- **Author:** Alireza Teimoury  


---

## Motivation

Foundation models have recently shown remarkable performance across many
computer vision tasks. In digital pathology, several large Vision Transformer
(ViT)–based models such as UNI, Virchow, or H-Optimus-0 have been proposed as
general-purpose feature extractors.

However, these models are:
- not natively designed for object detection,
- output token-based representations instead of pixel-aligned feature maps,
- and lack an inherent multi-scale feature pyramid.

This raises a central research question:

**Can foundation models be meaningfully adapted for dense prediction tasks such
as mitotic figure detection, and is the resulting effort justified compared to
classical detection backbones?**

---

## Research Questions

This project addresses the following questions:

1. Do pathology-specific and generic foundation models produce feature
   representations suitable for dense prediction tasks?
2. Can token-based representations from ViT backbones be adapted to
   object detection architectures?
3. How do different feature pyramid strategies influence detection performance,
   especially for small objects?
4. Do foundation models offer measurable advantages over classical CNN backbones
   when considering performance, training cost, and complexity?

---

## Core Contributions

The main contributions of this work are:

- **Systematic latent-space analysis** of multiple foundation models using
  PCA-based visualization and quantitative metrics.
- **Custom evaluation framework** for comparing feature smoothness, clustering
  behavior, and boundary sharpness across models.
- **Engineering of ViT-compatible feature pyramids** inspired by ViTDet-style
  adaptations.
- **Comparison of feature pyramid strategies**, including MMDetection-native
  FPNs and Detectron2-style high-resolution FPNs.
- **End-to-end object detection benchmarks** using RetinaNet, Faster R-CNN, and
  DINO-DETR.
- **Parameter-efficient fine-tuning** using LoRA to reduce training cost.
- **Final comparison against a classical baseline** (ResNet-50) to assess
  practical benefits.

The emphasis of this repository is on **advanced programming and system-level
design**, rather than proposing a new detection model.

---

## Investigated Foundation Models

The following foundation models were analyzed and benchmarked:

- Pathology-specific:
  - UNI
  - Virchow
  - H-Optimus-0
- Generic vision foundation models:
  - DINOv2 (including giant variants)

Initial model selection was guided by a detailed feature-space analysis before
entering the object detection phase.

---

## Access to Foundation Models

The investigated foundation models were obtained via Hugging Face.
Some of these models require submitting an access request and using an
authentication token in order to download the weights.

Model access therefore depends on approval by the respective model providers
and cannot be fully automated within this repository.

---

## Methodology Overview

### Feature Space Analysis

Frozen backbones were used to extract token-level representations from ViT-based
models. These representations were evaluated using:

- PCA-based dimensionality reduction (PC1–PC3),
- explained variance of leading components,
- feature smoothness across spatial neighborhoods,
- clustering quality using k-means and silhouette scores,
- cosine similarity differences across region boundaries.

Three analysis setups were considered:
- single image patches,
- sequences of multiple patches,
- large mosaic images composed of patches from different WSIs.

---

### Dataset & Preprocessing

- **Dataset:** MIDOG++
- Original WSI resolution: ~7000 × 5000 pixels
- WSIs were tiled into:
  - 1008 × 1008 patches (ViT-14)
  - 1024 × 1024 patches (ViT-16)
- A **20% overlap** was applied to avoid losing mitotic figures at patch borders.
- Annotations were **recomputed and reprojected** for all generated patches.
- Data split:
  - 70% training
  - 15% validation
  - 15% test

All preprocessing steps were implemented via custom scripts.

---

### Adapting ViT Features for Object Detection

Since ViT backbones output token sequences rather than image pyramids, multiple
adaptation strategies were explored:

- ViTDet-style feature pyramid construction
- MMDetection-native FPN implementations
- Detectron2-style high-resolution FPN logic

These approaches were compared both analytically (via PCA and feature analysis)
and empirically during detector training.

---

### Object Detection Experiments

Detection architectures evaluated in this project:

- RetinaNet
- Faster R-CNN
- DINO-DETR

Experiments were conducted in multiple phases:
1. Backbone comparison using RetinaNet
2. Reduced backbone set evaluated with Faster R-CNN
3. ViT-native detection using DINO-DETR
4. Final detector comparison using top-performing backbones
5. LoRA-based fine-tuning for efficiency
6. Benchmark against a ResNet-50 baseline

---

## Key Findings

- Foundation models can be adapted for mitotic figure detection using appropriate
  feature pyramid strategies.
- High-resolution FPNs are crucial for detecting small objects.
- H-Optimus-0 combined with Faster R-CNN yielded the strongest overall results.
- LoRA significantly reduced training time while preserving detection
  performance.
- When compared to a classical ResNet-50 backbone, foundation models did not
  provide a decisive performance advantage relative to their computational and
  engineering complexity.

---

## Additional Baseline Evaluation on an Independent Dataset

To assess generalization beyond the MIDOG++ benchmark, an additional evaluation
was performed on a separate dataset consisting of previously unseen whole-slide
images from a different source.

The best-performing foundation model configuration (H-Optimus-0 with Faster R-CNN)
was compared against a classical ResNet-50 backbone using the same detector setup.

While the overall performance gap was moderate, the foundation model showed
consistently better detection quality and fewer false positives, indicating
a slight but measurable advantage in generalization to unseen data.

---

## Repository Structure

- configs/ Detector and backbone configurations
- src/ custom_mmdet/ Custom backbones, necks, and integration code
- scripts/ Training, evaluation, and preprocessing scripts
- analysis/ PCA and feature-space analysis tools
- results/ Tables and figures generated during experiments

---

## Configuration Philosophy

All experiments follow a shared configuration structure.
Non-essential settings (data pipeline, optimization, training schedule) are kept
identical across experiments, while variations are limited to backbone and neck
components to ensure fair and interpretable comparisons.

---

## Reproducibility Notes

This repository is **research-oriented** and focuses on methodological analysis
rather than providing a fully automated training pipeline.

Reproducing the experiments requires:
- a working MMDetection installation,
- a compatible CUDA environment,
- access to the MIDOG++ dataset (not included due to license restrictions),
- significant GPU resources.

---

## Limitations

- The project does not aim to provide a production-ready detection system.
- Results are constrained by dataset size and available computational resources.
- Only a subset of possible detection architectures could be evaluated.

---

## How to Start (Experimental)

This repository is primarily intended for research and methodological analysis.
Setting up the full training pipeline requires experience with MMDetection,
CUDA-enabled GPUs, and large-scale histopathology datasets.

A typical workflow includes:

1. Setting up an MMDetection environment compatible with your CUDA version
2. Requesting access to required foundation models via Hugging Face
3. Preparing the MIDOG++ dataset or an equivalent WSI dataset
4. Adapting configuration files for the selected backbone and detector
5. Running training and evaluation scripts

Detailed setup instructions are intentionally omitted, as configuration choices
and hardware requirements may vary significantly across systems and research setups.


---

## Author

**Alireza Teimoury**  
Master’s Thesis – Hochschule Flensburg  
Focus: Advanced Programming
