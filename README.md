# Misogyny Meme Detection

A multimodal deep learning system for detecting misogynistic memes in code-mixed Chinese-English content, developed as part of the **Shared Task on Misogyny Meme Detection — LT-EDI@LDK 2025**.

📄 **Paper:** [ACL Anthology :LT-EDI 2025](https://aclanthology.org/2025.ltedi-1.22/)

---

## Overview

This repository contains the code and notebook for our submission to the LT-EDI@LDK 2025 shared task on misogyny meme detection. The task involves classifying memes as **Misogynistic** or **Non-Misogynistic**, where memes consist of both an image and text in a code-mixed Chinese-English format typical of online communication.

We explore both unimodal (text-only, image-only) and multimodal architectures, finding that combining **CharBERT + BiLSTM** for text with **Vision Transformer (ViT)** for images via a Gated Multimodal Unit (GMU) fusion achieves the best performance.

---

## Dataset

The dataset is provided under the LT-EDI@LDK 2025 shared task (Ponnusamy et al., 2024; Chakravarthi et al., 2024), consisting of code-mixed Chinese-English memes split into training, development, and test sets.

| Split | Misogyny | Non-Misogyny | Total |
|-------|----------|--------------|-------|
| Train | 349 | 841 | 1,190 |
| Dev   | 47  | 123 | 170   |
| Test  | 104 | 236 | 340   |
| **Total** | **500** | **1,200** | **1,700** |

The dataset is class-imbalanced, with non-misogynistic memes outnumbering misogynistic ones roughly 2.4:1.

---

## Preprocessing

**Text:**
- Removed URLs, emojis, punctuation, and numbers
- Converted Traditional Chinese to Simplified Chinese
- Tokenized using [jieba](https://pypi.org/project/jieba/)
- Transliterated to Romanized Chinese using [pypinyin](https://pypi.org/project/pypinyin/)

**Images:**
- Converted to RGB and resized to 224×224 pixels
- Contrast and brightness enhanced for improved visual quality

---

## Data Augmentation

To address class imbalance, augmentation was applied exclusively to the Misogyny class:

**Image augmentation** (via `torchvision`): brightness adjustment, grayscale conversion, and posterization.

**Text augmentation** (via [deep-translator](https://github.com/nidhaloff/deep-translator)): back-translation through French, German, and Spanish to generate semantically equivalent paraphrases.

| Class | Before | After |
|-------|--------|-------|
| Misogyny | 349 | 841 |
| Non-Misogyny | 841 | 841 |
| **Total** | **1,190** | **1,682** |

---

## Models

### Unimodal — Text

| Model | Notes |
|-------|-------|
| CharBERT-base-Chinese | Baseline |
| CharBERT + BiLSTM | 2-layer BiLSTM (hidden size 128 → 256-dim embeddings) |

**Training config:** max sequence length 128, Adam optimizer, lr = 1e-4, batch size 16, 20 epochs.

### Unimodal — Image

| Model | Feature Dim |
|-------|-------------|
| CLIP (visual encoder) | 512 |
| Vision Transformer (ViT) | 768 |
| ResNet-50 | 2,048 |
| EfficientNet-B0 | 1,280 |

All images resized to 224×224 and normalized with ImageNet statistics.

### Multimodal

| Text Encoder | Image Encoder | Fusion |
|---|---|---|
| CharBERT + BiLSTM | ViT | GMU ⭐ *Best Model* |
| CharBERT | ResNet-50 | GMU |
| CharBERT + BiLSTM | CLIP | GMU |
| CharBERT | EfficientNet-B0 | Concat + MLP |

All multimodal models trained with: class-weighted cross-entropy loss, AMP, learning rate scheduling, dropout, early stopping, and gradient clipping.

---

## Results

Performance is reported using weighted Precision (P), Recall (R), and F1, with **Macro-F1** as the primary metric.

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| CharBERT + BiLSTM (text only) | — | — | 0.81 |
| ViT (image only) | — | — | 0.65 |
| CLIP (image only) | — | — | 0.42 |
| **CharBERT + BiLSTM + CLIP (multimodal)** | **0.81** | **0.84** | **0.82** |

The best-performing model fused CharBERT + BiLSTM text embeddings (256-dim) with CLIP image features (512-dim) using GMU-style fusion, achieving an F1 of **0.82**.

---

## Repository Structure
```
├── misogynymeme (3).ipynb   # Main training and evaluation notebook
└── README.md
```

---

## Citation

If you use this work, please cite our paper:
```
@inproceedings{...,
  title     = {Misogyny Meme Detection},
  booktitle = {Proceedings of LT-EDI@LDK 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.ltedi-1.22/}
}
```

---

## Acknowledgements

This work was developed as part of the Shared Task on Misogyny Meme Detection at LT-EDI@LDK 2025, organized by Ponnusamy et al. and Chakravarthi et al.
