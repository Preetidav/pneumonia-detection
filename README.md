# pneumonia-detection
Fine-tuning Inception-V3 for Pneumonia Detection from Chest X-rays
# Pneumonia Detection using Inception-V3 (Transfer Learning)

This project fine-tunes an **Inception-V3** model to classify chest X-rays from the **PneumoniaMNIST** dataset into **Pneumonia** or **Normal** categories.

---

## Dataset: PneumoniaMNIST
- Source: [MedMNIST](https://medmnist.com/)
- Contains grayscale chest X-ray images resized to 28×28.
- Split:
  - Train: 388 Normal, 3494 Pneumonia
  - Validation: 135 Normal, 389 Pneumonia
  - Test: 234 Normal, 390 Pneumonia

---

## Model & Pipeline Overview

###  Transfer Learning
- **Base Model**: Inception-V3 pretrained on ImageNet
- **Modifications**:
  - Final FC layer → 1 neuron + sigmoid (binary classification)
  - Input resized to 299×299
- **Augmentations**:
  - Random Horizontal Flip
  - Random Rotation
  - Resize and Normalize

###  Pipeline Steps
1. Load and preprocess the data using `load_data.py`.
2. Define the model in `model.py`.
3. Train the model using `train.py`.
4. Evaluate and visualize results with `evaluate.py` & `plot_123.py`.

To run the full pipeline:
```bash
bash run_pipeline.sh

