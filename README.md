# pneumonia-detection
Fine-tuning Inception-V3 for Pneumonia Detection from Chest X-rays

# Pneumonia Detection from Chest X-Rays using Inception-V3

This project is focused on detecting pneumonia from chest X-ray images using transfer learning with the Inception-V3 model. The task was part of an assignment where the objective was to fine-tune a pre-trained model on the PneumoniaMNIST dataset and evaluate its performance on classifying normal and pneumonia cases.

##  Dataset Used
##  Dataset: PneumoniaMNIST

- Source: [MedMNIST](https://medmnist.com/)
- Format: Grayscale images (28×28), converted to RGB and resized to 299×299
- Splits:
  - **Train**: 388 Normal, 3494 Pneumonia  =  3883 images
  - **Validation**: 135 Normal, 389 Pneumonia =   524 images
  - **Test**: 234 Normal, 390 Pneumonia=        624 images
- Labels:
  - 0 = Normal
  - 1 = Pneumonia
  The dataset is imbalanced, with many more pneumonia images than normal.
---

##  Model Architecture

- **Base Model:** Inception-V3 pretrained on ImageNet.
- **Modifications:**
  - Final fully connected layer changed to a single neuron with sigmoid activation.
  - Auxiliary classifier head also updated.
  - Last Inception block (Mixed_7c) unfrozen for fine-tuning.
- **Input size:** 299×299 RGB (as required by Inception-V3)

---

##  Training Details

- **Loss Function:** Binary Cross Entropy (BCELoss)
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau
- **Epochs:** 10
- **Batch Size:** 32
- **Learning Rate:** 0.0003

###  Data Augmentation

Applied on training data to improve generalization:
- Resize to 299×299
- Random horizontal flip
- Random rotation (±10°)

---

##  Evaluation Metrics

| Metric      | Pneumonia | Normal |
|-------------|-----------|--------|
| Precision   | 0.78      | 0.96   |
| Recall      | 0.99      | 0.54   |
| F1-Score    | 0.87      | 0.69   |
| **Accuracy**  | **82%**     |
| **ROC AUC**   | **0.95**     |

Confusion Matrix:
|                       |  Predicted: Normal |  Predicted: Pneumonia  |
|                       |--------------------|------------------------|
| **Actual: Normal**    |        126         |          108           |
| **Actual: Pneumonia** |         5          |          385           |

## Observations & Insights

### 1. Class Imbalance
- Severe imbalance: 388 Normal vs. 3494 Pneumonia in training
- Model shows **excellent pneumonia detection** (Recall = 0.99)
- **Normal recall is poor (0.54)** due to false positives

### 2. Generalization
- Training metrics improve strongly
- Validation AUC plateaus — slight overfitting
- Suggests need for regularization or early stopping
  
##  Visual Outputs

- "accuracy_plot.png": Shows training/validation accuracy per epoch
- "auc_plot.png": ROC AUC progression
- "confusion_matrix.png": Error distribution between classes
- "roc_curve.png": Model's ROC curve on test set
- "misclassified_samples.png": Example images the model got wrong

## Challenges Faced

- **Imbalanced dataset**: The model performed well on pneumonia but had difficulty with normal cases.
- **Overfitting**: Validation loss plateaued after a few epochs.
- **Mitigation**:
  - Used data augmentation.
  - Monitored validation loss.
  - Saved only the best model.

---

##  How to Run

Install dependencies:
bash
pip install -r requirements.txt

python3 train.py
python3 evaluate.py
python3 plot.py
