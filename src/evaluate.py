import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ametrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
from model import get_model
from load_data import dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Evaluate on test set
test_loader = dataloaders["test"]
all_preds = []
all_labels = []
all_probs = []
all_images = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = outputs.cpu().numpy().flatten()
        preds = (outputs > 0.5).float().cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_images.extend(inputs.cpu().numpy())

# Save predictions
df = pd.DataFrame({
    "True Label": all_labels,
    "Predicted": all_preds,
    "Probability": all_probs
})
df.to_csv("test_predictions.csv", index=False)
print(" Saved test_predictions.csv")

# Classification Report
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")


# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png")


# Misclassified Image Visualization
print("\n Showing some misclassified images...")
misclassified = np.where(np.array(all_preds) != np.array(all_labels))[0]
if len(misclassified) > 0:
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(misclassified[:6]):
        img = np.transpose(all_images[idx], (1, 2, 0))  # C x H x W -> H x W x C
        label = int(all_labels[idx])
        pred = int(all_preds[idx])
        prob = all_probs[idx]
        plt.subplot(2, 3, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {label} | Pred: {pred} | Prob: {prob:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("misclassified_samples.png")
    
else:
    print(" No misclassified images found.")
