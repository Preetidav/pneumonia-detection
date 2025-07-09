import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from model import get_model
from load_data import dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = get_model().to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0003)

# Optional LR scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)


# Training Loop
def train_model(model, dataloaders, epochs=10):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds, all_labels = [], []

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if phase == 'train' and model.training:
                       outputs, aux_out = model(inputs)
                       loss = criterion(outputs, labels) + 0.4 * criterion(aux_out, labels)
                    else:
                         outputs = model(inputs)
                         loss = criterion(outputs, labels)

                    preds = (outputs > 0.5).float()

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds)
            rec = recall_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | AUC: {auc:.4f}")

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    print(" Best model saved.")

# Run training
train_model(model, dataloaders, epochs=10)
