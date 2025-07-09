import numpy as np
from torch.utils.data import DataLoader
from dataset import PneumoniaDataset, train_transform, val_test_transform

# Load data
data = np.load("pneumoniamnist.npz")
train_images, train_labels = data['train_images'], data['train_labels']
val_images, val_labels = data['val_images'], data['val_labels']
test_images, test_labels = data['test_images'], data['test_labels']

# Create datasets
train_dataset = PneumoniaDataset(train_images, train_labels, transform=train_transform)
val_dataset = PneumoniaDataset(val_images, val_labels, transform=val_test_transform)
test_dataset = PneumoniaDataset(test_images, test_labels, transform=val_test_transform)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Save for reuse
dataloaders = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader
}


# Optional: Check shapes
print("Data loaded and DataLoaders created.")
