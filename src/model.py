from torchvision import models
import torch.nn as nn

def get_model():
    # Use pretrained weights and enable aux_logits as required
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last inception block
    for param in model.Mixed_7c.parameters():
        param.requires_grad = True

    # Replace final FC layer for binary classification
    model.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )

    # Replace aux FC layer as well
    model.AuxLogits.fc = nn.Sequential(
        nn.Linear(768, 1),
        nn.Sigmoid()
    )

    return model
