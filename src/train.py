import os
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, AdamW
import torch.nn.functional as F
import pandas as pd
from dataset import SegmentationDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
num_classes = 2  # Two-class dataset
batch_size = 32
num_epochs = 100
learning_rate = 5e-5
checkpoint_path = "./models/best_model_checkpoint.pth"

# Paths
train_image_dir = "../data/train/images"
train_mask_dir = "../data/train/masks"
valid_image_dir = "../data/valid/images"
valid_mask_dir = "../data/valid/masks"

# Metrics
def compute_metrics(preds, labels, num_classes):
    preds = preds.flatten()
    labels = labels.flatten()
    intersection = torch.zeros(num_classes, dtype=torch.float32)
    union = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        intersection[cls] = ((preds == cls) & (labels == cls)).sum()
        union[cls] = ((preds == cls) | (labels == cls)).sum()

    miou = (intersection / (union + 1e-6)).mean().item()
    pixel_acc = (preds == labels).float().mean().item()
    return miou, pixel_acc

# Load datasets
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, reduce_labels=False)
train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, feature_extractor)
valid_dataset = SegmentationDataset(valid_image_dir, valid_mask_dir, feature_extractor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=num_classes)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Metrics storage
train_losses = []
val_mious = []
val_accuracies = []
best_miou = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Training step
    for batch in train_loader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

    # Validation step
    model.eval()
    total_miou = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=inputs)
            logits = outputs.logits  # (batch_size, num_classes, height, width)

            # Resize logits to match the labels' size
            logits_upsampled = F.interpolate(logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits_upsampled, dim=1)

            # Compute metrics
            miou, accuracy = compute_metrics(preds.cpu(), labels.cpu(), num_classes)
            total_miou += miou
            total_accuracy += accuracy

    avg_miou = total_miou / len(valid_loader)
    avg_accuracy = total_accuracy / len(valid_loader)
    val_mious.append(avg_miou)
    val_accuracies.append(avg_accuracy)
    print(f"Validation mIoU: {avg_miou:.4f}, Pixel Accuracy: {avg_accuracy:.4f}")

    # Save best model
    if avg_miou > best_miou:
        best_miou = avg_miou
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Best model saved with mIoU: {best_miou:.4f}")

# Save metrics and plot
metrics_df = pd.DataFrame({
    "Epoch": list(range(1, num_epochs + 1)),
    "Training Loss": train_losses,
    "Validation mIoU": val_mious,
    "Validation Accuracy": val_accuracies
})
metrics_df.to_csv("training_metrics.csv", index=False)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(metrics_df["Epoch"], metrics_df["Training Loss"], label="Training Loss")
plt.plot(metrics_df["Epoch"], metrics_df["Validation mIoU"], label="Validation mIoU")
plt.plot(metrics_df["Epoch"], metrics_df["Validation Accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Training Metrics")
plt.legend()
plt.grid()
plt.savefig("training_metrics_plot.png")
plt.show()