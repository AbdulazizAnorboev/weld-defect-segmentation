import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn.functional as F
from dataset import SegmentationDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
num_classes = 2  # Two-class dataset
batch_size = 32
checkpoint_path = "./models/best_model_checkpoint.pth"

# Paths
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
valid_dataset = SegmentationDataset(valid_image_dir, valid_mask_dir, feature_extractor)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=num_classes)
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)

# Evaluation
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
print(f"Validation mIoU: {avg_miou:.4f}, Pixel Accuracy: {avg_accuracy:.4f}")