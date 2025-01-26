import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
num_classes = 2  # Two-class dataset
checkpoint_path = "./models/best_model_checkpoint.pth"

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=num_classes)
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)

# Load feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, reduce_labels=False)

# Testing and visualization
def visualize_predictions(model, feature_extractor, images, masks, device):
    model.eval()
    for img_path, mask_path in zip(images, masks):
        image = Image.open(img_path).convert("RGB")
        original_mask = Image.open(mask_path)
        input_image = feature_extractor(image, return_tensors="pt")['pixel_values'].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=input_image)

        logits = outputs.logits
        logits_upsampled = F.interpolate(
            logits,
            size=original_mask.size[::-1],  # (width, height) in PIL format
            mode="bilinear",
            align_corners=False,
        )
        predicted_mask = torch.argmax(logits_upsampled, dim=1).squeeze(0).cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image)

        plt.subplot(1, 3, 2)
        plt.title("Original Mask")
        plt.imshow(np.array(original_mask), cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask, cmap="gray")

        plt.show()

# Example usage
sample_images = [
    "../data/test/images/sample1.jpg",
    "../data/test/images/sample2.jpg"
]
sample_masks = [
    "../data/test/masks/sample1_mask.png",
    "../data/test/masks/sample2_mask.png"
]

visualize_predictions(model, feature_extractor, sample_images, sample_masks, device)import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
num_classes = 2  # Two-class dataset
checkpoint_path = "./models/best_model_checkpoint.pth"

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=num_classes)
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)

# Load feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name, reduce_labels=False)

# Testing and visualization
def visualize_predictions(model, feature_extractor, images, masks, device):
    model.eval()
    for img_path, mask_path in zip(images, masks):
        image = Image.open(img_path).convert("RGB")
        original_mask = Image.open(mask_path)
        input_image = feature_extractor(image, return_tensors="pt")['pixel_values'].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=input_image)

        logits = outputs.logits
        logits_upsampled = F.interpolate(
            logits,
            size=original_mask.size[::-1],  # (width, height) in PIL format
            mode="bilinear",
            align_corners=False,
        )
        predicted_mask = torch.argmax(logits_upsampled, dim=1).squeeze(0).cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image)

        plt.subplot(1, 3, 2)
        plt.title("Original Mask")
        plt.imshow(np.array(original_mask), cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask, cmap="gray")

        plt.show()

# Example usage
sample_images = [
    "../data/test/images/sample1.jpg",
    "../data/test/images/sample2.jpg"
]
sample_masks = [
    "../data/test/masks/sample1_mask.png",
    "../data/test/masks/sample2_mask.png"
]

visualize_predictions(model, feature_extractor, sample_images, sample_masks, device)