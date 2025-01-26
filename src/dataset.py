import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Process image and mask
        image = self.feature_extractor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        mask = np.array(mask.resize((512, 512), Image.NEAREST))
        mask = torch.tensor(mask, dtype=torch.long)

        return {"pixel_values": image, "labels": mask}