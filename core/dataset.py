import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import tifffile

class PatchDatasetFromJson(Dataset):
    def __init__(self, json_path, root_dir, transform=None):
        """
        json_path: path to JSON containing structured patch info
        root_dir: dataset root
        transform: optional image transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # full set of possible labels
        self.label_map = {
            "hgg": 0,
            "lgg": 1,
            "mening": 2,
            "metast": 3,
            "normal": 4,
            "nondiagnostic": 5,
            "schwan": 6,
            "pituita": 7

        }

        with open(json_path, 'r') as f:
            data = json.load(f)

        for patient_id, patient_info in data.items():
            patient_tumor_class = patient_info.get("class", None)
            slides = patient_info.get("slides", {})
            
            if patient_tumor_class not in self.label_map:
                print(f"Warning: Unknown tumor class '{patient_tumor_class}' for {patient_id}")
                continue

            for slide_id, slide_info in slides.items():
                for patch_group, patches in slide_info.items():
                    if not patch_group.endswith("_patches"):
                        continue
                    
                    if patch_group == "tumor_patches":
                        label_str = patient_tumor_class
                    else:
                        label_str = patch_group.replace("_patches", "")

                    label_idx = self.label_map.get(label_str)
                    if label_idx is None:
                        print(f"Unknown label '{label_str}', skipping.")
                        continue

                    for rel_path in patches:
                        full_path = os.path.join(self.root_dir, rel_path)
                        self.samples.append((full_path, label_idx))

        print(f"Loaded {len(self.samples)} patches from JSON.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        # convert tif image to numpy array
        image = tifffile.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    

if __name__ == "__main__":
    dataset = PatchDatasetFromJson(
    json_path='/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset/meta/opensrh.json',
    root_dir='/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset/',
    transform=None  # Add your transforms here if needed
    )
    print(f"Dataset size: {len(dataset)}")

    # Print a sample to verify
    sample_image, sample_label = dataset[0]
    print(f"Sample image size: {sample_image.shape}, Label: {sample_label.item()}")