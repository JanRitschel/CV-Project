import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import tifffile

class PatchDatasetFromJson(Dataset):
    def __init__(self, root_path, transform=None, channel_indices=[0, 1]):
        """
        json_path: path to JSON containing structured patch info
        transform: optional image transforms
        """
        self.transform = transform
        self.root_dir = root_path
        self.channel_indices = channel_indices
        json_path = os.path.join(self.root_dir, 'meta', 'opensrh.json')
        json_samples_path = os.path.join(self.root_dir, 'meta', 'samples.json')
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

        # If "Samples" is "None" iterate through the JSON structure to build the 
        # samples list otherwise use the provided samples
        if not os.path.exists(json_samples_path):
            for patient_id, patient_info in data.items():
                patient_tumor_class = patient_info.get("class", None)
                slides = patient_info.get("slides", {})
                
                # Check if the patient tumor class is valid
                if patient_tumor_class not in self.label_map:
                    print(f"Warning: Unknown tumor class '{patient_tumor_class}' for {patient_id}")
                    continue

                # Iterate through slides and their patches
                for slide_id, slide_info in slides.items():
                    for patch_group, patches in slide_info.items():
                        if not patch_group.endswith("_patches"):
                            continue
                        
                        if patch_group == "tumor_patches":
                            # Avoid using the patient tumor class if it is "normal" or None
                            if patient_tumor_class == "normal" or patient_tumor_class is None:
                                print(f"Warning: Patient {patient_id} has 'normal' or no class defined but is in 'tumor_patches'. Skipping.")
                                continue
                            # Use the patient tumor class as label
                            label_str = patient_tumor_class
                        else:
                            # Use the patch group name as label since normal and nondiagnostic patches are not tumor patches
                            label_str = patch_group.replace("_patches", "")

                        # Get the label index from the label map
                        label_idx = self.label_map.get(label_str)
                        if label_idx is None:
                            print(f"Unknown label '{label_str}', skipping.")
                            continue

                        # Append each patch with its label in the samples list
                        for rel_path in patches:
                            full_path = os.path.join(self.root_dir, rel_path)
                            # Check if the file exists
                            if not os.path.exists(full_path):
                                print(f"Warning: Image file not found: {full_path}")
                                continue
                            # Append the full path and label index to samples
                            self.samples.append((full_path, label_idx))
            # Save samples as json file
            with open(os.path.join(self.root_dir, 'meta', 'samples.json'), 'w') as f:
                json.dump(self.samples, f, indent=4)
        else:
            with open(json_samples_path, 'r') as f:
                self.samples = json.load(f)
        
        print(f"Loaded {len(self.samples)} patches from JSON.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the image and label for a given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label) where image is a numpy array and label is a torch tensor.
        Raises:
            FileNotFoundError: If the image file does not exist.
            IOError: If the image cannot be read.
            KeyError: If the label is not found in the label map.
        """

        img_path, label = self.samples[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if len(self.channel_indices) > 1:
            image = tifffile.imread(img_path)
        else:
            image = tifffile.imread(img_path)[self.channel_indices[0], :, :]
            image = np.expand_dims(image, axis=0)  # Add channel dimension if only one channel is used
        image = torch.from_numpy(image).float()
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    

if __name__ == "__main__":
    dataset = PatchDatasetFromJson(
        root_path='/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset/meta/opensrh.json',
        transform=None  # Add your transforms here if needed
        )
    print(f"Dataset size: {len(dataset)}")

    # Print a sample to verify
    sample_image, sample_label = dataset[0]
    print(f"Sample image size: {sample_image.shape}, Label: {sample_label.item()}")