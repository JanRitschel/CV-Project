import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import timm
import numpy as np
from dataset import PatchDatasetFromJson
from tqdm import tqdm
from torchvision.transforms import Compose, Resize

# Constants
NUM_CLASSES = 8
NUM_EPOCHS = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_LIST = [32]
LR_LIST = [1e-3]


def get_levit_model(model_name="levit_128s", num_classes=8):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # Find the first Conv2d layer with 3 input channels
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            print(f"Replacing input layer: {name}")
            new_conv = torch.nn.Conv2d(
                2, module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:, :2] = module.weight[:, :2]  # copy 2 of 3 input channels
            # set the new conv in the model
            set_module_by_name(model, name, new_conv)
            break

    return model

# Helper function to replace a nested module by name
def set_module_by_name(model, name, module):
    parts = name.split('.')
    for p in parts[:-1]:
        model = getattr(model, p)
    setattr(model, parts[-1], module)



def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    for x, y in tqdm(loader):
        x = x.to(dtype=torch.float32, device=DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dtype=torch.float32, device=DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)


def cross_validate(dataset: PatchDatasetFromJson):
    y_labels = [label for _, label in dataset.samples]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_acc = 0.0
    best_params = {}

    for batch_size in tqdm(BATCH_SIZE_LIST):
        for lr in tqdm(LR_LIST):
            fold_scores = []

            for train_idx, val_idx in tqdm(skf.split(np.zeros(len(y_labels)), y_labels), desc=f"Batch size: {batch_size}, LR: {lr}"):
                train_ds = Subset(dataset, train_idx)
                val_ds = Subset(dataset, val_idx)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

                model = get_levit_model(num_classes=NUM_CLASSES).to(DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
                criterion = nn.CrossEntropyLoss()

                for _ in range(NUM_EPOCHS):
                    train_epoch(model, train_loader, criterion, optimizer)

                acc = evaluate(model, val_loader)
                fold_scores.append(acc)

            avg_score = np.mean(fold_scores)
            tqdm.write(f"[lr={lr}, batch_size={batch_size}] CV Acc: {avg_score:.4f}")

            if avg_score > best_acc:
                best_acc = avg_score
                best_params = {'batch_size': batch_size, 'lr': lr}

    tqdm.write("Best parameters:")
    tqdm.write(best_params)
    tqdm.write(f"Best CV Accuracy: {best_acc:.4f}")
    torch.save(model.state_dict(), "best_model.pth")

def main(default_path=None):

    transform = Compose([
        Resize((224, 224))  # expects torch.Tensor C×H×W
    ])

    parser = argparse.ArgumentParser(
        description="Trains a LeViT model with cross-validation on a dataset")
    #    parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
    parser.add_argument("-db", metavar="database", required=True,
                        help="Please provide the path to the database directory: ")
    
    arguments = vars(parser.parse_args())
    tqdm.write(f"Arguments: {arguments}")
    if arguments["db"]:
        default_path = arguments["db"]
        tqdm.write(f"Using database path: {default_path}")
    if os.path.isdir(default_path):
        
        dataset = PatchDatasetFromJson(default_path, transform=transform)
        cross_validate(dataset)
    else:
        tqdm.write(f"Provided path '{default_path}' is not a valid directory.")
        return


if __name__ == "__main__":
    # Load dataset
    root_path = '/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset'

    # Run cross-validation
    main(root_path)
