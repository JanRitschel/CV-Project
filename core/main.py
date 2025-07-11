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

# Constants
NUM_CLASSES = 8
NUM_EPOCHS = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_LIST = [16, 32]
LR_LIST = [1e-3, 3e-4]


def get_levit_model(model_name="levit_128s", num_classes=8):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    # Modify input layer for 2 channels
    old_conv = model.patch_embed.proj
    new_conv = nn.Conv2d(
        2, old_conv.out_channels, kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias
    )
    with torch.no_grad():
        new_conv.weight[:, :2] = old_conv.weight[:, :2]
    model.patch_embed.proj = new_conv
    return model


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    for x, y in loader:
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
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
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)


def cross_validate(dataset):
    y_labels = [label for _, label in dataset]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_acc = 0.0
    best_params = {}

    for batch_size in BATCH_SIZE_LIST:
        for lr in LR_LIST:
            fold_scores = []

            for train_idx, val_idx in skf.split(np.zeros(len(y_labels)), y_labels):
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
            print(f"[lr={lr}, batch_size={batch_size}] CV Acc: {avg_score:.4f}")

            if avg_score > best_acc:
                best_acc = avg_score
                best_params = {'batch_size': batch_size, 'lr': lr}

    print("Best parameters:")
    print(best_params)
    print(f"Best CV Accuracy: {best_acc:.4f}")

def main(default_path=None):

    parser = argparse.ArgumentParser(
        description="Trains a LeViT model with cross-validation on a dataset")
    #    parser.add_argument("--version", action="version", version='%(prog)s ' + __version__)
    parser.add_argument("-db", metavar="database", required=True,
                        help="Please provide the path to the database directory: ")
    
    arguments = vars(parser.parse_args())
    print(f"Arguments: {arguments}")
    if arguments["db"]:
        default_path = arguments["db"]
        print(f"Using database path: {default_path}")
    if os.path.isdir(default_path):
        dataset = PatchDatasetFromJson(default_path)
        cross_validate(dataset)
    else:
        print(f"Provided path '{default_path}' is not a valid directory.")
        return


if __name__ == "__main__":
    # Load dataset
    json_path = '/Users/Noah/Uni Koeln Git/Computer Vision/Project/Michigan-Dataset/meta/opensrh.json'

    # Run cross-validation
    main(json_path)
