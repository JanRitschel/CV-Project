import torch

# Global Constants
NUM_CLASSES = 8  # Number of classes in the dataset
NUM_EPOCHS = 20  # Number of epochs for final training
NUM_CROSS_VAL_EPOCHS = 3  # Number of epochs for each cross-validation fold
NUM_WORKERS = 4  # Number of workers for DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# List of batch sizes to test during cross-validation
BATCH_SIZE_LIST = [16, 32, 64]
# List of learning rates to test during cross-validation
LR_LIST = [1e-2, 1e-3, 1e-4]
K_SPLITS = 3  # Number of splits for cross-validation
DROP_RATE = 0.2  # Dropout rate for the model
SPLIT_SEED = 42  # Seed for random splits. Used for comparability between channels
CHANNEL_LIST = [1]  # Use first channel, can be modified to use more channels