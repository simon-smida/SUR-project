# File        : imageConfig.py
# Date        : 17.4. 2024
# Description : Configuration file for the visual detection part of the project

import torch


NS=5                      # Number of splits (folds) for cross-validation
LR=0.0001                 # Learning rate
EPOCHS=25                 # Number of epochs
BATCH_SIZE=32             # Batch size
CNN_DROPOUT_RATE=0.1      # Dropout rate for CNN
FC_DROPOUT_RATE=0.3       # Dropout rate for FC
DATASET='augmented_data'  # Dataset directory (data/augmented_data/augmented_balanced_data)
LOSS='BCE'                # Loss function (BCEWithLogitsLoss, BCELoss)
OVERSAMPLING=True         # Perform oversampling of minority class

config = {
    "loss"          : LOSS,
    "learning_rate" : LR,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "cnn_dropout"   : CNN_DROPOUT_RATE,
    "fc_dropout"    : FC_DROPOUT_RATE,
    "batch_norm"    : "True",
    "architecture"  : "CNN",
    "dataset"       : DATASET
}

# Format the run name to include key parameters
run_name = (
    f"{config['architecture']}_"
    f"{'OS' if OVERSAMPLING else 'WS'}_"
    f"{config['loss']}_"
    f"{config['dataset']}_"
    f"ep{config['epochs']}_"
    f"bs{config['batch_size']}_"
    f"bn{config['batch_norm']}_"
    f"lr{config['learning_rate']}_"
    f"cnnDp{config['cnn_dropout']}_"
    f"fcDp{config['fc_dropout']}"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")