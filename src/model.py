import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import glob
import re
import json

# CHECKPOINT_DIR is env-var driven so the exact same code works unchanged across
# Colab (pointed at a mounted Google Drive folder) and Kaggle (pointed at
# /kaggle/working, or a copied-in path from a Kaggle Dataset input) -- just set
# PID_CHECKPOINT_DIR before running. Falls back to a local relative folder otherwise.
CHECKPOINT_DIR = os.environ.get(
    "PID_CHECKPOINT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # ensure it exists

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
TRAINING_STATE_PATH = os.path.join(CHECKPOINT_DIR, "training_state.json")

def _init_architecture():
    """
    create the custom architecture of the ResNet50 model by replacing the last fully connected layer by a new one for binary classification instead of the default 1000 classes
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT) # loads the weights for the pretrained model 
    num_ftrs = model.fc.in_features  # gets the num of input features for the last layer  
    model.fc = nn.Linear(num_ftrs, 2) # create the final layer for the binary classification 
    
    # unfreeze all the parameters to start teaching the model the new noise patterns in the residuals
    for param in model.parameters():
        param.requires_grad = True  
        
    return model

def get_model(device):
    """
    basically loads the latest version of the model and optimizer in order to resume training or to use the weights for inference or testing 
    returns  --> model (weights) , shard number (the shard to resume training from) , start epoch (the epoch to resume training from), optimizer ( the previous state of the optimizer  )
    """
    model = _init_architecture().to(device) # initially loads the custom model to the gpu
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # create a new optimizer with clean history in caase of a new model (usually this is overwritten)
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_shard_*.pth"))# locates the checkpoints directory
    # If no checkpoints exist just return a fresh one with the custom architecture
    if not checkpoint_files:
        print("--- No checkpoint found. Starting fresh training (LR=0.0001). ---")
        return model, 20, 0, optimizer # Default start shard is 20

    latest_file = checkpoint_files[0] # there is always one checpoint file saved in the directory so it is safe to take the first one
    match = re.search(r'checkpoint_shard_(\d+).pth', latest_file) # in earlier versions i used the name of the file to determine the shard number of course that is obsolete and error prone and unnecessary i matter of fact as the latest shard is saved in the checkpoint file but i was so lazy ( mostly afraid lol ) to change it
    shard_num = int(match.group(1)) if match else 20

    print(f"--- Loading checkpoint: {os.path.basename(latest_file)} ---")
    
    checkpoint = torch.load(latest_file, map_location=device) # loads the latest checkpoint data to the gpu
    model.load_state_dict(checkpoint['model_state_dict']) # update the weights with the retrieved ones
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # update the optimizer state with the retrieved one
    start_epoch = checkpoint.get('epoch', 0) # you know it 
    
    print(f"--- Resuming from Shard {shard_num}, Epoch {start_epoch} ---")
   
    return model, shard_num, start_epoch, optimizer

def save_checkpoint(model, optimizer, shard_idx, epoch, loss):
    """
   this saves the current state of the model ( weights , optimizer ,  curr_epoch, curr_shard )   to a checkpoint file
    """
    old_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_shard_*.pth")) # locates old ROLLING checkpoints only (never touches best_model.pth)
    #  deletes the old checkpoints (if any) to ensure there is only one latest version
    for old_file in old_checkpoints:
        try:
            os.remove(old_file)
        except OSError:
            pass 

    new_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_shard_{shard_idx}.pth") # generate the new path using the shard number (OBSEEELLLETTTEEE)
    # make the new state to be saved
    state = {  
        'shard': shard_idx,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    # write to a temp file first, then atomically rename into place. torch.save()
    # itself is NOT atomic -- if training gets interrupted (^C, disconnect, kernel
    # restart) exactly while it's writing, a direct save to new_path could leave a
    # truncated/corrupt .pth. os.replace() is an atomic filesystem rename on POSIX,
    # so new_path is either the old complete file or the new complete file, never a
    # partial write.
    tmp_path = new_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, new_path)
    print(f"--- Checkpoint Saved: Shard {shard_idx}, Epoch {epoch}, Loss {loss:.4f} ---")


def _load_training_state():
    """reads the small json file tracking the best validation accuracy seen so far.
    kept separate from the .pth checkpoints so it's cheap to read/write every shard,
    and persists across session restarts (Kaggle/Colab) same as the checkpoints do."""
    if os.path.exists(TRAINING_STATE_PATH):
        with open(TRAINING_STATE_PATH, "r") as f:
            return json.load(f)
    return {"best_val_accuracy": -1.0, "best_shard": None, "best_epoch": None}


def _save_training_state(state):
    with open(TRAINING_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def save_best_checkpoint(model, optimizer, shard_idx, epoch, val_accuracy):
    """
    saves a SEPARATE best_model.pth, but only when val_accuracy beats the best
    seen so far (tracked persistently in training_state.json). this keeps the
    single-best-performing snapshot around independent of the rolling
    checkpoint_shard_*.pth (which is just "most recent", not "best").
    returns True if this shard was a new best and got saved, False otherwise.
    """
    state = _load_training_state()
    if val_accuracy > state.get("best_val_accuracy", -1.0):
        checkpoint = {
            'shard': shard_idx,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
        }
        tmp_path = BEST_MODEL_PATH + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, BEST_MODEL_PATH) # atomic rename, same reasoning as save_checkpoint above
        state["best_val_accuracy"] = val_accuracy
        state["best_shard"] = shard_idx
        state["best_epoch"] = epoch
        _save_training_state(state)
        print(f"--- New Best Model Saved: Shard {shard_idx}, Epoch {epoch}, Val Acc {val_accuracy:.2f}% ---")
        return True
    return False


def load_best_model(device):
    """loads best_model.pth (highest validation accuracy seen so far) -- use this
    for inference/testing instead of the rolling checkpoint, since the rolling
    one is just whatever shard finished most recently, not necessarily the best."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"No best model found at {BEST_MODEL_PATH}")
    model = _init_architecture().to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"--- Loaded Best Model: Shard {checkpoint['shard']}, Epoch {checkpoint['epoch']}, "
          f"Val Acc {checkpoint.get('val_accuracy', 'N/A')} ---")
    return model