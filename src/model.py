import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import glob
import re
import json

# EDIT: checkpoint dir can now be overridden with an environment variable
# (PID_CHECKPOINT_DIR). This matters most on Colab: set it to a path inside
# your mounted Google Drive so checkpoints survive when the runtime resets.
# Falls back to the original "../checkpoints" behavior if unset.
CHECKPOINT_DIR = os.environ.get(
    "PID_CHECKPOINT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"),
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # ensure it exists


def _init_architecture():
    """
    create the custom architecture of the ResNet50 model by replacing the last fully connected layer by a new one for binary classification instead of the default 1000 classes
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)  # loads the weights for the pretrained model
    num_ftrs = model.fc.in_features  # gets the num of input features for the last layer
    model.fc = nn.Linear(num_ftrs, 2)  # create the final layer for the binary classification

    # unfreeze all the parameters to start teaching the model the new noise patterns in the residuals
    for param in model.parameters():
        param.requires_grad = True

    return model


def get_model(device):
    """
    basically loads the latest version of the model and optimizer in order to resume training or to use the weights for inference or testing
    returns  --> model (weights) , shard number (the shard to resume training from) , start epoch (the epoch to resume training from), optimizer ( the previous state of the optimizer  )
    """
    model = _init_architecture().to(device)  # initially loads the custom model to the gpu
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # create a new optimizer with clean history in caase of a new model (usually this is overwritten)
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_shard_*.pth"))  # locates the checkpoints directory
    # If no checkpoints exist just return a fresh one with the custom architecture
    if not checkpoint_files:
        print("--- No checkpoint found. Starting fresh training (LR=0.0001). ---")
        return model, 20, 0, optimizer  # Default start shard is 20

    latest_file = checkpoint_files[0]  # there is always one checpoint file saved in the directory so it is safe to take the first one
    match = re.search(r'checkpoint_shard_(\d+).pth', latest_file)  # in earlier versions i used the name of the file to determine the shard number of course that is obsolete and error prone and unnecessary i matter of fact as the latest shard is saved in the checkpoint file but i was so lazy ( mostly afraid lol ) to change it
    shard_num = int(match.group(1)) if match else 20

    print(f"--- Loading checkpoint: {os.path.basename(latest_file)} ---")

    checkpoint = torch.load(latest_file, map_location=device)  # loads the latest checkpoint data to the gpu
    model.load_state_dict(checkpoint['model_state_dict'])  # update the weights with the retrieved ones
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # update the optimizer state with the retrieved one
    start_epoch = checkpoint.get('epoch', 0)  # you know it

    print(f"--- Resuming from Shard {shard_num}, Epoch {start_epoch} ---")

    return model, shard_num, start_epoch, optimizer


def save_checkpoint(model, optimizer, shard_idx, epoch, loss):
    """
   this saves the current state of the model ( weights , optimizer ,  curr_epoch, curr_shard )   to a checkpoint file
    """
    old_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))  # locates the checkpoints directory
    #  deletes the old checkpoints (if any) to ensure there is only one latest version
    for old_file in old_checkpoints:
        try:
            os.remove(old_file)
        except OSError:
            pass

    new_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_shard_{shard_idx}.pth")  # generate the new path using the shard number (OBSEEELLLETTTEEE)
    # make the new state to be saved
    state = {
        'shard': shard_idx,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, new_path)  # saves the state to the new path
    print(f"--- Checkpoint Saved: Shard {shard_idx}, Epoch {epoch}, Loss {loss:.4f} ---")


# ---------------------------------------------------------------------------
# EDIT: overfitting protection. None of this existed before -- pipeline.py
# only ever overwrote a single rolling checkpoint, so if the model overfit
# late in training, the only saved weights were the overfit ones with no way
# to fall back to an earlier, better-generalizing version.
#
# This adds two things, matching Keras' ModelCheckpoint(save_best_only=True)
# + EarlyStopping(patience=N):
#   1. A SEPARATE "best_model.pth" that only updates when validation
#      accuracy improves (the rolling checkpoint_shard_*.pth from
#      save_checkpoint() above is untouched, and still used for resuming).
#   2. A patience counter persisted to disk (training_state.json) so it
#      survives a Colab crash/restart, not just an in-memory variable.
# ---------------------------------------------------------------------------

_STATE_PATH = os.path.join(CHECKPOINT_DIR, "training_state.json")


def _load_training_state():
    if os.path.exists(_STATE_PATH):
        with open(_STATE_PATH, "r") as f:
            return json.load(f)
    return {"best_val_accuracy": -1.0, "no_improve_count": 0}


def _save_training_state(state):
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f)


def save_best_checkpoint(model, optimizer, shard_idx, epoch, val_accuracy):
    """
    Saves model+optimizer to best_model.pth ONLY if val_accuracy beats every
    previous validation accuracy seen so far. Call this once per validation
    cycle (i.e. right after run_val_cycle computes accuracy).
    Does not touch the rolling checkpoint_shard_*.pth used for resuming.
    """
    state = _load_training_state()

    if val_accuracy > state["best_val_accuracy"]:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save({
            'shard': shard_idx,
            'epoch': epoch,
            'val_accuracy': val_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_path)
        print(f"--- New best model! Val Accuracy: {val_accuracy:.2f}% (prev best: {state['best_val_accuracy']:.2f}%) -> saved to best_model.pth ---")
        state["best_val_accuracy"] = val_accuracy
        state["no_improve_count"] = 0
    else:
        state["no_improve_count"] += 1
        print(f"--- No improvement ({val_accuracy:.2f}% <= best {state['best_val_accuracy']:.2f}%). no_improve_count = {state['no_improve_count']} ---")

    _save_training_state(state)
    return state["no_improve_count"]


def should_early_stop(patience=10):
    """
    Returns True if validation accuracy hasn't improved for `patience`
    consecutive validation cycles (shards). Reads from the persisted
    training_state.json, so this survives crashes/resumes correctly --
    it isn't reset just because the runtime restarted.
    """
    state = _load_training_state()
    return state["no_improve_count"] >= patience


def load_best_model(device):
    """
    Loads the best-validation-accuracy checkpoint (not the latest/resume
    one). Use this for final testing/inference once training is done --
    it protects you from accidentally evaluating an overfit late checkpoint.
    """
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No best_model.pth found in {CHECKPOINT_DIR} yet.")

    model = _init_architecture().to(device)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"--- Loaded best model: Val Accuracy {checkpoint['val_accuracy']:.2f}% (Shard {checkpoint['shard']}, Epoch {checkpoint['epoch']}) ---")
    return model
