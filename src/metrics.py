import os
import csv
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display needed/available on Kaggle/Colab batch runs
import matplotlib.pyplot as plt

# METRICS_DIR is env-var driven for the same reason CHECKPOINT_DIR is in model.py:
# same code works unchanged on Colab (Drive-mounted path) or Kaggle (/kaggle/working,
# or copied in from a Kaggle Dataset). Falls back to a local relative folder otherwise.
METRICS_DIR = os.environ.get(
    "PID_METRICS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics")
)
os.makedirs(METRICS_DIR, exist_ok=True)

METRICS_CSV_PATH = os.path.join(METRICS_DIR, "metrics.csv")
LEARNING_CURVE_PATH = os.path.join(METRICS_DIR, "learning_curves.png")

_CSV_HEADER = ["timestamp", "epoch", "shard", "train_loss", "val_accuracy"]


def log_metrics(epoch, shard_idx, train_loss, val_accuracy):
    """
    appends one row per validated shard to metrics.csv (creating it with a header
    the first time), then regenerates the learning curve PNG from the FULL history
    read back off disk. this is intentionally file-based rather than in-memory only,
    so metrics survive a crash/disconnect/session restart on Kaggle or Colab --
    the next session just keeps appending to the same file and the curves pick up
    exactly where they left off.
    """
    file_exists = os.path.exists(METRICS_CSV_PATH)
    with open(METRICS_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_CSV_HEADER)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            epoch,
            shard_idx,
            f"{train_loss:.6f}",
            f"{val_accuracy:.4f}",
        ])

    _plot_learning_curves()


def _read_metrics_history():
    rows = []
    if not os.path.exists(METRICS_CSV_PATH):
        return rows
    with open(METRICS_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _plot_learning_curves():
    rows = _read_metrics_history()
    if not rows:
        return

    steps = list(range(1, len(rows) + 1))  # sequential validated-shard index; simplest x-axis, continuous across epoch boundaries
    losses = [float(r["train_loss"]) for r in rows]
    accs = [float(r["val_accuracy"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(steps, losses, color="tab:red")
    ax1.set_title("Training Loss (per validated shard)")
    ax1.set_xlabel("Validated Shard #")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(steps, accs, color="tab:blue")
    ax2.set_title("Validation Accuracy (per validated shard)")
    ax2.set_xlabel("Validated Shard #")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(LEARNING_CURVE_PATH, dpi=120)
    plt.close(fig)  # avoid accumulating open figures across hundreds of shards
