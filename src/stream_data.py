import io
import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import pid as pid

# raise the default HF hub timeout (10s) since large .arrow shard requests
# were timing out repeatedly and forcing endless retries during val cache init
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

_FULL_STREAM = load_dataset("nebula/GenImage-arrow", split="train", streaming=True).take(1_280_000)#loads the dataset (we will work with only half of it cuz 2.18 is enormous headache when living in egypt)
# global variables
NUM_SHARDS = 200
SHARD_SIZE = 6400  
BATCH_SIZE = 64
# path where the built validation cache is persisted so we never have to
# rebuild it from the HF stream more than once per machine.
# stored as a memory-mapped .npy (not torch .pt) so that reading it back
# later doesn't require loading the whole ~4-5GB array into RAM at once.
# env-var driven for the same reason CHECKPOINT_DIR/METRICS_DIR are: same code
# works unchanged on Colab (Drive-mounted) or Kaggle (/kaggle/working) --
# just make sure this points somewhere WRITABLE (Kaggle's /kaggle/input is read-only).
VAL_CACHE_DIR = os.environ.get(
    "PID_VAL_CACHE_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(VAL_CACHE_DIR, exist_ok=True)
VAL_CACHE_IMAGES_PATH = os.path.join(VAL_CACHE_DIR, "val_cache_images.npy")
VAL_CACHE_LABELS_PATH = os.path.join(VAL_CACHE_DIR, "val_cache_labels.npy")
# this is to cache the validation data. _CACHED_VAL_IMAGES is a numpy memmap
# (backed by disk, lazily paged in), NOT a fully-resident in-RAM array
_CACHED_VAL_IMAGES = None
_CACHED_VAL_LABELS = None
# track current state tho the names are a bit misleading as this is an older version and i didnt feel like changing much  
_CURR_TRAIN_INDEX = 20
_CURR_TEST_INDEX = 1

def _process_pil(sample):
    """the data is saved in arrow fromat which sends me a stream of bytes for each image so i have to convert it to a PIL image first also  the is not explicitly mentioned so i had to extract it form the path"""
    try:
        img_data = sample['image'] #get the image (streamof bytes)
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_data = img_data['bytes'] 
        img = Image.open(io.BytesIO(img_data)).convert("RGB") # form the image of the stream and make sure to be in RGB
        path = sample.get('image_path', '')#get the path to use for labeling
        label = 1 if '/ai/' in path.lower() else 0# the logic to ge the label
        return img, label
    except Exception:
        return None, None

def __init_val_cache__():
    """caches the first shard (index 0) for validation ( 6,400 images or 4480 sometimes when the cpu is full )

    IMPORTANT (memory): this used to build a python list of every residual
    tensor and then torch.stack() it, which briefly holds BOTH the list and
    the stacked copy in RAM at once (~2x the data size, ~9GB+ here) and was
    what crashed the Colab session with an OOM. now each residual is written
    directly into a pre-allocated memory-mapped .npy file on disk as it's
    computed, so at no point do we hold more than one image's worth of data
    in RAM. later reads (get_val_split / run_val_cycle) also stay memmap-backed
    so the full validation set is never fully resident in RAM either."""
    global _CACHED_VAL_IMAGES, _CACHED_VAL_LABELS # get the global variables

    if os.path.exists(VAL_CACHE_IMAGES_PATH) and os.path.exists(VAL_CACHE_LABELS_PATH):
        print(f"--- Loading Validation Cache from disk (memory-mapped, {VAL_CACHE_IMAGES_PATH}) ---")
        _CACHED_VAL_LABELS = np.load(VAL_CACHE_LABELS_PATH)
        count = len(_CACHED_VAL_LABELS)
        full_map = np.lib.format.open_memmap(VAL_CACHE_IMAGES_PATH, mode='r')
        _CACHED_VAL_IMAGES = full_map[:count] # trim any unused preallocated rows
        print(f"--- Validation Cache ready: {count} images (memory-mapped, not resident in RAM) ---")
        return

    print("--- Initializing Validation Cache (Shard 0 - 6400 images) ---")

    val_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=0) # get the stream for the validation shard
    img_h, img_w = 256, 256 # matches the fixed resize done inside pid.apply_pid_algorithm

    # preallocate the on-disk array at the max possible size (SHARD_SIZE) so we
    # can write into it incrementally; float16 halves the footprint vs float32
    # with negligible precision impact for validation purposes
    disk_array = np.lib.format.open_memmap(
        VAL_CACHE_IMAGES_PATH, mode='w+', dtype=np.float16, shape=(SHARD_SIZE, 3, img_h, img_w)
    )

    labels_list = []
    count = 0
    for s in val_shard:
        img, lbl = _process_pil(s) # use the helper function to get the image
        if img is not None:
            img_residual = pid.apply_pid_algorithm(img) # apply PiD, shape (H, W, 3) float32
            # write straight into the memmap slot (CHW, float16) -- no accumulation in RAM
            disk_array[count] = np.transpose(img_residual, (2, 0, 1)).astype(np.float16)
            labels_list.append(lbl)
            count += 1

    disk_array.flush()
    del disk_array # release the write-mode memmap handle

    _CACHED_VAL_LABELS = np.array(labels_list, dtype=np.int64)
    np.save(VAL_CACHE_LABELS_PATH, _CACHED_VAL_LABELS)

    # reopen read-only and trim to the actual count (some images may have failed to decode)
    full_map = np.lib.format.open_memmap(VAL_CACHE_IMAGES_PATH, mode='r')
    _CACHED_VAL_IMAGES = full_map[:count]
    print(f"--- Validation Cache saved to disk: {count} images (memory-mapped) ---")

def get_val_split():
    """return the memory-mapped validation images/labels, initializing (building or loading) first if needed.
    NOTE: _CACHED_VAL_IMAGES is a numpy memmap, not a regular in-RAM array -- callers should
    slice small batches from it and convert those slices to torch tensors on demand
    (see run_val_cycle in pipeline.py), rather than pulling the whole thing into RAM."""
    if _CACHED_VAL_IMAGES is None:
        __init_val_cache__()
    return _CACHED_VAL_IMAGES, _CACHED_VAL_LABELS

def get_next_train_batch(   start_shard = 20  , batch_size=BATCH_SIZE):
    """makes the generator for the training data and uses yields and sharding to make the streaming seamless and not have to wait for the whole dataset to load"""
    global _CURR_TRAIN_INDEX
    _CURR_TRAIN_INDEX = start_shard # update the global variable to the starting shard according to when the function is called (start or resuming)
    
    for shard_idx in range(_CURR_TRAIN_INDEX, NUM_SHARDS):
        _CURR_TRAIN_INDEX = shard_idx
        curr_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=shard_idx) # gets the needed shard
        batch_images, batch_labels = [], []
        for sample in curr_shard:
            img, lbl = _process_pil(sample) # process the image
            if img is not None:
                batch_images.append(img)
                batch_labels.append(lbl)
            
            if len(batch_images) == batch_size:
                yield batch_images, batch_labels # yield batch by batch
                batch_images, batch_labels = [], []
        
        if batch_images:
            yield batch_images, batch_labels

def get_test_batch(   start_shard = 1 , batch_size=BATCH_SIZE):
    """same as before but for testing you can fifure that on your own :)"""
    global _CURR_TEST_INDEX
    _CURR_TEST_INDEX = start_shard
    
    for shard_idx in range(_CURR_TEST_INDEX, 20):
        _CURR_TEST_INDEX = shard_idx
        curr_shard = _FULL_STREAM.shard(num_shards=NUM_SHARDS, index=shard_idx)
        
        batch_images, batch_labels = [], []
        for sample in curr_shard:
            img, lbl = _process_pil(sample)
            if img is not None:
                batch_images.append(img)
                batch_labels.append(lbl)
            
            if len(batch_images) == batch_size:
                yield batch_images, batch_labels
                batch_images, batch_labels = [], []
        
        if batch_images:
            yield batch_images, batch_labels