import torch
import torch.nn as nn
import numpy as np
import stream_data
import pid as pid_module
import model as model_manager
import metrics as metrics_logger
import gc

# gloal variables
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    """
    main training function this connects the data from the genrator to the model passing by the pid operation
    """
    model, initial_shard, start_epoch, optimizer = model_manager.get_model(DEVICE) # initialize the model
    criterion = nn.CrossEntropyLoss()# define the loss functio which is CE as mentioned in the paper

    print(f"\n[SYSTEM] Pipeline initialized on {DEVICE}")
    print(f"[SYSTEM] Resuming from Epoch {start_epoch} at Shard {initial_shard}")

    total_epochs = 10 # hopefully this will be enough     
    for epoch in range(start_epoch, total_epochs):
        
        print(f"\n{'='*40}")
        print(f"STARTING EPOCH {epoch}")
        print(f"{'='*40}")

        train_gen = stream_data.get_next_train_batch(start_shard=initial_shard, batch_size=BATCH_SIZE) # get the train generator
        batch_in_shard_counter = 0 # each shard has 6400 images which is 100 batch so this tracks that
        current_active_shard = initial_shard # in case we are resuming from somewhere not the start (shard 20)

        for batch_idx, (pil_images, labels) in enumerate(train_gen, start=1): # this is the main training loop and it eneminates the data from the generator (that yeilds batches) and passes it to the model 
            model.train() # set the model to training mode to be able to update the weights
            batch_in_shard_counter += 1
            
            try:
                processed_list = []
                for img in pil_images:
                    residual = pid_module.apply_pid_algorithm(img) # apply pid 
                    res_t = torch.from_numpy(residual).permute(2, 0, 1).float() # convert to tensors and permute the color channels 
                    processed_list.append(res_t)  
                
                inputs = torch.stack(processed_list).to(DEVICE) # stack the tensors in batchs and moves them to gpu
                targets = torch.tensor(labels).to(DEVICE) # same
                del pil_images, processed_list # i faced problem with the memory so i spammed the garbage collector calls to try to ease things a bit
                gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} transformation failed: {e}")
                continue

            optimizer.zero_grad() # resets the gradients to update each batch independently
            outputs = model(inputs) # gets Y(X)
            loss = criterion(outputs, targets) # calculate the loss
            loss.backward() # calculates the gradients based on the loss
            optimizer.step() # updates the weights

            if batch_idx % 10 == 0:
                print(f"[TRAIN] Epoch {epoch} | Shard {current_active_shard} | Batch {batch_in_shard_counter}/100 | Loss: {loss.item():.4f}")

            if batch_in_shard_counter >= 100:
                print(f"\n[MILESTONE] Finished Shard {current_active_shard}. Saving checkpoint...")

                # save the checkpoint BEFORE validation: training for this shard is
                # already done and the weights are final, so we don't want a hung/
                # interrupted validation pass (which streams a fresh HF cache) to
                # cost us the whole shard's progress on the next rerun
                model_manager.save_checkpoint(model, optimizer, current_active_shard, epoch, loss.item())

                # free whatever's left from this shard's training batches before validation
                # (which will itself build/read the memmapped val cache) kicks off
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"[MILESTONE] Validating Shard {current_active_shard}...")
                run_val_cycle(model, current_active_shard, epoch, optimizer, loss.item()) # run validation after each shard (accuracy is logged only, checkpoint already safe)
                current_active_shard += 1 # move to the next shard
                batch_in_shard_counter = 0
                
                if current_active_shard > 199: # break and move to the next epoch
                    print(f"[INFO] Completed shard 199. Ending Epoch {epoch}.")
                    break

        initial_shard = 20

def run_val_cycle(model, shard_idx, epoch, optimizer, latest_loss):
    """
   evaluate the model on a cached validation set

   val_images_full is a numpy memmap (disk-backed, not resident in RAM as a whole).
   we slice it batch by batch -- each slice only pages in that small chunk from
   disk -- and convert just that chunk to a torch tensor before moving it to the
   GPU, instead of ever holding the full ~6400-image validation set in RAM.
    """
    model.eval() # set the model to evaluation mode to make sure no weights are updated and no data leakage
    val_images_full, val_labels_full = stream_data.get_val_split() # get the memory-mapped validation set (built/loaded lazily)
    
    correct = 0
    total = 0
    val_batch_size = 32 
    
    print(f"[VAL] Evaluating on {len(val_images_full)} images...")
    
    with torch.no_grad():
        for i in range(0, len(val_images_full), val_batch_size):
            # only this small slice gets read from disk / paged into RAM
            val_chunk_np = val_images_full[i : i + val_batch_size]
            val_labels_np = val_labels_full[i : i + val_batch_size]

            val_chunk = torch.from_numpy(np.array(val_chunk_np, dtype=np.float32)).to(DEVICE)
            val_labels = torch.from_numpy(val_labels_np).to(DEVICE)

            outputs = model(val_chunk)
            _, predicted = torch.max(outputs.data, 1)
            
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()

            del val_chunk_np, val_chunk, val_labels # drop references to each chunk immediately after use

    accuracy = 100 * correct / total
    print(f"[RESULT] Accuracy: {accuracy:.2f}% | Shard Loss: {latest_loss:.4f}")
    # NOTE: checkpoint is now saved by train() right before this function is called,
    # so a hung/interrupted validation pass no longer risks losing the shard's progress

    # auto-save metrics + regenerate learning curve plot, and update best_model.pth
    # if this shard set a new high-water mark for validation accuracy
    metrics_logger.log_metrics(epoch, shard_idx, latest_loss, accuracy)
    model_manager.save_best_checkpoint(model, optimizer, shard_idx, epoch, accuracy)

    # release any GPU/CPU memory held from validation before the next shard's training starts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()