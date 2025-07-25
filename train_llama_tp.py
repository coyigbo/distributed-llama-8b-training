import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import os
import time
from tqdm import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import wandb
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model and tokenizer
model_name = "/scratch/cco2066/Llama3.2-3B-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files_paths = [os.path.join(data_dir, f) for f in os.listdir(
            data_dir) if f.endswith(".txt")]
        self.data = []
        for file_path in self.files_paths:
            with open(file_path, "r") as f:
                self.data.append(f.read())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def compute_metrics(loss):
    perplexity = math.exp(loss)
    return {"perplexity": perplexity, "loss": loss}

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

def main():
    # Hyperparameters
    batch_size = 4
    learning_rate = 2e-5
    epochs = 10
    world_size = 2
    rank = int(os.environ.get('RANK', '0'))
    max_grad_norm = 1.0
    warmup_steps = 100
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb for experiment tracking
    if rank == 0:
        wandb.init(project="llama-training", config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "model": model_name
        })

    # Initialize distributed training
    logger.info("Initializing distributed training...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=True,
    ).to(rank)

    # Apply tensor parallelism
    parallelize_module(model, parallel_mode="column", devices=[0, 1])

    # Prepare dataset
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    dataset = SimpleDataset(tokenizer, data_dir)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_dataloader) * epochs
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3)
    best_val_loss = float('inf')

    # Training loop
    model.train()
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        # Training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                         disable=rank != 0)
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)

            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=input_ids)
            
            loss = outputs.loss
            total_train_loss += loss.item()
            train_steps += 1
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0:
                train_pbar.set_postfix({'loss': loss.item()})
                wandb.log({
                    "train_batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })

        # Validation
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]",
                          disable=rank != 0)
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)

                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=input_ids)
                
                loss = outputs.loss
                total_val_loss += loss.item()
                val_steps += 1

                if rank == 0:
                    val_pbar.set_postfix({'loss': loss.item()})

        # Compute average losses and metrics
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        train_metrics = compute_metrics(avg_train_loss)
        val_metrics = compute_metrics(avg_val_loss)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}")
            logger.info(f"Average train loss: {avg_train_loss:.4f}")
            logger.info(f"Average validation loss: {avg_val_loss:.4f}")
            logger.info(f"Train perplexity: {train_metrics['perplexity']:.4f}")
            logger.info(f"Validation perplexity: {val_metrics['perplexity']:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_perplexity": train_metrics['perplexity'],
                "val_perplexity": val_metrics['perplexity']
            })

            # Save checkpoint if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, scheduler, epoch + 1, 
                              best_val_loss, checkpoint_dir)

        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

        dist.barrier() 

    if rank == 0:
        wandb.finish()
    logger.info(f"Rank {rank}: Training complete")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
