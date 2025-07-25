import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipeline.sync import Pipe
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import os
import torch.multiprocessing as mp
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import sys
from datetime import datetime

# Set up logging
def setup_logging(rank):
    log_file = f'training_rank_{rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Get model path and data directory from environment variables or use defaults
model_name = os.getenv('LLAMA_MODEL_PATH', "meta-llama/Llama-3.2-7b-hf")
data_dir = os.getenv('TRAINING_DATA_DIR', "data")


class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=128, is_validation=False, validation_split=0.1):
        self.files_paths = [os.path.join(data_dir, f) for f in os.listdir(
            data_dir) if f.endswith(".txt")]
        
        # Sort files for deterministic split
        self.files_paths.sort()
        
        # Split into train and validation
        split_idx = int(len(self.files_paths) * (1 - validation_split))
        if is_validation:
            self.files_paths = self.files_paths[split_idx:]
        else:
            self.files_paths = self.files_paths[:split_idx]
            
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and preprocess data
        for file_path in self.files_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                text = ' '.join(text.split())
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Create labels for causal language modeling
        labels = inputs["input_ids"].clone()
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLaMA with Pipeline Parallelism')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    return parser.parse_args()


def train_pipeline_parallelism(rank, world_size, data_dir, batch_size, epochs, learning_rate, max_length=128, validation_split=0.1, max_grad_norm=1.0):
    try:
        setup_logging(rank)
        logging.info(f"Initializing process rank {rank} out of {world_size}")
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        device = torch.device(f'cuda:{rank}')
        logging.info(f"Process {rank} using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        logging.info("Loading model...")
        model_cpu = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        num_llama_layers = model_cpu.config.num_hidden_layers

        temp_sequential_layers = [model_cpu.model.embed_tokens]
        for i in range(num_llama_layers):
            temp_sequential_layers.append(model_cpu.model.layers[i])
        temp_sequential_layers.append(model_cpu.model.norm)
        temp_sequential_layers.append(model_cpu.lm_head)

        model_to_pipe = nn.Sequential(*temp_sequential_layers)

        num_total_parts = len(model_to_pipe)
        pipe_balance = [num_total_parts // world_size] * world_size
        pipe_remainder = num_total_parts % world_size
        for i in range(pipe_remainder):
            pipe_balance[i] += 1

        pipe_model = Pipe(model_to_pipe, balance=pipe_balance,
                          chunks=world_size, checkpoint='always')

        # Create train and validation datasets
        train_dataset = TextDataset(data_dir, tokenizer, max_length=max_length, is_validation=False, validation_split=validation_split)
        val_dataset = TextDataset(data_dir, tokenizer, max_length=max_length, is_validation=True, validation_split=validation_split)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        optimizer = AdamW(pipe_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
        pipe_model.train()

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 3

        logging.info(f"Starting training on rank {rank}")
        for epoch in range(epochs):
            try:
                start_time = time.time()
                total_train_loss = 0
                num_train_batches = 0
                
                # Training loop
                pipe_model.train()
                for i, batch_data in enumerate(train_loader):
                    try:
                        input_ids = batch_data["input_ids"].to(pipe_model.devices[0])
                        labels = batch_data["labels"].to(pipe_model.devices[-1])
                        
                        optimizer.zero_grad()
                        
                        if pipe_model.final_stage:
                            outputs = pipe_model(input_ids)
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                            loss.backward()
                            
                            # Add gradient clipping
                            torch.nn.utils.clip_grad_norm_(pipe_model.parameters(), max_grad_norm)
                            
                            total_train_loss += loss.item()
                            num_train_batches += 1
                            
                            if (i + 1) % 10 == 0:
                                logging.info(f"Rank {rank} Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss {loss.item():.4f}")
                        
                        optimizer.step()
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logging.error(f"Error in batch processing: {str(e)}")
                        continue
                
                # Validation loop
                pipe_model.eval()
                total_val_loss = 0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        input_ids = batch_data["input_ids"].to(pipe_model.devices[0])
                        labels = batch_data["labels"].to(pipe_model.devices[-1])

                        if pipe_model.final_stage:
                            outputs = pipe_model(input_ids)
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                            total_val_loss += loss.item()
                            num_val_batches += 1

                if pipe_model.final_stage:
                    avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
                    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
                    epoch_time = time.time() - start_time
                    
                    logging.info(f"Rank {rank} Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f} seconds")
                    logging.info(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                    
                    # Learning rate scheduling
                    scheduler.step(avg_val_loss)
                    
                    # Early stopping and model checkpointing
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        if rank == 0:  
                            checkpoint_path = os.path.join(data_dir, f"checkpoint_epoch_{epoch+1}.pt")
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipe_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': avg_train_loss,
                                'val_loss': avg_val_loss,
                            }, checkpoint_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            logging.info(f"Early stopping triggered after {epoch+1} epochs")
                            break

                dist.barrier()
                
            except Exception as e:
                logging.error(f"Error in epoch {epoch+1}: {str(e)}")
                continue
        
        dist.destroy_process_group()
        logging.info(f"Rank {rank}: Training process complete")
        
    except Exception as e:
        logging.error(f"Critical error in process rank {rank}: {str(e)}")
        raise

def main():
    try:
        args = parse_args()
        
        world_size = torch.cuda.device_count()
        if world_size < 2:
            logging.error("This script requires at least 2 GPUs for pipeline parallelism")
            return
        
        logging.info(f"Pipeline Parallelism with {world_size} GPUs")
        logging.info("Training configuration:")
        logging.info(f"- Batch size: {args.batch_size}")
        logging.info(f"- Epochs: {args.epochs}")
        logging.info(f"- Learning rate: {args.learning_rate}")
        logging.info(f"- Max sequence length: {args.max_length}")
        logging.info(f"- Validation split: {args.validation_split}")
        logging.info(f"- Early stopping patience: {args.patience}")
        logging.info(f"- Max gradient norm: {args.max_grad_norm}")
        
        mp.spawn(
            train_pipeline_parallelism,
            args=(world_size, data_dir, args.batch_size,
                  args.epochs, args.learning_rate, args.max_length,
                  args.validation_split, args.max_grad_norm),
            nprocs=world_size,
            join=True
        )
        logging.info("Pipeline Parallelism process completed")
        
    except Exception as e:
        logging.error(f"Critical error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
