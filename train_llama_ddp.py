import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import os
from peft import get_peft_model, LoraConfig, TaskType

# Set memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define dataset
data_dir = "/scratch/cco2066/programming-assignment-2/txt_files"

# Define model
model_name = "/scratch/cco2066/Llama3.2-3B-hf"

class TextDataset(Dataset):
    def __init__(self, data_dir):
        self.files_paths = [os.path.join(data_dir, f) for f in os.listdir(
            data_dir) if f.endswith(".txt")]
        self.data = []
        for file_path in self.files_paths:
            with open(file_path, "r") as f:
                self.data.append(f.read())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def train_data_parallelism(rank, world_size, data_dir, batch_size, epochs, learning_rate, gradient_accumulation_steps):
    """Train the Llama model using data parallelism with PEFT"""
    # Set up environment for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize distributed environment
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device
    device = torch.device(f'cuda:{rank}')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Split dataset into train and validation
    full_dataset = TextDataset(data_dir)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Load base model with reduced memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": rank},
        load_in_8bit=True
    )

    # Apply LoRA adapter
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])

    # Define dataset and sampler within each process
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  
        sampler=val_sampler,
        num_workers=0
    )

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    best_val_loss = float('inf')
    patience = 3  
    patience_counter = 0
    training_stats = []

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        total_loss = 0
        total_tokens = 0
        model.train()

        # Training phase
        for i, batch in enumerate(train_loader):
            torch.cuda.empty_cache()

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Calculate number of tokens
            num_tokens = torch.sum(attention_mask).item()
            total_tokens += num_tokens

            # Forward pass
            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask, 
                          labels=input_ids)
            loss = outputs.loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            total_loss += loss.item() * gradient_accumulation_steps

            if (i + 1) % 10 == 0 and rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, "
                    f"Loss: {loss.item()*gradient_accumulation_steps:.4f}, "
                    f"LR: {current_lr:.2e}")

        # Validation phase
        model.eval()
        val_loss = 0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                num_tokens = torch.sum(attention_mask).item()
                val_tokens += num_tokens
                
                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask, 
                              labels=input_ids)
                val_loss += outputs.loss.item()

        # Calculate metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        # Update learning rate
        scheduler.step()
        
        # Save stats
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity,
            'train_tokens': total_tokens,
            'val_tokens': val_tokens,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        training_stats.append(epoch_stats)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if rank == 0:
                # Save best model checkpoint
                model.module.save_pretrained(f"best_model_checkpoint")
                tokenizer.save_pretrained(f"best_model_checkpoint")
        else:
            patience_counter += 1

        epoch_time = time.time() - start_time
        if rank == 0:
            print(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f} seconds\n"
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n"
                f"Train Perplexity: {train_perplexity:.2f}, Val Perplexity: {val_perplexity:.2f}\n"
                f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Synchronize processes
        dist.barrier()

    # Save final training statistics if rank 0
    if rank == 0:
        import json
        with open('training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=4)

    # Clean up
    dist.destroy_process_group()


def main():
    batch_size = 1
    epochs = 3
    learning_rate = 2e-5
    gradient_accumulation_steps = 4 
    world_size = torch.cuda.device_count()

    print(f"Starting training with {world_size} GPUs")
    mp.spawn(train_data_parallelism, args=(world_size, data_dir,
             batch_size, epochs, learning_rate, gradient_accumulation_steps), nprocs=world_size, join=True)
    print("Fine-tuning process completed!")


if __name__ == "__main__":
    main()
