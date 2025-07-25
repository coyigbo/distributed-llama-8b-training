# Distributed LLaMA Training with PEFT

## NYU Course Project

**Course**: Big Data and Machine Learning Systems  
**Student**: Chuma Oyigbo

## Project Overview

This project implements distributed training of the LLaMA language model using PyTorch's Distributed Data Parallel (DDP) and Parameter-Efficient Fine-Tuning (PEFT). The implementation showcases efficient training practices for large language models in a distributed computing environment.

## Technical Implementation

### Key Features

- **Distributed Training**: Utilizes PyTorch DDP for efficient multi-GPU training
- **Memory Optimization**: Implements gradient accumulation and 8-bit quantization
- **PEFT Integration**: Uses LoRA for parameter-efficient model adaptation
- **Performance Monitoring**:
  - Training and validation loss tracking
  - Perplexity metrics
  - Token processing statistics
  - Learning rate scheduling

### Architecture

- **Model**: LLaMA 3.2B with LoRA adapters
- **Training Strategy**:
  - Distributed data parallel training
  - Cosine learning rate scheduling
  - Early stopping with patience
  - Gradient accumulation steps: 4
  - Learning rate: 2e-5
  - Mixed precision training

### Training Metrics

The training process tracks:

- Training and validation loss
- Perplexity metrics
- Token processing statistics
- Learning rate changes
- Model checkpoints

## Requirements

```
torch
transformers
peft
```

## Usage

To run the training:

```bash
python train_llama_ddp.py
```

The script automatically:

1. Initializes distributed training environment
2. Loads and prepares the dataset
3. Configures the model with LoRA adapters
4. Executes distributed training
5. Saves the best model checkpoint
6. Generates training statistics

## Output

- **Model Checkpoints**: Saved in `best_model_checkpoint/`
- **Training Statistics**: Saved in `training_stats.json`
- **Console Output**: Real-time training metrics including:
  - Loss values
  - Perplexity
  - Learning rates
  - Training progress

## Implementation Details

### Data Processing

- 90/10 train/validation split
- Dynamic batch sizing with gradient accumulation
- Distributed data sampling

### Optimization Techniques

1. **Memory Management**:

   - 8-bit quantization
   - Gradient accumulation (4 steps)
   - Efficient memory clearing

2. **Training Stability**:

   - Cosine learning rate scheduling
   - Gradient clipping
   - Early stopping mechanism

3. **Performance Monitoring**:
   - Comprehensive metric tracking
   - Best model checkpointing
   - Process synchronization

## Academic Context

This implementation was developed as part of the Big Data and Machine Learning Systems course at New York University. The project demonstrates practical application of distributed systems concepts in machine learning, focusing on:

- Distributed computing principles
- Large-scale model training
- Memory-efficient processing
- Performance optimization
- Metric tracking and analysis
