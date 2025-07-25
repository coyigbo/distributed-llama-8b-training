# Distributed LLaMA Training with PEFT

Large-scale LLM training implementation with distributed data parallel, developed as part of a project for NYU's Big Data and Machine Learning Systems course, leveraging NYU’s High Performance Computing (HPC) clusters for distributed training.

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
