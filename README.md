# Distributed LLaMA Training with PEFT, Tensor, and Pipeline Parallelism

[![Made at NYU](https://img.shields.io/badge/Made%20at-NYU-violet)](https://www.nyu.edu)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org)
[![Training Platform](https://img.shields.io/badge/Training%20Platform-NYU%20HPC-blue)](https://hpc.nyu.edu)

I developed this project as part of the **Big Data and Machine Learning Systems** course at **New York University**. It explores large-scale training of an **8-billion parameter LLaMA language model** using three distributed training strategies. All models were trained on **NYU’s High Performance Computing (HPC)** infrastructure using multiple Nvidia A100 GPUs.

---

## Overview

This repository demonstrates three advanced strategies for distributed training of large language models using PyTorch:

1. **Distributed Data Parallel (DDP)** + Parameter-Efficient Fine-Tuning (LoRA)
2. **Tensor Parallelism**
3. **Pipeline Parallelism**

---

## Project Structure

- `train_llama_ddp.py` – DDP + PEFT training script  
- `train_llama_tp.py` – Tensor Parallelism training script  
- `train_llama_pp.py` – Pipeline Parallelism training script
- `README.md` – This file

---

## 1. DDP + Parameter-Efficient Fine-Tuning (LoRA)

Fine-tunes LLaMA 3.2 8B model using PyTorch's DistributedDataParallel (DDP) and LoRA (Low-Rank Adaptation) for memory-efficient adaptation.

**Key Features:**

- Model: LLaMA 3.2B with LoRA adapters
- Distributed Training via PyTorch DDP
- 8-bit quantization (via `bitsandbytes`)
- Gradient accumulation (4 steps)
- Mixed precision training (FP16)
- Cosine learning rate scheduling
- Early stopping with configurable patience
- Gradient clipping

**Monitored Metrics:**

- Training and validation loss
- Perplexity
- Token throughput
- Learning rate changes
- Best model checkpointing

---

## 2. LLaMA Training with Tensor Parallelism

Implements tensor parallelism using PyTorch to split model weights column-wise across multiple GPUs.

**Key Features:**

- Column-wise tensor parallelism (2 GPUs)
- Mixed precision (FP16)
- 8-bit quantization for memory savings
- Custom text data loader
- Gradient synchronization across devices
- Optimizer: AdamW
- NCCL backend for distributed ops

---

## 3. LLaMA Training with Pipeline Parallelism

Trains LLaMA using PyTorch’s `Pipe` for pipeline parallelism, distributing layers across multiple GPUs.

**Key Features:**

- Stage-wise model partitioning
- Gradient checkpointing
- Mixed precision (FP16)
- Cosine LR scheduling
- Gradient clipping
- Early stopping
- Full metric tracking and logging
- Automatic model checkpointing

---

## Key Learnings

- Scalable training using DDP, Tensor, and Pipeline Parallelism
- Memory-efficient fine-tuning via LoRA and quantization
- Leveraged HPC infrastructure for LLM training
- Hands-on experience with distributed PyTorch workflows

---

## Acknowledgements

This work was completed as part of the **Big Data and Machine Learning Systems** course at **New York University**.  
Training was conducted on the **NYU High Performance Computing (HPC)** cluster across multiple Nvidia A100 GPUs.
