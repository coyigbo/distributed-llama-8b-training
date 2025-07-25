🦙 Distributed LLaMA Training with PEFT, Tensor, and Pipeline Parallelism



Developed for the Big Data and Machine Learning Systems course at New York University, this project explores large-scale LLaMA language model training using three distributed strategies. All models were trained using NYU's High Performance Computing (HPC) infrastructure.

📚 Overview
This project includes three implementations of distributed training for LLaMA models using PyTorch:

Distributed Data Parallel (DDP) + PEFT with LoRA

Tensor Parallelism

Pipeline Parallelism

Each approach demonstrates a different strategy for scaling large language model training while optimizing memory usage and performance.

📂 Project Structure
bash
Copy
Edit
.
├── train_ddp_peft.py          # DDP + PEFT training script
├── train_llama_tp.py          # Tensor Parallelism training script
├── train_llama_pp.py          # Pipeline Parallelism training script
├── data/                      # Input text data
└── README.md                  # This file
🔧 Implementation Highlights
1️⃣ DDP + Parameter-Efficient Fine-Tuning (LoRA)
Fine-tunes LLaMA 3.2B using PyTorch DDP and LoRA for efficient memory and compute usage.

🧠 Model: LLaMA 3.2B + LoRA adapters

⚙️ Strategy: DistributedDataParallel

💾 Memory Optimization:

8-bit quantization (via bitsandbytes)

Gradient accumulation (4 steps)

Mixed-precision (FP16)

📈 Training Techniques:

Cosine LR scheduling

Early stopping

Gradient clipping

📊 Metrics Tracked:

Training/validation loss

Perplexity

Token throughput

Learning rate

Checkpointing best model

2️⃣ Tensor Parallelism
Splits LLaMA model weights column-wise across GPUs using PyTorch’s distributed tensor parallel capabilities.

🧠 Parallelism: Column-wise tensor parallelism across 2 GPUs

⚙️ Optimization:

8-bit quantization

Gradient synchronization

Mixed-precision training (FP16)

🧩 Custom Data Loader: Processes multiple text files

🔧 Training Setup:

AdamW optimizer

NCCL backend

📝 Logging: Progress & metrics tracked during training

3️⃣ Pipeline Parallelism
Implements stage-wise pipeline parallelism using torch.distributed.pipeline.sync.Pipe.

🔗 Model Partitioning: LLaMA layers split across pipeline stages (GPUs)

⚙️ Memory Optimization:

Gradient checkpointing

FP16 precision

📉 Training Stability:

Cosine LR schedule

Early stopping

Gradient clipping

🧪 Custom CLI Options:

--batch_size, --epochs, --learning_rate, --patience, --max_grad_norm, etc.

📊 Monitoring: Full metric tracking, validation splitting, and automatic checkpointing

🧠 Key Learnings
Implemented scalable distributed training architectures using DDP, Tensor, and Pipeline parallelism

Optimized memory and compute using quantization and PEFT (LoRA)

Leveraged NYU’s HPC resources for real-world LLM training at scale

Developed robust logging, metric tracking, and model management tools

📍 Acknowledgements
This project was completed as part of the Big Data and Machine Learning Systems course at New York University.
All training was conducted on the NYU High Performance Computing (HPC) cluster.
