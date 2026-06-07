# Fast & Efficient LLM Inference with vLLM 

This repository tracks my hands-on learning journey through post-training model quantization and optimized inference serving. It documents how to compress large language models and deploy them efficiently for production-grade throughput and minimal latency.

##  Tech Stack & Ecosystem

*   **Runtime Backend:** [vLLM](https://github.com) (High-throughput, memory-efficient LLM serving engine)
*   **Compression Engine:** [LLM Compressor](https://github.com) (Production quantization toolkit from the vLLM project)
*   **Environment Manager:** [uv](https://github.com) (Blazingly fast Python package and environment manager written in Rust)
*   **Base Architecture:** Qwen 0.6B Causal LM

---

##  Repository Structure

```text
learning-vllm/
├── recipes/                  # Quantization configuration files
│   └── gptq_w4a16.yaml       # GPTQ settings (W4A16 precision)
├── notebooks/                # Step-by-step interactive learning flow
│   ├── 01_llm_compressor.ipynb   # Post-training quantization via GPTQ
│   ├── 02_basic_inference.ipynb  # Running base vs. quantized models offline
│   ├── 03_sampling_params.ipynb  # Tweaking generation parameters (temp, top_p)
│   └── 04_async_llm_engine.ipynb # Deep dive into vLLM's asynchronous engine
├── scripts/                  # Executable pipeline automation
│   ├── download_model.py     # High-speed parallel weight downloader
│   ├── simple_server.py      # Launching an OpenAI-compatible API server
│   └── benchmark_speed.py    # Throughput and latency evaluation
├── models/                   # Local model weights folder (*Git Ignored*)
├── data/                     # Evaluation datasets and prompt templates
└── requirements.txt          # Python ecosystem configurations
```

---

##  Quick Start Guide

### 1. Prerequisites
Ensure you have the Rust-backed `uv` package manager installed:
```bash
curl -LsSf https://astral.sh | sh
source \$HOME/.local/bin/env
```

### 2. Environment Setup & Synchronization
Sync the dependencies and automatically spin up the isolated virtual environment:
```bash
uv pip sync requirements.txt
```

### 3. Boost Download Bandwidth
Enable Hugging Face's native Rust-accelerated parallel downloading module:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### 4. Launch the Learning Workspace
Open the interactive notebooks directly through the managed virtual environment path:
```bash
uv run --with jupyterlab jupyter lab
```

---

##  Learning Roadmap & Core Concepts

### Post-Training Quantization (PTQ)
Using `llmcompressor`, we pass the model through a `oneshot` process alongside a calibration dataset (WikiText-2). This analyzes activation maps to shrink weight precisions from 16-bit (`BF16`) to 4-bit (`W4A16`) with minimal degradation to perplexity score. 

### PagedAttention & vLLM Serving
Once the compressed weights are exported, the model is pushed to vLLM. This bypasses structural bottlenecking by dynamically managing the Key-Value (KV) cache memory using virtual paging techniques, vastly expanding the concurrent batching threshold.

---

##  Acknowledgements

Special thanks to **[DeepLearning.AI](https://deeplearning.ai)** for providing the underlying structural concepts, optimization curriculum, and syllabus patterns that inspired this learning repository. 
