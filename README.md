# MACHINE TRANSLATION IN SPECIALIZED DOMAINS WITH QE-GUIDED ACTIVE LEARNING AND LLM TEACHERS

## Quick start

### Gemma 4 E2B full fine-tuning on 4 GPUs

```bash
# 1. Clone the repository
git clone https://github.com/contiloop/ADS.git
cd ADS

# 2. Install dependencies
# Requires Python 3.11 and an NVIDIA CUDA environment
make set

# 3. Authenticate with Hugging Face
hf auth login

# 4. Authenticate with Weights & Biases (optional)
wandb login

# 5. Download the prepared Gemma dataset
make download-prepared-data \
  HF_DATASET_REPO=alwaysgood/financial-english-source-corpus-gemma4-e2b-1280 \
  HF_DATASET_LOCAL_DIR=data/prepared/financial-english-source-corpus-gemma4-e2b-1280

# 6. Configure the teacher API
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# 7. Run full fine-tuning across all subsets on 4 GPUs
make train-stage \
  TRAIN_CONFIG=configs/config.yaml \
  SFT_NPROC_PER_NODE=4 \
  TRAIN_OVERRIDES='model=gemma4_e2b_it training=full inference.num_gpus=4 qe.selection.num_gpus=4' \
  EVAL_OVERRIDES='eval.generation.num_gpus=4'
```

### Qwen3.5-4B full fine-tuning on 4 GPUs

After the environment setup and authentication above, download the Qwen dataset
and select the Qwen model explicitly:

```bash
make download-prepared-data \
  HF_DATASET_REPO=alwaysgood/financial-english-source-corpus-qwen35-1280 \
  HF_DATASET_LOCAL_DIR=data/prepared/financial-english-source-corpus-qwen35-1280 \
  HF_DATASET_REVISION=main

make train-stage \
  SFT_NPROC_PER_NODE=4 \
  TRAIN_OVERRIDES='model=qwen35_4b_it training=full inference.num_gpus=4 qe.selection.num_gpus=4' \
  EVAL_OVERRIDES='eval.generation.num_gpus=4'
```

Without overrides, `make train-stage` uses Gemma 4 E2B full fine-tuning with
4 GPUs for SFT, student inference, QE selection, and evaluation generation.
Explicit `TRAIN_OVERRIDES`, `EVAL_OVERRIDES`, and `SFT_NPROC_PER_NODE` take
precedence over these defaults.

### Runtime version notes

- Python: `3.11`
- CUDA wheel index: `https://download.pytorch.org/whl/cu128`
- Torch stack: `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0`
- vLLM: `vllm==0.19.1`
- Unsloth stack: `unsloth==2026.7.2`, `unsloth-zoo==2026.7.2`
- HF training stack: `transformers==5.5.0`, `trl==0.24.0`, `datasets==3.4.1`
- Hugging Face transfer stack: `huggingface_hub>=1.14.0,<2`, `hf-xet>=1.5.0,<2`
- FlashAttention2: `flash-attn==2.8.3`
- NumPy: `numpy==2.2.6`

`make set` also prepares isolated COMET and MetricX environments for
evaluation. MetricX uses `pyarrow==20.0.0`, `protobuf==3.20.3`, and
`fsspec==2023.6.0`, plus `numpy==1.26.4`, in its isolated environment for
compatibility with its pinned old metric stack.
