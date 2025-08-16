# GRPO Fine-Tuning with SmolVLM2 on Path‑VQA

This notebook walks through a compact GRPO (Group Relative Preference Optimization) fine‑tuning setup for the multimodal SmolVLM2‑2.2B‑Instruct model, using a small streamed subset of the Path‑VQA dataset. It includes data preparation, LLM‑graded rewards, LoRA adapters, vLLM‑assisted generation, a Rich console callback, and basic GPU diagnostics.

If you’re running this in Colab or locally, execute the cells in order from top to bottom.

## Quick start
1. Install dependencies (first two sections).
2. Load the dataset and generate `all_grpo_samples`.
3. Load the model + LoRA and set hyperparameters.
4. Configure `GRPOTrainer` and run `trainer.train()`.

## Prerequisites
- Python 3.10+ (notebook metadata uses 3.11)
- A GPU (e.g., NVIDIA T4/A10/RTX) with recent CUDA
- Sufficient VRAM for 2.2B model + generation (use gradient checkpointing; see tips below)

## Section‑by‑section walkthrough

### 1) Install dependencies
Installs TRL, Transformers, Datasets, PEFT, vLLM, and supporting libs (wandb, openai client, hf_xet). These provide:
- Training loop and GRPO utilities (trl)
- Model and processor loading (transformers)
- Dataset streaming (datasets)
- Parameter‑efficient fine‑tuning (peft/LoRA)
- Fast generation backend (vLLM; optional)

### 2) Install Rich
Adds Rich for nicer, panel‑style training logs and live generation previews.

### 3) Load dataset (Path‑VQA subset)
- Streams the training split from `flaviagiammarino/path-vqa`.
- Takes the first 1,000 examples for quick iteration (`ds_subset = ds_subset.take(1000)`).
Streaming avoids downloading the full dataset, saving time and disk space.

### 4) Data preprocessing and sample generation
Defines helper functions and builds GRPO‑ready samples:

- `pil_to_data_uri(image, format='PNG')`:
  - Utility to convert PIL images to data URIs. Kept for reference; not used directly in the pipeline.

- `get_grpo_sample(example)`:
  - Validates and normalizes the image (ensures RGB; resizes to 128×128 to reduce VRAM).
  - Appends a strict output format to the question:
    - The model must reply using:
      - <think>...</think> for reasoning
      - <answer>...</answer> for the final answer
  - Builds a chat‑style prompt with roles and content that includes the resized image and the text question.
  - Returns a dict with:
    - `prompt` (system + user messages with image)
    - `image` (resized PIL Image)
    - `solution` (ground‑truth answer)

- `dataset_gen_grpo(dataset_source)`:
  - A generator that yields GRPO samples by applying `get_grpo_sample` across the streamed dataset.
  - Skips malformed items and logs errors without stopping the run.
  - The notebook then collects all samples into `all_grpo_samples`.

Tip: The small 128×128 resize is intentional to reduce GPU memory during training and generation.

### 5) Inspect first generated sample
A quick `print(all_grpo_samples[0])` sanity check to confirm structure and content.

### 6) Scoring and reward function using OpenRouter
Implements an LLM‑graded reward:

- `call_model(...)`: Calls OpenRouter’s chat completions API via the OpenAI‑compatible client.
- `judge_answer(completion, solution)`: Builds a grading prompt and requests a single float score based on:
  - Answer correctness and adherence to the required output format (<think> + <answer>).
  - Extracts a numeric score using a regex; defaults to 0.0 if parsing fails.
- `reward_function(completions, solution, ...)`: Vectorizes `judge_answer` across a batch.

Security note:
- Do NOT hardcode API keys. Use environment variables instead.
- Example:
  ```bash
  export OPENROUTER_API_KEY="your_key_here"
  ```
  In Python:
  ```python
  import os
  client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=os.getenv("OPENROUTER_API_KEY"),
  )
  ```

### 7) Load SmolVLM2 model and apply LoRA; training hyperparameters
- Loads `HuggingFaceTB/SmolVLM2-2.2B-Instruct` with `trust_remote_code=True` and `device_map="auto"`.
- Configures tokenizer padding to left (helps multimodal generation in some setups).
- Applies LoRA to attention projections (`q_proj`, `k_proj`, `v_proj`) with r=8, alpha=32, dropout=0.1.
- Sets core hyperparameters:
  - `OUTPUT_DIR = "./grpo_smolvlm_finetuned_bf16"`
  - `LEARNING_RATE = 1e-5`
  - `NUM_TRAIN_EPOCHS = 1`

LoRA reduces the number of trainable parameters and VRAM needs, while preserving base model weights.

### 8) GPU memory cleanup utilities
- `free_memory()` empties CUDA cache and runs garbage collection.
- Handy to insert between large steps if you encounter OOM issues.

### 9) Placeholder (previously commented‑out training block)
A commented configuration showing an earlier training approach. The actual training uses the newer block below with richer logging and vLLM options.

### 10) Rich callback to render live completions and metrics
- `RichCompletionCallback` listens to trainer logs:
  - Prints generated completions in tidy panels (truncates long outputs).
  - Shows per‑sample rewards if available.
  - Displays training metrics (losses, etc.).
  - Rate‑limits updates to avoid flooding the output.

This makes it easier to visually track model behavior and reward trends during training.

### 11) Configure GRPO Trainer and run training
- Uses `GRPOConfig` with:
  - `per_device_train_batch_size=4`
  - `gradient_checkpointing=True` (memory saver)
  - `num_generations=4`, `log_completions=True` (for reward shaping and visibility)
  - `max_prompt_length=1400`
  - vLLM enabled for faster generation: `use_vllm=True`, `vllm_mode="colocate"`
  - `bf16=False` in the provided config (toggle if your GPU supports bf16 for speed/memory benefits)
- Initializes `GRPOTrainer` with:
  - The model, processor, `reward_function`, training args, `all_grpo_samples`
  - The `RichCompletionCallback`
- Starts training with `trainer.train()`.

### 12) GPU diagnostics
Prints:
- CUDA availability and version
- GPU name
- Compute capability
Useful for confirming the environment is ready for training/generation.

## Defaults and rationale
- Dataset subset: 1,000 streamed items for speed.
- Image resize: 128×128 RGB to reduce memory and bandwidth (good for quick tests).
- LoRA on attention projections: a common, effective target set.
- Generation backend: vLLM for faster sampling during GRPO.
- Strict output format (<think> + <answer>): simplifies automated grading.

## Tips and troubleshooting
- OOM/memory pressure:
  - Reduce `per_device_train_batch_size`.
  - Keep image size at 128×128 or smaller.
  - Ensure `gradient_checkpointing=True`.
  - Consider enabling bf16 if your GPU supports it.
  - Run `free_memory()` between steps.
- OpenRouter errors (e.g., 401/403):
  - Ensure `OPENROUTER_API_KEY` is set and the key has access to the chosen model.
- Slow generation:
  - vLLM is already enabled; still, reduce `num_generations`, shorten prompts, or use a smaller grader model.
- Reproducibility:
  - Seeds aren’t set; add manual seeding if you need deterministic runs.

## Security and compliance
- Never commit API keys or secrets. Prefer environment variables or secret managers.
- Review the licenses and usage policies for:
  - Path‑VQA dataset
  - SmolVLM2‑2.2B‑Instruct model
  - Any third‑party services (OpenRouter)

## Acknowledgments
- TRL (GRPO), Hugging Face Transformers/PEFT
- vLLM for high‑throughput generation
- Path‑VQA dataset authors