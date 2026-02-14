# paligemma-from-scratch

## Installation
```bash
uv sync
```

## Quick Start
1. Model Download
```bash
uv run hf download google/paligemma-3b-pt-224 \
    --local-dir ./models/paligemma/paligemma-3b-pt-224
```

2. Inference
```bash
uv run python inference.py \
    --model_path "./models/paligemma/paligemma-3b-pt-224" \
    --prompt "this cat is" \
    --image_file_path "images/example1.jpeg" \
    --max_tokens_to_generate 100 \
    --top_p 0.9 \
    --do_sample True \
    --only_cpu False
```
