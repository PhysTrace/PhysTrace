# PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="arXiv">
  </a>
</p>

Official codebase for the paper **"PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models"**.

<div>
<!-- TODO: replace with the teaser figure -->
<img width="100%" alt="PhysTrace Overview" src="https://github.com/user-attachments/assets/0f78430b-34e6-497a-bcdf-a1ed5f450023" />
</div>

## Overview

This repository provides the evaluation code for **PhysTrace**, which localizes *when* a video breaks physical consistency instead of returning a single video-level score.

- **Sliding-window scoring**: each video is split into overlapping windows, scored by a video-text alignment model (PE-AV-Large by default) and a reasoning MLLM (Qwen3-VL-8B-Thinking by default).
- **IntPhys2-Timing**: temporal violation annotations (onset / end) on top of IntPhys2, covering the four physical attributes `solidity`, `permanence`, `immutability`, and `continuity`.

## 1. Repository Structure

```text
project_root/
├── AlignmentModels/      # video-text alignment / MLLM code (submodules) and weights
├── DiffusionModels/      # video generative model code (submodules) and weights
├── IntPhys2Timing/       # violation GT CSVs and per-attribute prompts
├── PhysTrace/            # core library (windowing, CLIP / MLLM scorers)
├── demo/                 # interactive web demo
├── score/                # batch inference and scoring
├── requirements.txt
└── README.md
```

### Main entry points

- `demo/app.py`: interactive demo (browser UI) for inspecting per-window results
- `score/run_test.py`: batch inference over every IntPhys2 pair
- `score/run_eval.py`: scoring of the inference results against the IntPhys2-Timing GT

## 2. Installation

### a) Environment

Tested on **NVIDIA A100** with `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`.

```bash
pip3 install -r requirements.txt
```

### b) Code Download

```bash
git clone --recursive <REPO_URL>
# or, after a plain clone
git submodule update --init --remote --recursive
```

Or clone the external repositories manually:

```bash
# Alignment Scoring Models
git clone https://github.com/facebookresearch/perception_models.git AlignmentModels/PE_AV_large

# Diffusion models (only for Section 4: Evaluation for Video Generative Models)
git clone https://github.com/Wan-Video/Wan2.1.git DiffusionModels/Wan2.1
git clone https://github.com/nvidia-cosmos/cosmos-predict2.git DiffusionModels/cosmos-predict2
```

### c) Weight Download

```bash
hf auth login
```

**Alignment Scoring Models**

```bash
hf download facebook/pe-av-large --revision perception_models --local-dir AlignmentModels/weights/PE-AV-Large
```

**MLLMs**

```bash
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir AlignmentModels/weights/Qwen3-VL-8B-Thinking

# optional
hf download Qwen/Qwen3-VL-32B-Thinking --local-dir AlignmentModels/weights/Qwen3-VL-32B-Thinking
hf download nvidia/Cosmos-Reason2-8B --local-dir AlignmentModels/weights/Cosmos-Reason2-8B
```

**Diffusion models** (only for [4. Evaluation for Video Generative Models](#4-evaluation-for-video-generative-models))

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
    --local-dir DiffusionModels/weights/wan/Wan2.1-T2V-1.3B

hf download nvidia/Cosmos-Predict2-2B-Video2World --revision f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
    --local-dir DiffusionModels/weights/cosmos/nvidia/Cosmos-Predict2-2B-Video2World \
    --include config.json model_index.json model-480p-16fps.pt "scheduler/*" "text_encoder/*" "tokenizer/*" "vae/*"

# Cosmos-Predict2 loads T5 with from_pretrained(<dir>), so the tokenizer must sit beside the text encoder weights
M=DiffusionModels/weights/cosmos/nvidia/Cosmos-Predict2-2B-Video2World
cp $M/tokenizer/{special_tokens_map.json,spiece.model,tokenizer.json,tokenizer_config.json} $M/text_encoder/
```

**Alignment Scoring Models (optional)**

```bash
# ---- internvideo2-stage2-6b ----
# internvideo2-stage2
hf download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
    --local-dir AlignmentModels/weights/InternVideo2-Stage2

# bert tokenizer
hf download google-bert/bert-large-uncased vocab.txt tokenizer.json tokenizer_config.json config.json \
    --local-dir AlignmentModels/weights/InternVideo2-Stage2/bert-large-uncased

# ---- internvideo2-clip-6b ----
# internvideo2-stage2 (same as above; skip if already downloaded)
hf download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
    --local-dir AlignmentModels/weights/InternVideo2-Stage2

# InternVL-C 13B (25GB)
hf download OpenGVLab/InternVL internvl_c_13b_224px.pth \
    --local-dir AlignmentModels/weights/InternVideo2-CLIP

# CLIP delta
hf download OpenGVLab/InternVideo2-CLIP-6B-224p-f8 6B_clip.pth \
    --local-dir AlignmentModels/weights/InternVideo2-CLIP

# LLaMA tokenizer
hf download hfl/chinese-alpaca-lora-7b tokenizer.model tokenizer_config.json special_tokens_map.json \
    --local-dir AlignmentModels/weights/InternVideo2-CLIP/tokenizer
```

### d) Data Download: IntPhys2 (for IntPhys2-Timing)

```bash
python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='facebook/IntPhys2', repo_type='dataset', local_dir='/Path/To/IntPhys2')"
```

## 3. Evaluation with IntPhys2-Timing

### a) Demo

```bash
CUDA_VISIBLE_DEVICES=0 python3 demo/app.py \
    --intphys2-dir /Path/To/IntPhys2 \
    --result-dir results/demo \
    --port 1535
```

- `--intphys2-dir`: IntPhys2 root (uses `Main/Videos` and `Main/metadata.csv`; the CLIP cache `Main/videos-npz` is created beside them)
- `--result-dir`: where `results.csv` / `trials.csv` are written
- `--gt-csv`: violation GT drawn on the UI chart (defaults to `IntPhys2Timing/intphys2_gt.csv`)
- `--mllm-backend`: `qwen3-vl-thinking` (default), `qwen3-vl-32b-thinking`, `cosmos-reason2`
- `--clip-model`: `pe-av-large` (default), `internvideo2-stage2-6b`, `internvideo2-clip-6b`

### b) Inference

```bash
CUDA_VISIBLE_DEVICES=0 python3 score/run_test.py \
    --intphys2-dir /Path/To/IntPhys2 \
    --result-dir results/test \
    --window-seconds 2 --hop-seconds 1 \
    --clip-frames 60 --mllm-frames 60 \
    --categories solidity=BoxSoldity
```

- `--intphys2-dir`, `--result-dir`: same as the demo (same result format, so the result directory can be opened in the demo)
- `--window-seconds`, `--hop-seconds`: window length and hop between windows, in seconds
- `--clip-frames`, `--mllm-frames`: frames fed to CLIP / the MLLM per window
- `--categories`: limit the categories run per attribute, e.g. `solidity=BoxSoldity` (default: every category)
- `--mllm-backend`: `qwen3-vl-thinking` (default), `qwen3-vl-32b-thinking`, `cosmos-reason2`
- `--clip-model`: `pe-av-large` (default), `internvideo2-stage2-6b`, `internvideo2-clip-6b`

### c) Scoring

```bash
python3 score/run_eval.py --result-dir results/test
```

- `--result-dir`: result directory of the inference above
- `--strict-edges`: a band that only touches a GT edge is not counted as a hit (optional)
- `--out`: output file stem (default: `<result-dir>/eval_gt`)

> **Note**  
> `score/run_test.py` is resumable. Re-running the same command skips pairs that are already stored; use `--status` to print progress and `--force` to re-analyze.

## 4. Evaluation for Video Generative Models

Each IntPhys2 video is inverted (clean → noise) by a video generative model, and the x0 estimate decoded at each sigma is evaluated with the same pipeline as Section 3. Wan2.1 and Cosmos-Predict2 are run separately, so give each model its own output directory.

### a) Video Generation

**Wan2.1-T2V-1.3B**

```bash
CUDA_VISIBLE_DEVICES=0 python3 PhysTrace/prob_wan.py \
    -i /Path/To/IntPhys2 \
    -o results/wan \
    --sigmas 0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --stride 8
```

**Cosmos-Predict2-2B**

```bash
CUDA_VISIBLE_DEVICES=0 python3 PhysTrace/prob_cosmos.py \
    -i /Path/To/IntPhys2 \
    -o results/cosmos \
    --sigmas 0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --frames 93
```

- `-i`: IntPhys2 root (same as `--intphys2-dir` above), a folder of `.mp4` files, or a single video
- `-o`: output directory; one `sigma_<value>/Main/Videos` folder (with `metadata.csv`) is written per sigma, plus `sigma_0.00_INPUT` for the input frames
- `--sigmas`: comma-separated sigmas (= flow time) at which the x0 estimate is decoded
- `--stride` (Wan): use every N-th source frame (`8` = 7.5 fps from 60 fps)
- `--frames` (Cosmos): keep exactly N frames spread evenly over the whole clip (`93` = one model window, no padding)
- `--gt-csv`: generate only the pairs listed in this GT CSV (defaults to `IntPhys2Timing/intphys2_gt.csv`)

### b) Inference

Pass one sigma folder of the generation output as `--intphys2-dir`.

**Wan2.1-T2V-1.3B**

```bash
CUDA_VISIBLE_DEVICES=0 python3 score/run_test.py \
    --intphys2-dir results/wan/sigma_0.50 \
    --result-dir results/wan_test/sigma_0.50 \
    --window-seconds 2 --hop-seconds 1 \
    --clip-frames 16 --mllm-frames 16 \
    --categories solidity=BoxSoldity
```

**Cosmos-Predict2-2B**

```bash
CUDA_VISIBLE_DEVICES=0 python3 score/run_test.py \
    --intphys2-dir results/cosmos/sigma_0.50 \
    --result-dir results/cosmos_test/sigma_0.50 \
    --window-seconds 2 --hop-seconds 1 \
    --clip-frames 18 --mllm-frames 18 \
    --categories solidity=BoxSoldity
```

- `--intphys2-dir`: a `sigma_<value>` folder written by the generation step
- `--clip-frames`, `--mllm-frames`: lower than in Section 3, since the generated videos have a lower frame rate (Wan: 7.5 fps, Cosmos: 93 frames per clip)
- The other options are the same as [3. b) Inference](#b-inference)

### c) Scoring

```bash
python3 score/run_eval.py --result-dir results/wan_test/sigma_0.50
python3 score/run_eval.py --result-dir results/cosmos_test/sigma_0.50
```

- `--result-dir`: result directory of the inference above (one per model and sigma)

> **Note**  
> To compare sigmas, repeat b) and c) for each `sigma_<value>` folder, including `sigma_0.00_INPUT` as the reference.

## Citation

If you find this repository useful, please cite the paper.

```bibtex
@article{yoo2026phystrace,
    title={{PhysTrace}: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models},
    author={Yoo, Taekoan and Jung, Wonkyung and Kim, Kyunghun and Kong, Kyeongbo},
    journal={arXiv preprint arXiv:XXXX.XXXXX},
    year={2026},
    url={TODO}
}
```
