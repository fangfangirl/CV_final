# Happy Whale and Dolphin Identification

This repository contains our final project for a computer vision course. The task is based on the Kaggle **Happy Whale and Dolphin Identification** competition, where the goal is to identify individual whales and dolphins from images.

The project uses several deep learning models to extract image embeddings, builds similarity-based prediction matrices, and combines multiple models and crop modes to generate the final submission.

> A Traditional Chinese draft is kept in `README.zh-TW.md` for reference.

## Introduction

The main idea of this project is to treat individual identification as a metric-learning and retrieval problem. Each single model is trained with an ArcFace-style classification head, then used to extract embeddings and logits for both training and test images. During inference, we combine:

- KNN similarity scores
- Logit-based class scores
- Class prototype scores
- Species-aware score adjustment
- Weighted crop-mode ensembles
- Final multi-model ensemble

The experiments were mainly run on **Google Colab A100**. Since Colab runtimes can disconnect after long sessions, the notebooks are designed to save and resume checkpoints through Hugging Face.

## Project Structure

| File | Description |
| --- | --- |
| `cv_V2_model.ipynb` | EfficientNetV2-XL single-model notebook for training, embedding extraction, and crop inference. |
| `cv_V2_model_7.ipynb` | EfficientNetV2-L single-model notebook. It also contains multi-model concat related logic. |
| `cv_B7_model.ipynb` | EfficientNet-B7 single-model notebook using the B7 training configuration. |
| `cv_B6_model_charm_hf_ipynb.ipynb` | EfficientNet-B6 single-model notebook with full-body/backfin/charm crop settings. |
| `cv_final_ensemble_hf.ipynb` | Final ensemble notebook. It downloads model outputs from Hugging Face, builds KNN/prototype matrices, and generates submissions. |
| `scripts/clean_notebooks_for_github.py` | Utility script for cleaning notebook outputs and removing accidental token strings before pushing to GitHub. |
| `README.zh-TW.md` | Preserved Traditional Chinese README draft. |

## Method Summary

- Backbones:
  - `tf_efficientnetv2_xl.in21k`
  - `tf_efficientnetv2_l.in21k`
  - `tf_efficientnet_b7_ns`
  - `tf_efficientnet_b6_ns`
- Loss / head:
  - ArcFace-style margin head
  - Sub-center ArcFace
  - Adaptive margin
  - Optional focal loss
- Augmentation:
  - Albumentations
- Crop modes:
  - Original image
  - Full-body bounding box crop
  - Backfin crop
  - Charm / additional crop variants for B6
- Inference:
  - Embedding similarity
  - KNN voting
  - Logit score fusion
  - Class prototype scoring
  - Species bonus / penalty
  - Weighted crop ensemble
- Storage:
  - Checkpoints, embeddings, matrices, and submissions are uploaded to Hugging Face instead of GitHub.

## Dataset

The project uses the following Kaggle / KaggleHub resources:

- `happy-whale-and-dolphin`
- `jpbremer/fullbodywhaleannotations`
- `genlaxai/happy-whale-backfin-cropped`
- `changchoufang/fullybody-charm`

These files are large and are **not included** in this repository. The notebooks can download or locate the datasets through KaggleHub when running in Colab or Kaggle Notebook.

For local execution, prepare the data in a structure similar to:

```bash
CV_final/
  data/
    happy-whale-and-dolphin/
      train.csv
      sample_submission.csv
      train_images/
      test_images/
    fullbodywhaleannotations/
    happy-whale-backfin-cropped/
    fullybody-charm/
  checkpoints/
  embeddings/
  mats/
  submissions/
```

Then update the path candidates in the `CFG` class of each notebook, or set the corresponding paths manually before running the data preparation cells.

## Environment Setup

### Colab A100 Setup

This project was mainly developed on **Google Colab A100**.

1. Open the target notebook in Colab.
2. Enable GPU runtime.
3. Use A100 if available.
4. Run the setup cells that install:

```bash
pip install -q kagglehub timm albumentations huggingface_hub
```

5. Set the following secrets or environment variables:

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
os.environ["HF_REPO_ID"] = "your_name/your_hf_repo"
```

For Colab, `HF_TOKEN` can also be stored in Colab Secrets. For Kaggle access, use `kagglehub.login()` or Kaggle credentials.

### Local Setup

If you want to run the project locally, a CUDA GPU with large VRAM is strongly recommended. Full training of the large EfficientNet models is not practical on CPU.

1. Clone the repository:

```bash
git clone <your-github-repo-url>
cd CV_final
```

2. Create a Python environment:

```bash
conda create -n happywhale-cv python=3.11 -y
conda activate happywhale-cv
```

3. Install PyTorch according to your CUDA version:

Visit the official PyTorch installation page and choose the correct command:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

4. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

5. Authenticate Kaggle / KaggleHub:

```bash
kaggle competitions download -c happy-whale-and-dolphin
```

or use `kagglehub.login()` inside the notebooks.

6. Set Hugging Face environment variables:

```bash
set HF_TOKEN=your_huggingface_token
set HF_REPO_ID=your_name/your_hf_repo
```

On PowerShell:

```powershell
$env:HF_TOKEN="your_huggingface_token"
$env:HF_REPO_ID="your_name/your_hf_repo"
```

7. Open Jupyter:

```bash
jupyter notebook
```

Then run the notebooks in the same order described below.

## Usage

### 1. Single-Model Training

Use one of the single-model notebooks:

- `cv_V2_model.ipynb`
- `cv_V2_model_7.ipynb`
- `cv_B7_model.ipynb`
- `cv_B6_model_charm_hf_ipynb.ipynb`

Before training, check the `CFG` class:

```python
CFG.do_train = True
CFG.resume = True
CFG.hf_upload = True
CFG.hf_repo_id = "your_name/your_hf_repo"
```

During training, checkpoints are saved locally and uploaded to Hugging Face. The most important checkpoint path is:

```python
CFG.hf_checkpoint_file = "checkpoints/last_checkpoint.pth"
```

### 2. Colab 24-Hour Disconnect and Resume

Colab A100 sessions may disconnect after long training runs. To continue training after a disconnect:

1. Reopen the notebook in Colab.
2. Re-enable A100 GPU.
3. Re-run setup, imports, KaggleHub, and config cells.
4. Make sure the same Hugging Face repo is used:

```python
CFG.hf_repo_id = "your_name/your_hf_repo"
CFG.resume = True
CFG.resume_from_hf = True  # if the notebook provides this option
CFG.hf_checkpoint_file = "checkpoints/last_checkpoint.pth"
```

5. Run the resume / checkpoint loading cell.
6. Continue the training cells.

Because `hf_upload_each_epoch` is enabled in the training notebooks, the latest checkpoint should be available on Hugging Face after each epoch. If the runtime disconnects, download or resume from that checkpoint instead of restarting from epoch 0.

### 3. Embedding Extraction

After training a model, extract train/test embeddings:

```python
CFG.do_train = False
CFG.do_embedding = True
```

The output files are saved as `.npz` files and uploaded to the configured Hugging Face repository under the embedding directory, for example:

```python
CFG.hf_embedding_dir = "embeddings"
```

or, for the B7 setting:

```python
CFG.hf_embedding_dir = "embeddings_31"
```

### 4. Crop Inference

To build prediction matrices and crop-mode submissions:

```python
CFG.run_mode = "crop_inference"
CFG.do_crop_inference = True
```

The notebooks combine several crop modes such as:

```python
CFG.crop_infer_modes = ["fullbody", "backfin", "none"]
```

Weighted crop submissions are generated according to `CFG.crop_weight_sets`.

### 5. Final Ensemble

Run:

```text
cv_final_ensemble_hf.ipynb
```

This notebook downloads model outputs from Hugging Face, restores prediction matrices, applies weighted model/crop fusion, and creates final submission files.

## GitHub Upload Notes

Before pushing to GitHub, clean notebook outputs:

```bash
python scripts/clean_notebooks_for_github.py
```

Then check:

```bash
git status
git diff --stat
```

Do not commit:

- Kaggle datasets
- Hugging Face tokens
- checkpoints
- embeddings
- `.npy` / `.npz` prediction matrices
- submissions
- Colab cache files

These files are excluded by `.gitignore`.

## Important Security Note

Never hard-code Kaggle or Hugging Face tokens in notebooks. Use environment variables, Colab Secrets, Kaggle Secrets, or `getpass()`.

If a token was ever saved inside a notebook, revoke it from the Kaggle / Hugging Face website and create a new one before running future experiments.

## Repository Initialization

If this folder is not yet a Git repository:

```bash
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
