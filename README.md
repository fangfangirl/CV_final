# Happy Whale and Dolphin Identification

[![Project report](https://img.shields.io/badge/Project-report-red)](report/CV_Report.pdf)
[![Project slide](https://img.shields.io/badge/Project-slide-yellow)](report/CV_Slide.pdf)

This repository contains our final project for a computer vision course. The project is based on the Kaggle **Happy Whale and Dolphin Identification** competition, where the goal is to identify individual whales and dolphins from images.

We mainly treat the task as an image retrieval / metric-learning problem. Each model is trained to produce discriminative embeddings, then inference combines KNN similarity, class prototypes, logits, species information, crop-mode ensembles, and final multi-model fusion.

The experiments were run primarily on **Google Colab A100**. Because Colab sessions may disconnect after long training runs, the notebooks save checkpoints, embeddings, matrices, and submissions to Hugging Face so training and inference can be resumed.

> The Traditional Chinese is preserved as [README.zh-TW.md](README.zh-TW.md).

## Course / Team

- Course: NYCU Computer Vision 2026
- Team members:
  - 張周芳, 314511037
  - 劉霈琳, 313554063
  - 何柏翰, 112550028

## Project Structure

| File | Description |
| --- | --- |
| [cv_v2xl_model_hf_ipynb.ipynb](cv_v2xl_model_hf_ipynb.ipynb) | EfficientNetV2-XL notebook for training, embedding extraction, and crop inference. |
| [cv_B7_model_hf_ipynb.ipynb](cv_B7_model_hf_ipynb.ipynb) | EfficientNet-B7 notebook using the B7 training configuration. |
| [cv_B6_model_charm_hf_ipynb.ipynb](cv_B6_model_charm_hf_ipynb.ipynb) | EfficientNet-B6 notebook with full-body, backfin, and charm crop settings. |
| [cv_convnext_model_hf.ipynb](cv_convnext_model_hf.ipynb) | ConvNeXt notebook for another single-model branch. |
| [cv_final_ensemble_hf.ipynb](cv_final_ensemble_hf.ipynb) | Final ensemble notebook. It downloads model outputs from Hugging Face, builds / loads matrices, fuses predictions, and generates submissions. |
| [requirements.txt](requirements.txt) | Python dependencies for local execution and Jupyter kernels. |
| [data/README.md](data/README.md) | Documents the expected local dataset directory layout. |

## Method Summary

- Backbones:
  - EfficientNetV2-XL
  - EfficientNet-B7
  - EfficientNet-B6
  - ConvNeXt
- Training:
  - ArcFace-style classification head
  - Sub-center ArcFace
  - Adaptive margin
  - Optional focal loss
  - Full-body / backfin crop augmentation
- Inference:
  - Embedding similarity
  - KNN score matrix
  - Logit score matrix
  - Class prototype score matrix
  - Species bonus / penalty
  - Weighted crop ensemble
  - Final multi-model ensemble

## Dataset

The notebooks use Kaggle / KaggleHub datasets:

- `happy-whale-and-dolphin`: https://www.kaggle.com/competitions/happy-whale-and-dolphin
- `jpbremer/fullbodywhaleannotations`: https://www.kaggle.com/datasets/jpbremer/fullbodywhaleannotations
- `genlaxai/happy-whale-backfin-cropped`: https://www.kaggle.com/datasets/genlaxai/happy-whale-backfin-cropped
- `changchoufang/fullybody-charm`: https://www.kaggle.com/datasets/changchoufang/fullybody-charm

The datasets are large and are **not included** in this repository. The notebooks can download or locate the data through KaggleHub when running in Colab.

For local execution, place datasets under `data/`. A template is provided in [data/README.md](data/README.md).

Prepare a structure similar to:

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

If your local paths are different, update the path candidates in the `CFG` class of each notebook before running training or inference.

Some notebooks include a helper named `_get_or_download_kaggle_paths()`. In Colab/Kaggle, it can reuse KaggleHub paths from previous cells. On a local machine, either place the data under the default `data/` layout, edit the fallback paths in this helper, or define the path variables before running the cell:

```python
happy_whale_and_dolphin_path = "data/happy-whale-and-dolphin"
jpbremer_fullbodywhaleannotations_path = "data/fullbodywhaleannotations"
backfin_path = "data/happy-whale-backfin-cropped"
fullbody_charm_path = "data/fullybody-charm"  # only needed by notebooks that use charm crops
```

## Pseudo Labels

This project can also use an optional [data/pseudo.csv](data/pseudo.csv) file for pseudo-label training. The file is provided under `data/`:

```bash
CV_final/
  data/
    pseudo.csv
```

We provide the pseudo-label CSV used by this project, but we do **not** provide the code for generating pseudo labels. If you want to create your own pseudo labels, please build your own generation pipeline and place the resulting CSV at `data/pseudo.csv` or set `CFG.pseudo_csv` / `PSEUDO_CSV` to your own file path.

To enable pseudo-label training, set:

```python
CFG.use_pseudo = True
CFG.pseudo_csv = "data/pseudo.csv"
```

The notebooks also support overriding the path through the `PSEUDO_CSV` environment variable:

```python
import os
os.environ["PSEUDO_CSV"] = "data/pseudo.csv"
```

Expected columns include:

- `image`
- `top1_pred`
- `top1_score`
- `top2_pred`
- `top2_score`
- `score_margin_top1_top2`

The notebooks filter pseudo labels by confidence threshold, margin threshold, known individual IDs, and maximum pseudo samples per ID.

## Hugging Face Artifacts

The trained model checkpoints and extracted embeddings are provided through Hugging Face because the files are too large to include in this code package.

Each single-model notebook uses a Hugging Face repository for:

- checkpoints, for example `checkpoints/last_checkpoint.pth`
- train/test embedding `.npz` files
- component matrices
- crop submission files

Current repository settings used by the notebooks:

| Model / Output | Hugging Face repository setting |
| --- | --- |
| EfficientNetV2-XL | `fangfang777/happywhale_v2xl_pseudo_768` |
| EfficientNet-B7 | `fangfang777/happywhale_b7_multicrop_fnb_subcenter_species_pseudo_1024_by_lin` |
| EfficientNet-B6 | `liu-peilin/happywhale_b6_pseudo_charm_round2_1024_v2` |
| ConvNeXt | `liu-peilin/conv` |
| Final ensemble output | `liu-peilin/happywhale_ensemble_v2xl_b6_b7_conv_v1` |

If the repositories are private, users need a Hugging Face token with read permission. Configure it before running notebooks:

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
```

For the final ensemble notebook, model repositories can be overridden with:

```python
os.environ["HF_REPO_ID_V2XL"] = "your_name/v2xl_repo"
os.environ["HF_REPO_ID_B6"] = "your_name/b6_repo"
os.environ["HF_REPO_ID_B7"] = "your_name/b7_repo"
os.environ["HF_REPO_ID_CONV"] = "liu-peilin/conv"
```

To skip training and use provided artifacts, run the setup/config cells, set the correct Hugging Face repo ids, then run the download / inference / ensemble cells.

### Artifact Folder Naming

Some Hugging Face repositories contain many experiment folders. Use the following rule when selecting artifacts:

- `checkpoints/last_checkpoint.pth` is the default checkpoint used for resume or inference.
- `embeddings` usually corresponds to the default / latest checkpoint path used by the notebook.
- `embeddings_25`, `embeddings_31`, or other numbered embedding folders correspond to a specific checkpoint epoch or experiment version.
- Crop submission folders may also include numbers or score-weight tags, for example `crop_submission_new_correct_single_502030_bonus0.1_31`.
- When a notebook sets `CFG.hf_embedding_dir = "embeddings_31"`, use the matching numbered embedding folder instead of the generic `embeddings` folder.

In short, if a model branch uses a dedicated epoch/version, its embedding folder name usually carries that number. If no number is specified, use the default artifacts generated from `last_checkpoint`.

## Setup Reference Links

- Google Colab: https://colab.research.google.com/
- PyTorch local installation selector: https://pytorch.org/get-started/locally/
- Conda installation guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- Kaggle API documentation: https://www.kaggle.com/docs/api
- KaggleHub package: https://github.com/Kaggle/kagglehub
- Hugging Face access tokens: https://huggingface.co/settings/tokens
- Hugging Face token permissions: https://huggingface.co/docs/hub/security-tokens

## Environment Setup

### Colab A100

This is the recommended environment for reproducing the experiments.

1. Open a notebook in Google Colab.
2. Select a GPU runtime and use A100 if available.
3. Run the install cell in the notebook, or install the main dependencies manually:

```bash
pip install -q kagglehub timm albumentations huggingface_hub
```

4. Configure Kaggle access with `kagglehub.login()` or Colab/Kaggle credentials. See the [Kaggle API documentation](https://www.kaggle.com/docs/api) and the [KaggleHub package](https://github.com/Kaggle/kagglehub).
5. Configure Hugging Face access through Colab Secrets or environment variables. Hugging Face access tokens can be created from the [User Access Tokens page](https://huggingface.co/settings/tokens).

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
os.environ["HF_REPO_ID"] = "your_name/your_hf_repo"
```

Do not hard-code real tokens in notebooks.

### Local Machine

Local execution is possible, but full training is only practical with a CUDA GPU with enough VRAM. CPU execution is not recommended for training these models.

1. Download the project files and enter the project folder:

```bash
cd CV_final
```

2. Create and activate a Conda environment. Conda installation instructions are available in the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda create -n happywhale-cv python=3.11 -y
conda activate happywhale-cv
```

3. Install PyTorch based on your system configuration:

   - Visit the [official PyTorch local installation selector](https://pytorch.org/get-started/locally/).
   - Select your OS, package type (`pip`), and CUDA version.
   - Run the generated command.

   Example for CUDA 12.6:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

4. Install the remaining packages:

```bash
pip install -r requirements.txt
```

5. Prepare Kaggle credentials or login through KaggleHub. For credential setup, see the [Kaggle API documentation](https://www.kaggle.com/docs/api).

```bash
kaggle competitions download -c happy-whale-and-dolphin
```

or run `kagglehub.login()` in the notebooks.

6. Set Hugging Face environment variables. For token creation and permissions, see the [Hugging Face token documentation](https://huggingface.co/docs/hub/security-tokens).

For Windows PowerShell:

```powershell
$env:HF_TOKEN="your_huggingface_token"
$env:HF_REPO_ID="your_name/your_hf_repo"
```

For Linux / macOS:

```bash
export HF_TOKEN="your_huggingface_token"
export HF_REPO_ID="your_name/your_hf_repo"
```

If you want to use the provided model checkpoints and embeddings directly, set `HF_TOKEN` and the corresponding `HF_REPO_ID_*` variables described in the Hugging Face Artifacts section.

## Workflow

### 1. Train a Single Model

Open one of the model notebooks:

- [cv_v2xl_model_hf_ipynb.ipynb](cv_v2xl_model_hf_ipynb.ipynb)
- [cv_B7_model_hf_ipynb.ipynb](cv_B7_model_hf_ipynb.ipynb)
- [cv_B6_model_charm_hf_ipynb.ipynb](cv_B6_model_charm_hf_ipynb.ipynb)
- [cv_convnext_model_hf.ipynb](cv_convnext_model_hf.ipynb)

Check the `CFG` class before running:

```python
CFG.do_train = True
CFG.resume = False
CFG.hf_upload = True
CFG.hf_repo_id = "your_name/your_hf_repo"
```

The notebooks save checkpoints locally and upload them to Hugging Face. The common checkpoint path is:

```python
CFG.hf_checkpoint_file = "checkpoints/last_checkpoint.pth"
```

If you want to continue training from our provided model checkpoint, first download the checkpoint from the corresponding Hugging Face repository, place it at the path expected by the notebook, and enable resume:

```python
CFG.resume = True
CFG.hf_checkpoint_file = "checkpoints/last_checkpoint.pth"
```

Users normally cannot upload new checkpoints back to our Hugging Face repositories unless they have write permission to those repositories. To save your own continued training results, create your own Hugging Face repository and set:

```python
CFG.hf_upload = True
CFG.hf_repo_id = "your_name/your_hf_repo"
```

In short: use our Hugging Face artifacts for downloading / resuming, then upload your new checkpoints and embeddings to your own Hugging Face repository.

### 2. Resume After Colab Disconnects

Colab A100 sessions may disconnect after long runs, often around the 24-hour runtime limit. The notebooks are designed to resume from Hugging Face checkpoints.

To resume:

1. Reopen the same notebook.
2. Reconnect to an A100 GPU runtime.
3. Re-run the install, import, data path, and config cells.
4. Use the same Hugging Face repository:

```python
CFG.hf_repo_id = "your_name/your_hf_repo"
CFG.resume = True
CFG.resume_from_hf = True  # download a checkpoint from Hugging Face if the notebook supports this option
CFG.hf_checkpoint_file = "checkpoints/last_checkpoint.pth"
```

5. Run the checkpoint loading cell.
6. Continue training.

If `hf_upload_each_epoch` is enabled, the latest checkpoint should be uploaded after each epoch, so you do not need to restart training from epoch 0.

### 3. Extract Embeddings

After training, switch to embedding extraction:

```python
CFG.do_train = False
CFG.do_embedding = True
```

The notebooks export train/test `.npz` files and upload them to the configured Hugging Face embedding directory, such as:

```python
CFG.hf_embedding_dir = "embeddings"
```

If embeddings are already available on Hugging Face, you can skip extraction and run the notebook cells that download embeddings for inference.

### 4. Build Crop Inference Outputs

To build component matrices and crop-mode submissions:

```python
CFG.run_mode = "crop_inference"
CFG.do_crop_inference = True
```

The notebooks use crop modes such as:

```python
CFG.crop_infer_modes = ["fullbody", "backfin", "none"]
```

The weighted crop combinations are controlled by:

```python
CFG.crop_weight_sets
```

### 5. Run Final Ensemble

Open [cv_final_ensemble_hf.ipynb](cv_final_ensemble_hf.ipynb).

This notebook downloads the saved model outputs from Hugging Face, combines the model/crop matrices, applies score fusion, and generates final submission files.

Before running the final ensemble, make sure the Hugging Face repositories for all model branches are accessible through `HF_TOKEN` and the `HF_REPO_ID_*` environment variables.

If you trained your own model branches by following the previous steps, you can also ensemble your own artifacts. In [cv_final_ensemble_hf.ipynb](cv_final_ensemble_hf.ipynb), update the model repository variables and folder names to match your Hugging Face outputs:

```python
os.environ["HF_REPO_ID_V2XL"] = "your_name/your_v2xl_repo"
os.environ["HF_REPO_ID_B6"] = "your_name/your_b6_repo"
os.environ["HF_REPO_ID_B7"] = "your_name/your_b7_repo"
os.environ["HF_REPO_ID_CONV"] = "your_name/your_convnext_repo"
```

Also check the crop weights, embedding folder names, and crop submission folder names used inside the final ensemble notebook. They should point to the artifacts generated by your own training / inference runs.

## Results

### Overall Pipeline

The figure below summarizes the proposed training and inference pipeline, including pseudo-label training, embedding extraction, crop-mode inference, and final ensemble fusion.

![Proposed pipeline](fig/proposed_pipeline_pseudo.png)

### Leaderboard Results

| Method | Public | Private |
| --- | ---: | ---: |
| EfficientNet-B7 | 0.84643 | 0.81620 |
| EfficientNet-B6 | 0.83664 | 0.80298 |
| EfficientNetV2-XL | 0.83697 | 0.79945 |
| ConvNeXt-B | 0.80054 | 0.74866 |
| Final ensemble | 0.86095 | **0.83044** |
| Weak baseline | -- | 0.84909 |

The final leaderboard screenshot is shown below.

![Final leaderboard result](fig/final_result.jpg)
