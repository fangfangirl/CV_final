# Happy Whale and Dolphin Identification

這個專案是「基於深度學習之視覺辨識」期末專題的程式碼整理，主題為 Kaggle
Happy Whale and Dolphin Identification。專案使用多個影像分類/檢索模型產生
embedding，再以 KNN、class prototype、species bonus 與多模型 ensemble 產生最終
submission。

## 專案內容

| 檔案 | 說明 |
| --- | --- |
| `cv_V2_model.ipynb` | EfficientNetV2-XL 單模型訓練、embedding extraction、crop inference。 |
| `cv_V2_model_7.ipynb` | EfficientNetV2-L 單模型訓練、embedding extraction、crop inference，並包含多模型 concat 相關設定。 |
| `cv_B7_model.ipynb` | EfficientNet-B7 單模型訓練、embedding extraction、crop inference。 |
| `cv_B6_model_charm_hf_ipynb.ipynb` | EfficientNet-B6 單模型流程，包含 full-body/backfin crop 與 pseudo/charm 設定。 |
| `cv_final_ensemble_hf.ipynb` | 從 Hugging Face 下載各模型輸出，建立 KNN/prototype matrix，做 weighted ensemble 並輸出 submission。 |
| `scripts/clean_notebooks_for_github.py` | 上傳 GitHub 前清除 notebook outputs、widget metadata 與誤留的 token 字串。 |

## 方法摘要

- Backbone：`tf_efficientnetv2_xl.in21k`、`tf_efficientnetv2_l.in21k`、`tf_efficientnet_b7_ns`、`tf_efficientnet_b6_ns`
- Loss / head：ArcFace-style margin head、adaptive margin、Focal Loss
- Data augmentation：Albumentations
- Crop strategy：original image、full-body bounding box、backfin crop 等多種 crop mode
- Inference：embedding similarity、KNN、class prototype、species-aware score adjustment
- Ensemble：不同模型與 crop mode 的 prediction matrix 加權融合

## 環境需求

建議在 Google Colab 或 Kaggle Notebook 使用 GPU 執行。原始實驗使用大型
EfficientNet 模型，訓練時建議使用高記憶體 GPU；單機 CPU 不適合完整重跑。

主要套件列在 `requirements.txt`，可先安裝：

```bash
pip install -r requirements.txt
```

Notebook 內也有對 `kagglehub`、`timm`、`albumentations`、
`huggingface_hub` 的安裝指令，Colab/Kaggle 可直接照 notebook cell 執行。

## 資料來源

本專案會透過 Kaggle / KaggleHub 使用以下資料：

- `happy-whale-and-dolphin` competition dataset
- `jpbremer/fullbodywhaleannotations`
- `genlaxai/happy-whale-backfin-cropped`
- `changchoufang/fullybody-charm`

資料量很大，請不要把 dataset、cache、模型權重、embedding、prediction matrix
或 submission 輸出直接 commit 到 GitHub。這些檔案已在 `.gitignore` 中排除。

## Token 設定

Notebook 需要 Kaggle 與 Hugging Face 權限時，請使用環境變數或 notebook secret，
不要把 token 寫死在程式碼中。

Hugging Face token 可用以下任一方式提供：

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

或在 Colab/Kaggle Secret 中建立 `HF_TOKEN`。Kaggle 資料下載請使用
`kagglehub.login()` 或 Kaggle Notebook 內建 credentials。

## 執行流程

1. 先執行單模型 notebook：
   - `cv_V2_model.ipynb`
   - `cv_V2_model_7.ipynb`
   - `cv_B6_model_charm_hf_ipynb.ipynb`
2. 在 `CFG` 中確認：
   - `run_mode`
   - `hf_repo_id`
   - `output_dir`
   - batch size / image size / epoch 設定
3. 產生並上傳各模型的 checkpoint、embedding、matrix 或 crop submission 到 Hugging Face。
4. 執行 `cv_final_ensemble_hf.ipynb`，下載各模型輸出並產生 ensemble submission。

## GitHub 上傳前檢查

上傳前建議重新執行：

```bash
python scripts/clean_notebooks_for_github.py
```

接著確認沒有 token 或大型輸出：

```bash
git status
git diff --stat
```

若這個資料夾還不是 git repository，可初始化後上傳：

```bash
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## 注意事項

- 已清除 notebook outputs，避免 GitHub 檔案過大。
- 若之前 token 曾出現在 notebook 中，建議到 Kaggle / Hugging Face 後台撤銷並重新產生。
- Hugging Face repo id、資料路徑與模型輸出位置需依自己的帳號與執行環境調整。
