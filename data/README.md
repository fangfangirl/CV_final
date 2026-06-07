# Dataset Directory

Place local datasets under this directory when running the notebooks outside Colab.

Download pages:

- Happy Whale and Dolphin competition: https://www.kaggle.com/competitions/happy-whale-and-dolphin
- Full-body annotations: https://www.kaggle.com/datasets/jpbremer/fullbodywhaleannotations
- Backfin cropped images: https://www.kaggle.com/datasets/genlaxai/happy-whale-backfin-cropped
- Fullybody charm crops: https://www.kaggle.com/datasets/changchoufang/fullybody-charm

The recommended structure is:

```bash
data/
  happy-whale-and-dolphin/
    train.csv
    sample_submission.csv
    train_images/
      <image files>
    test_images/
      <image files>
  fullbodywhaleannotations/
    <full-body annotation files>
  happy-whale-backfin-cropped/
    <backfin crop files>
  fullybody-charm/
    <charm / additional crop files>
  pseudo.csv
```

The notebooks also support KaggleHub / Colab paths such as `/content` and `/kaggle/input`.
If you use a different local layout, update the path candidates in the `CFG` cell or in the `_get_or_download_kaggle_paths()` helper of each notebook.

For example, the local fallback paths may look like:

```python
official_path = "data/happy-whale-and-dolphin"
fullbody_path = "data/fullbodywhaleannotations"
backfin_path_local = "data/happy-whale-backfin-cropped"
```

Replace these with your own absolute or relative paths if your data is stored elsewhere.

Alternatively, define these variables in a cell before running `_get_or_download_kaggle_paths()`:

```python
happy_whale_and_dolphin_path = "data/happy-whale-and-dolphin"
jpbremer_fullbodywhaleannotations_path = "data/fullbodywhaleannotations"
backfin_path = "data/happy-whale-backfin-cropped"
fullbody_charm_path = "data/fullybody-charm"  # only needed by notebooks that use charm crops
```

The provided pseudo-label file should be placed here:

```bash
data/pseudo.csv
```

When enabling pseudo-label training, set:

```python
CFG.use_pseudo = True
CFG.pseudo_csv = "data/pseudo.csv"
```

Keep real dataset files local. This directory keeps this README and the provided [pseudo.csv](pseudo.csv) in the code package.
