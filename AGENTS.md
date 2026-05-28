# AGENTS.md

## Project Shape
- This workspace is a Kaggle/TensorFlow chest X-ray experiment area plus Word/PDF report assets, not a package app.
- There is no README, CI, lockfile, formatter config, test runner, or repo-local OpenCode config; the executable sources of truth are notebooks in `trains/`.
- Preserve user-authored report files (`*.docx`, `Bao_Cao_VGG19_ResNet50_FCSSAM.md`) unless explicitly asked to edit them.

## Current Results Folder
- `trains/` contains the 4 completed VGG19/ResNet50-FCSSAM pairs; do not delete or overwrite these notebooks.
- Best current pair is `vgg19-on-effusion-heavy-normal3-nih.ipynb` and `train-resnet50-fcssam-on-effusion-heavy-normal3.ipynb`.
- Best observed ResNet50-FCSSAM test threshold result: macro F1 `0.865358`, effusion precision `0.643892`, recall `0.760391`, F1 `0.697309`, AUC `0.935079`.
- Best observed VGG19 test threshold result: macro F1 `0.852399`, effusion precision `0.603550`, recall `0.748166`, F1 `0.668122`, AUC `0.926994`.

## Labels And Prose
- Current active class order is `normal`, `pneumonia`, `effusion`.
- In Vietnamese prose, render labels as `bình thường`, `viêm phổi`, `tràn dịch màng phổi`; avoid raw code labels in report text.
- Old notebooks may contain `pneumonia_only` / `effusion_only`; do not reintroduce those names for current experiments unless intentionally discussing old runs.

## Dataset Lineage
- Old clean pair used `normal:pneumonia_only:effusion_only` with train `2:1:1`.
- Dataset ver1 uses `normal`, `pneumonia`, `effusion`; effusion excludes images containing both Effusion and Pneumonia.
- Effusion-heavy ver2 uses all NIH images containing `Effusion`, train `normal:pneumonia:effusion = 2:1:3`, val/test `70:15:15`.
- Current preferred normal3 dataset uses all NIH images containing `Effusion`, train `3:1:3`, val/test `70:15:15`.
- Current normal3 Kaggle root used by the final pair: `/kaggle/input/datasets/thuanminh1310/datasets-version3/pbl4_main_effusion_heavy_normal3_nih_datasets`.
- Manifest filenames for normal3: `main_train_3_1_3_effusion_heavy_normal3.csv`, `main_val_70_15_15_effusion_heavy_normal3.csv`, `main_test_70_15_15_effusion_heavy_normal3.csv`.

## Required Kaggle Inputs
- NIH image root: `/kaggle/input/datasets/organizations/nih-chest-xrays/data`.
- Pneumonia image root: `/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia` or its `chest_xray` child.
- Manifests contain absolute Kaggle image paths, so training notebooks need both original image datasets as inputs, not just the manifest dataset.

## Fair Comparison Constraint
- VGG19 and ResNet50-FCSSAM must use the same dataset version, split, `IMAGE_SIZE`, grayscale decode, resize method, contrast step, grayscale-to-RGB conversion, augmentation, and normalization.
- Current shared preprocessing before model: grayscale decode, bilinear resize to `256x256`, `enhance_xray_contrast`, `tf.image.grayscale_to_rgb`, train-only `RandomTranslation(0.015, 0.015)`, `RandomZoom(0.03, 0.03)`, `RandomFlip("horizontal")`, RGB->BGR, subtract Caffe/ImageNet mean `[103.939, 116.779, 123.680]`.
- Architecture differences are allowed after input preprocessing: VGG19 uses `vgg_gap`; ResNet50-FCSSAM uses ResNet50 + FCSSAM + `resnet_gap`.

## Training/Evaluation Notes
- P100 notebooks use TensorFlow/Keras, `mixed_float16`, `IMAGE_SIZE = 256`, `BATCH_SIZE = 16`, `EFFUSION_BOOST = 1.0` in current normal3 runs.
- Threshold notebooks tune an `effusion` threshold on validation from `0.30` to `0.90`, then apply that threshold to test.
- Prefer `macro_f1`, per-class `effusion` precision/recall/F1, confusion matrix, and AUC over accuracy alone.
- JIKI paper results are not directly comparable: it is a smaller binary `normal` vs `pleural effusion` task, while current work is 3-class single-label with mixed NIH labels.

## Validation
- Validate notebook JSON and Python code cells after edits: `python -c "import json, ast; from pathlib import Path; files=['NOTEBOOK.ipynb']; [ast.parse(''.join(c.get('source', []))) for f in files for c in json.loads(Path(f).read_text(encoding='utf-8'))['cells'] if c.get('cell_type')=='code']; print('ok')"`.
- Local full training usually cannot run because data paths are Kaggle-only; local verification is limited to syntax/string checks unless Kaggle artifacts are downloaded.

## Outputs To Ask For After Kaggle Runs
- Summary metrics: `/kaggle/working/<run>/artifacts/<model_key>_summary_metrics.csv`.
- Per-class report: `/kaggle/working/<run>/artifacts/<model_key>_class_report.csv`.
- Threshold search: `/kaggle/working/<run>/artifacts/<model_key>_effusion_threshold_search.csv`.
- Confusion figures: `/kaggle/working/<run>/figures/<model_key>_*_confusion.png`.
- Training history: `/kaggle/working/<run>/models/<model_key>/history.csv`.

## Editing Conventions
- Use `apply_patch` for manual edits; avoid ad-hoc writes that can corrupt notebooks.
- Create new experiment filenames and new `/kaggle/working/...` output dirs; do not overwrite completed notebooks in `trains/` or result artifacts.
- Keep generated notebooks output-cleared unless the user intentionally provides Kaggle-output notebooks for analysis.
