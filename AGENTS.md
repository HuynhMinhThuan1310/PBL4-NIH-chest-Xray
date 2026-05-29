# AGENTS.md

## Project Shape
- This repo is a Kaggle/TensorFlow chest X-ray experiment workspace plus Word/PDF report assets, not a package app.
- There is no root README for the project, CI, lockfile, formatter config, test runner, or repo-local OpenCode config.
- Main executable sources are notebooks in `trains/`; explainability notebooks are in `code/explainability_notebooks/`.
- Current Word report is `Bao_Cao_PBL4_VGG19_DenseNet121_FCSSAM.docx`.

## Report Assets
- `report_assets/figures/` contains generated PNG/SVG sources used for the Word report.
- Figures already inserted into `.docx` are embedded; moving source images does not remove them from Word.
- Do not edit user-authored references (`fanet.docx`, `vgg19.docx`, PDFs, old reports) unless explicitly asked.

## Current Main Models
- The current report compares `VGG19-GAP` and `DenseNet121-FCSSAM`.
- Treat `VGG19-GAP` as a main independent model, not a secondary/baseline/sub-model.
- Describe `DenseNet121-FCSSAM` as following FA-Net/FCSSAM ideas, not as a guaranteed exact reproduction of the paper.
- `legacy_resnet50_*` notebooks are old comparison artifacts and are not part of the current main report.

## Notebook Sources
- Main dataset pair: `trains/vgg19_gap_phien_ban_chinh_3_1_3.ipynb` and `trains/densenet121_fcssam_phien_ban_chinh_3_1_3.ipynb`.
- Auxiliary pairs use suffixes `phien_ban_ban_dau_2_1_1` and `tang_tran_dich_2_1_3`.
- Explainability notebooks: `code/explainability_notebooks/vgg19_gap_gradcam_fact.ipynb`, `code/explainability_notebooks/densenet121_fcssam_gradcam_attention_fact.ipynb`, and legacy `code/explainability_notebooks/legacy_resnet50_fcssam_gradcam_fact.ipynb`.

## Labels And Report Prose
- Active code class order is `normal`, `pneumonia`, `effusion`.
- In Vietnamese report prose, render labels as `bình thường`, `viêm phổi`, `tràn dịch màng phổi`; avoid raw code labels.
- Do not reintroduce old `pneumonia_only` / `effusion_only` names unless intentionally discussing old runs.
- Avoid reader-facing internal terms such as `normal3`, notebook filenames, Kaggle paths, or `.ipynb` names in the Word report.
- Do not discuss NIH multi-label/mixed-label caveats in the report unless the user explicitly asks; describe the groups simply as the three target classes.

## Dataset Versions
- Current main dataset is the `3 : 1 : 3` train ratio version, reported as `phiên bản chính` or `phiên bản cuối`.
- Auxiliary versions: `2 : 1 : 1` is `phiên bản ban đầu`; `2 : 1 : 3` is `phiên bản tăng tràn dịch`.
- Required Kaggle image inputs for training are NIH chest X-rays and the chest-xray-pneumonia dataset; manifests alone are not enough because image paths point to original Kaggle datasets.
- Current main manifest root used in notebooks is `/kaggle/input/datasets/thuanminh1310/datasets-version3/pbl4_main_effusion_heavy_normal3_nih_datasets`.

## Fair Comparison Constraints
- VGG19-GAP and DenseNet121-FCSSAM must use the same dataset version, split, `IMAGE_SIZE`, grayscale decode, resize method, contrast step, grayscale-to-RGB conversion, augmentation, and normalization.
- Current shared preprocessing before model: grayscale decode, bilinear resize to `256x256`, `enhance_xray_contrast`, `tf.image.grayscale_to_rgb`, train-only `RandomTranslation(0.015, 0.015)`, `RandomZoom(0.03, 0.03)`, `RandomFlip("horizontal")`, RGB->BGR, subtract Caffe/ImageNet mean `[103.939, 116.779, 123.680]`.
- Architecture differences are allowed after preprocessing: VGG19-GAP uses `vgg_gap`; DenseNet121-FCSSAM uses DenseNet121 + FCSSAM + `densenet_gap`.

## Current Results To Preserve
- Main VGG19-GAP test-threshold metrics: macro F1 `0.852399`, effusion precision `0.603550`, recall `0.748166`, F1 `0.668122`, macro AUC `0.957893`.
- Main DenseNet121-FCSSAM test-threshold metrics: macro F1 `0.859390`, effusion precision `0.616000`, recall `0.753056`, F1 `0.677668`, macro AUC `0.965433`.
- DenseNet121-FCSSAM `2 : 1 : 3` auxiliary run has higher macro F1 (`0.869955`) and effusion F1 (`0.700893`), but the report uses the `3 : 1 : 3` pair as the main fair comparison.

## Evaluation And Explainability
- Prefer `macro_f1`, per-class `tràn dịch màng phổi` precision/recall/F1, confusion matrix, and AUC over accuracy alone.
- Threshold notebooks tune an `effusion` threshold on validation, then apply the selected threshold to test.
- Grad-CAM layers used for report comparison: VGG19-GAP `block4_conv4`; DenseNet121-FCSSAM `conv4_block6_concat`.
- FACT deletion AUC lower is better in report prose because target probability drops faster after deleting hot regions.

## Validation
- Validate edited notebooks with: `python -c "import json, ast; from pathlib import Path; files=['NOTEBOOK.ipynb']; [ast.parse(''.join(c.get('source', []))) for f in files for c in json.loads(Path(f).read_text(encoding='utf-8'))['cells'] if c.get('cell_type')=='code']; print('ok')"`.
- Full training usually cannot run locally because data paths are Kaggle-only; local verification is limited to JSON/syntax/string checks unless Kaggle artifacts are downloaded.
- For Word edits, verify the document opens with Word COM when possible and check stale text such as `ResNet`, `normal3`, `notebook`, `.ipynb`, `pneumonia_only`, `effusion_only`, `Vị trí chèn`, `sẽ trình bày`.

## Editing Rules
- Use `apply_patch` for manual text edits; avoid ad-hoc writes that can corrupt notebooks.
- Create new experiment filenames and new `/kaggle/working/...` output dirs; do not overwrite completed notebooks or Kaggle result artifacts.
- Keep generated notebooks output-cleared unless the user intentionally provides Kaggle-output notebooks for analysis.
- In the Word report, keep table captions above tables and figure captions below figures; captions are centered/bold, explanatory paragraphs are justified/not bold.
