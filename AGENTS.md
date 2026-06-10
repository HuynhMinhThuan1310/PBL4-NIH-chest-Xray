# AGENTS.md

## Project Shape
- This is a Kaggle/TensorFlow chest X-ray experiment workspace plus Word/PDF report assets, not an app/package repo.
- There is no CI, lockfile, formatter config, test runner, or repo-local OpenCode config; local verification is limited to document checks and notebook JSON/Python syntax.
- Current report to edit is `report/Báo Cáo PBL4.docx`; `report/bao-cao.docx` is an older report copy.
- `README.md` still references the old report path in places; prefer this file and the current `.docx` contents when they conflict.

## Main Scope
- The report compares two main independent models on the main `2 : 1 : 1` dataset: `VGG19-GAP` and `DenseNet121-FCSSAM`.
- Treat `VGG19-GAP` as a main independent model, not a baseline/sub-model.
- Describe `DenseNet121-FCSSAM` as following FA-Net/FCSSAM ideas, not as an exact reproduction of the paper.
- `2 : 1 : 3` and `3 : 1 : 3` are auxiliary comparison runs only.
- Legacy `ResNet50-FCSSAM` notebook code is old artifact content and not part of the current main report.

## Source Files
- Dataset creation: `notebooks/data.ipynb`.
- Main training/evaluation: `notebooks/vgg-gap.ipynb`, `notebooks/densenet.ipynb`.
- Explainability: `notebooks/gradcam-vgg.ipynb`, `notebooks/gradcam-densenet.ipynb`.
- Auxiliary comparisons: `notebooks/so-sanh/` for `2 : 1 : 3` and `3 : 1 : 3` only.
- Extra non-main experiment: `notebooks/vgg-fcssam.ipynb`; mention only as a tried variant that did not improve over VGG19-GAP.

## Report Language And Labels
- Active main notebook class order is `normal`, `pneumonia_only`, `effusion_only`.
- In Vietnamese report/slide prose, render labels as `bình thường`, `viêm phổi`, `tràn dịch màng phổi`.
- Avoid reader-facing internal terms in the Word report: `normal3`, notebook filenames, Kaggle paths, `.ipynb`, `pneumonia_only`, `effusion_only`.
- Do not discuss NIH multi-label/mixed-label caveats unless explicitly asked; describe the groups as the three target classes.

## Current Results To Preserve
- VGG19-GAP test-threshold: accuracy `0.863618`, macro F1 `0.835117`, weighted F1 `0.870336`, macro AUC `0.951155`; tràn dịch precision `0.552361`, recall `0.739011`, F1 `0.632197`; threshold `0.71`.
- DenseNet121-FCSSAM test-threshold: accuracy `0.884631`, macro F1 `0.841222`, weighted F1 `0.884030`, macro AUC `0.953359`; tràn dịch precision `0.650704`, recall `0.634615`, F1 `0.642559`; threshold `0.79`.
- Auxiliary `2 : 1 : 3`: macro F1 VGG `0.843077`, DenseNet `0.869955`; effusion F1 VGG `0.638858`, DenseNet `0.700893`.
- Auxiliary `3 : 1 : 3`: macro F1 VGG `0.852399`, DenseNet `0.859390`; effusion F1 VGG `0.668122`, DenseNet `0.677668`.
- VGG19-FCSSAM trial on `2 : 1 : 1`: accuracy `0.859497`, macro F1 `0.824360`, weighted F1 `0.864322`, macro AUC `0.939412`, effusion F1 `0.611591`; conclude it did not improve over VGG19-GAP.

## Pipeline And Explainability Details
- Fair comparisons must keep the same dataset version, split, image size, grayscale decode, resize, contrast step, RGB conversion, augmentation, and normalization.
- Shared preprocessing: grayscale decode, bilinear resize to `256x256`, `enhance_xray_contrast`, grayscale-to-RGB, train-only `RandomTranslation(0.015, 0.015)`, `RandomZoom(0.03, 0.03)`, `RandomFlip("horizontal")`, RGB-to-BGR, subtract Caffe/ImageNet mean `[103.939, 116.779, 123.680]`.
- Threshold evaluation tunes the tràn dịch threshold on validation, then applies the chosen threshold to test.
- Report Grad-CAM++ layers: VGG19-GAP `block5_conv4`; DenseNet121-FCSSAM `conv4_block24_concat`.
- FACT deletion AUC lower is better because target probability drops faster after deleting hot regions: VGG19-GAP `0.453555`, DenseNet121-FCSSAM `0.320526`.
- `gradcam-densenet.ipynb` includes FCSSAM internal attention maps (`CAM effect`, `SAMavg`, `SAMmax`, `FCSSAM`); VGG19-GAP has Grad-CAM++ only, not internal attention maps.

## Word Report Rules
- Do not edit user-authored references in `report/refs/` unless explicitly asked.
- In Word, keep table captions above tables and figure captions below figures.
- Captions are centered/bold; explanatory paragraphs are justified/not bold.
- Prefer PNG for inserted figures; SVG can have Word caching/font issues. `report/figs/vgg19_gap_architecture.svg` is a source figure, not ideal for direct Word insertion.
- If adding headings, set Word outline levels so they appear in the Navigation Pane.

## Validation
- Validate edited notebooks with: `python -c "import json, ast; from pathlib import Path; files=['NOTEBOOK.ipynb']; [ast.parse(''.join(c.get('source', []))) for f in files for c in json.loads(Path(f).read_text(encoding='utf-8'))['cells'] if c.get('cell_type')=='code']; print('ok')"`.
- Full training usually cannot run locally because data paths/checkpoints are Kaggle-only.
- For Word edits, verify `zip bad: None`, parse `word/document.xml`, and open with Word COM when possible.
- After Word edits, scan for stale text: `normal3`, `.ipynb`, `pneumonia_only`, `effusion_only`, `Vị trí chèn`, `sẽ trình bày`, `conv4_block6_concat`, `block4_conv4`, `3.3.5`.

## Editing Gotchas
- Use `apply_patch` for manual text/notebook edits; avoid ad-hoc writes that can corrupt notebooks.
- Create new experiment filenames and new `/kaggle/working/...` output dirs; do not overwrite completed Kaggle artifacts.
- Keep generated notebooks output-cleared unless the user intentionally provides Kaggle-output notebooks for analysis.
