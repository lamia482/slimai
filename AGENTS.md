# AGENTS.md

## Cursor Cloud specific instructions

### Overview
SLIMAI is a deep learning framework for computational pathology. It uses PyTorch + MMEngine for config-driven training pipelines. Key tasks: classification, MIL (Multiple Instance Learning on WSIs), DINO self-supervised learning, and detection.

### Project layout
- `slimai/` — main package (data, helper, models, runner, templates)
- `tools/run.py` — primary entry point for training/inference/evaluation
- `slimai/templates/` — config templates (classify.py, h5_mil.py, wsi_mil.py, dino.py)
- `chore/` — build scripts and `requirements.txt`, `setup.py`

### Running training
See `README.md` for standard commands. Key entry:
```
python tools/run.py --config slimai/templates/<template>.py --device <cpu|cuda>
```

### Important caveats for Cloud Agents (no GPU)

1. **CPU-only mode**: Use `--device cpu`. Also requires overriding `pin_memory=False` in dataloader configs and disabling AMP (`gradient.amp=False`). The `classify.py` template defaults to `pin_memory=True` which crashes on CPU-only PyTorch.

2. **Torch version**: The `slimai/helper/common.py` REQUIREMENTS check uses naive string comparison for versions. Torch versions >= 2.10.x will fail the `>= "2.7.0"` check due to lexicographic comparison (`"1" < "7"`). Use torch 2.7.x through 2.9.x to avoid this.

3. **SDK stub**: The KFBIO `sdk.reader` module is proprietary and not pip-installable. A stub package must be present in site-packages for the import chain to work. The stub raises `NotImplementedError` when actually called — only WSI (.kfb) workflows need the real SDK.

4. **SwanLab**: Required at runtime (the `Record.format()` method references `swanlab.Image`/`swanlab.Text`). Set `SWANLAB_MODE=disabled` to skip cloud logging.

5. **No test framework**: The project has no pytest/unittest tests. Validation is done by running training templates.

### Lint
Pyright is configured in `.vscode/settings.json` with `typeCheckingMode: "basic"`. Run:
```
pyright slimai/
```
Pre-existing type errors (~500) are expected; focus on new errors from your changes.

### PYTHONPATH
The workspace root must be on `PYTHONPATH` since there's no `pyproject.toml` at the root — `setup.py` lives in `chore/`:
```
export PYTHONPATH="/workspace:$PYTHONPATH"
```
