from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def render_calibration_v3_markdown(payload: Dict[str, Any], export_dir: Path) -> str:
  meta = payload.get("meta", {})
  parity_max_tol = meta.get("parity_max_tol", meta.get("parity_tol", 5e-5))
  parity_mean_tol = meta.get("parity_mean_tol", 5e-6)
  ort_provider = meta.get("ort_provider", "CPUExecutionProvider")
  batch_size = meta.get("batch_size", "N")
  seed = meta.get("seed", "")
  return f"""# calibration_v3_trial0 校准与校验说明

## 1. 产物说明
- ONNX：`patch_encoder.onnx`、`slide_encoder.onnx`
- 配置：`export_manifest.json`（含 preprocess）
- 校准快照：`calibration_v3_trial0.pkl`（schema_version=1.0）
- 完整校验报告：`validation_main.html`

## 2. pkl 内容速查
- trial_idx=0，batch_size={batch_size}，seed={seed}
- inputs.patch_tensor: [N,3,H,W]
- intermediate.embedding_arr: pt / ort 及 error 统计
- outputs.pt / outputs.ort: slide_encoder 全部 export 输出

## 3. 离线重放校验

### 3.1 环境
Python + numpy + mmengine + onnxruntime；工作目录 `{export_dir}`。

### 3.2 加载并重放
```python
from pathlib import Path
import mmengine
import numpy as np
import onnxruntime as ort

export_dir = Path("{export_dir}")
data = mmengine.load(export_dir / "calibration_v3_trial0.pkl")
patch = data["inputs"]["patch_tensor"]["array"]

patch_sess = ort.InferenceSession(str(export_dir / "patch_encoder.onnx"), providers=["{ort_provider}"])
emb = patch_sess.run(None, {{"patch_tensor": patch.astype(np.float32)}})[0]
ref_emb = data["intermediate"]["embedding_arr"]["ort"]["array"]
emb_diff = np.abs(emb - ref_emb)
print("embedding max diff", float(np.max(emb_diff)), "mean diff", float(np.mean(emb_diff)))

slide_sess = ort.InferenceSession(str(export_dir / "slide_encoder.onnx"), providers=["{ort_provider}"])
slide_out = slide_sess.run(None, {{"embedding_arr": emb.astype(np.float32)}})
slide_names = data["outputs"]["slide_output_names"]
for name, arr, ref in zip(slide_names, slide_out, [data["outputs"]["ort"][n] for n in slide_names]):
    if "label" in name:
        ok = int(np.asarray(arr).reshape(-1)[0]) == int(np.asarray(ref).reshape(-1)[0])
        print(name, "exact", ok)
    else:
        diff_arr = np.abs(np.asarray(arr) - np.asarray(ref))
        print(
            name,
            "max diff", float(np.max(diff_arr)),
            "mean diff", float(np.mean(diff_arr)),
            "passed",
            float(np.max(diff_arr)) < {parity_max_tol} and float(np.mean(diff_arr)) < {parity_mean_tol},
        )
```

## 4. 判定标准
- float: max diff < {parity_max_tol} 且 mean diff < {parity_mean_tol}
- int label: exact match

## 5. 导出上下文
- config_path: {meta.get("config_path", "")}
- ckpt_path: {meta.get("ckpt_path", "")}
- model_type: {meta.get("model_type", "")}
- ort_provider: {ort_provider}
- created_at: {meta.get("created_at", "")}
"""


def write_calibration_v3_markdown(output_dir: Path, payload: Dict[str, Any]) -> Path:
  output_dir = Path(output_dir)
  path = output_dir / "calibration_v3_trial0.md"
  path.write_text(render_calibration_v3_markdown(payload, output_dir.resolve()), encoding="utf-8")
  return path
