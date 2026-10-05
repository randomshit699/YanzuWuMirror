"""从 impostor_meta.json 重建 impostor_labels.json（缓存丢失后的补救）。"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facecmp import config

meta_path = config.ASSET_DIR / "calibration" / "impostor_meta.json"
emb_path = config.ASSET_DIR / "calibration" / "impostor_embeddings.npy"

meta = json.loads(meta_path.read_text("utf-8"))
labels = [name for name, n in meta["people"].items() for _ in range(n)]

n_emb = int(np.load(emb_path).shape[0])
if len(labels) != n_emb:
    print(f"数量不一致: 标签 {len(labels)} vs 描述子 {n_emb}")
    raise SystemExit(1)

out = config.ASSET_DIR / "calibration" / "impostor_labels.json"
out.write_text(json.dumps(labels, ensure_ascii=False, indent=1), "utf-8")
print(f"已重建 {out}（{len(labels)} 个标签，与描述子数量一致）")
