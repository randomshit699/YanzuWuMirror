"""检查对齐预览图是否正常输出（截图里那块纯蓝说明有问题）。"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from facecmp import config
from facecmp.engine import FaceEngine

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")
os.makedirs(OUT, exist_ok=True)

engine = FaceEngine()
for name in sorted(os.listdir(config.REFERENCE_DIR)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")) or name.startswith("_"):
        continue
    img = cv2.imread(str(config.REFERENCE_DIR / name))
    face = engine.best_face(img, require_acceptable=True)
    if face is None:
        print(name, "no face")
        continue
    p = face.preview
    print(f"{name[:40]:42s} preview {p.shape} dtype={p.dtype} "
          f"mean={p.mean():.1f} std={p.std():.1f} "
          f"contig={p.flags['C_CONTIGUOUS']} BGR=({p[:,:,0].mean():.0f},{p[:,:,1].mean():.0f},{p[:,:,2].mean():.0f})")
    cv2.imwrite(os.path.join(OUT, f"preview_{name[:24]}.png"), p)

    # 关键点检查
    lm = face.landmarks
    print(f"   eye_r={lm[36:42].mean(0).round(0)} eye_l={lm[42:48].mean(0).round(0)} "
          f"nose={lm[30].round(0)} mouth_r={lm[48].round(0)} mouth_l={lm[54].round(0)} chin={lm[8].round(0)}")
