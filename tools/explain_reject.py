"""逐张打印参考照的检测框与画质指标，排查为什么某张被门控拦下。"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from facecmp import config
from facecmp.engine import FaceEngine

engine = FaceEngine()
for name in sorted(os.listdir(config.REFERENCE_DIR)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")) or name.startswith("."):
        continue
    img = cv2.imread(str(config.REFERENCE_DIR / name))
    if img is None:
        continue
    print(f"\n{name}  img={img.shape}")
    boxes = engine.detect(img)
    if not boxes:
        print("   没有任何检测框")
        continue
    for box, score in boxes[:3]:
        face = engine.analyze(img, box, score)
        if face is None:
            print(f"   box={box} 关键点失败")
            continue
        q = face.quality
        print(
            f"   box={box} size={face.size:4d} roll={q['roll']:+6.1f} "
            f"yaw={q['yaw']:+.2f} pitch={q['pitch']:.2f} "
            f"bright={q['brightness']:5.1f} sharp={q['sharpness']:7.1f} "
            f"ok={face.acceptable!s:5s} {face.hint}"
        )
