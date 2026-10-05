"""检查参考图库：逐张列出画质指标，并两两比较找重复/误匹配。"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from facecmp import config
from facecmp.engine import FaceEngine

engine = FaceEngine()
embs = {}
rows = []
for f in sorted(os.listdir(config.REFERENCE_DIR)):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")) or f.startswith("_"):
        continue
    img = cv2.imread(str(config.REFERENCE_DIR / f))
    if img is None:
        print("读取失败", f)
        continue
    face = engine.best_face(img, require_acceptable=True)
    if face is None:
        print("无人脸", f)
        continue
    embs[f] = face.embedding
    q = face.quality
    rows.append(
        f"{f[:46]:48s} size={face.size:4d} ok={face.acceptable!s:5s} "
        f"roll={q['roll']:+6.1f} yaw={q['yaw']:+.2f} pitch={q['pitch']:.2f} "
        f"bright={q['brightness']:5.0f} sharp={q['sharpness']:7.0f}  {face.hint}"
    )
print("\n".join(rows))

print("\n== 两两余弦（<0.90 高度可疑：不同人或检测失败）")
names = list(embs)
suspects = []
for a, b in itertools.combinations(names, 2):
    c = float(embs[a] @ embs[b])
    flag = ""
    if c < 0.85:
        flag = "  <== 疑似不同人"
        suspects.append((a, b, c))
    elif c > 0.99:
        flag = "  <== 疑似同一张图"
    print(f"{c:.4f}  {a[:40]:42s} {b[:40]:42s}{flag}")
print("\n可疑对:", len(suspects))
