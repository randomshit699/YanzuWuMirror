"""引擎冒烟测试：对参考图做人脸检测 + 对齐 + 描述子，并输出可视化。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from facecmp import config
from facecmp.engine import FACE_TEMPLATE, FaceEngine, similarity_transform

engine = FaceEngine()
ref_dir = config.REFERENCE_DIR
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")
os.makedirs(out_dir, exist_ok=True)

embs = {}
for name in sorted(os.listdir(ref_dir)):
    if not name.lower().endswith((".jpg", ".jpeg", ".png")) or name.startswith("_"):
        continue
    path = os.path.join(ref_dir, name)
    img = cv2.imread(path)
    if img is None:
        print("READ FAIL", path)
        continue
    t0 = time.time()
    boxes = engine.detect_boxes(img)
    t_det = time.time() - t0
    face = engine.best_face(img)
    print(f"\n== {name}  {img.shape}")
    print("   boxes:", [(b, round(s, 3)) for b, s in boxes][:5], f"det={t_det*1000:.0f}ms")
    if face is None:
        print("   NO FACE")
        continue
    print("   chosen box", face.box, "size", face.size)
    print("   landmarks:\n", np.round(face.landmarks, 1))
    print("   quality:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in face.quality.items()})
    print("   hint:", face.hint or "(ok)")
    print("   emb norm", round(float(np.linalg.norm(face.embedding)), 4), "dim", face.embedding.shape)
    embs[name] = face.embedding

    # 可视化：原图框+点，以及对齐后的人脸
    vis = img.copy()
    x, y, w, h = face.box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 3)
    for i, p in enumerate(face.landmarks):
        cv2.circle(vis, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    tag = name.replace(".jpg", "")
    cv2.imwrite(os.path.join(out_dir, f"vis_{tag}.jpg"), vis)
    cv2.imwrite(os.path.join(out_dir, f"aligned_{tag}.jpg"), face.aligned)
    # 把模板点画到对齐图上，验证对齐是否正确
    chk = face.aligned.copy()
    for p in FACE_TEMPLATE:
        cv2.circle(chk, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(out_dir, f"aligned_chk_{tag}.jpg"), chk)

# 参考图互相之间的相似度（应该彼此很高）
print("\n== 参考图两两相似度")
names = list(embs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        c = float(embs[names[i]] @ embs[names[j]])
        print(f"   {names[i][:28]:30s} vs {names[j][:28]:30s} cos={c:.4f}")

# 相似变换自检：单位正方形 -> 旋转缩放平移
src = np.array([[0, 0], [1, 0], [0, 1]], float)
M = similarity_transform(src, FACE_TEMPLATE[:3])
print("\n== similarity_transform src->template[:3] =\n", np.round(M, 4))
print("   expect row0 ~ [65.53, -0.20, 30.29]")
