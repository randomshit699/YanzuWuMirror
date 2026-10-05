"""标定：统计"陌生人 vs 吴彦祖"的余弦分布，生成 0-100 分数的映射锚点。

几个关键设计：

1. **必须走真实打分路径**：用 Matcher.raw_similarity（即 1:N + top-K 融合）
   来算冒名者分数，而不是"和参考均值比"这种简化写法，否则锚点不在同一条
   曲线上，最终分数会系统性偏高。

2. **"同一人"分布要排除自身**：参考图两两比较时要把样本自己从检索库里剔除，
   否则余弦恒等于 1.0，锚点直接失效（这是实测踩过的坑）。

3. **图库先自检**：用 Matcher.validate_gallery 剔除"其实是别人"的参考照，
   避免合影误检把整个图库带偏。

产出 assets/calibration/calibration.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from facecmp import config
from facecmp.engine import FaceEngine
from facecmp.scoring import Calibrator, Matcher, ReferenceEntry, cosine_to_euclidean


def load_gallery(engine: FaceEngine):
    """加载参考图库。姿态不合格的（侧脸/糊/太小）直接剔除。"""
    entries, rejected = [], []
    for name in sorted(os.listdir(config.REFERENCE_DIR)):
        if name.startswith("_") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = str(config.REFERENCE_DIR / name)
        img = cv2.imread(path)
        if img is None:
            rejected.append((name, "读不出"))
            continue
        face = engine.best_face(img, require_acceptable=True)
        if face is None:
            rejected.append((name, "检测不到人脸"))
            continue
        entry = ReferenceEntry(
            name=name, path=path, embedding=face.embedding,
            image=img, box=face.box, landmarks=face.landmarks,
        )
        if not face.acceptable:
            entry.note = face.hint
            rejected.append((name, f"姿态/画质不合格: {face.hint}"))
        else:
            entries.append(entry)
            print(f"  保留 {name:44s} size={face.size}")
    for name, why in rejected:
        print(f"  丢弃 {name:44s} {why}")
    return entries, rejected


def main() -> int:
    engine = FaceEngine()
    print("== 参考图库")
    entries, rejected = load_gallery(engine)
    if not entries:
        raise SystemExit("没有合格的参考照")

    matcher = Matcher(Calibrator({}))
    matcher.set_references(entries)
    report = matcher.validate_gallery()
    for w in report.warnings:
        print("  ! ", w)
    for name, why in report.rejected:
        print("  ! 自检剔除:", name, why)
    entries = report.entries
    matcher.set_references(entries)
    if len(entries) < 2:
        raise SystemExit("自检后参考照不足 2 张，无法建立'同一人'分布")

    # ---- 同人分布：把样本自身从检索库里排除掉
    same = []
    for a in entries:
        sims = [float(a.embedding @ r.embedding) for r in entries if r.name != a.name]
        if sims:
            same.append(max(sims))
    same_arr = np.array(same)
    print(f"\n== 同一人（排除自身后取最高分）n={len(same)} "
          f"min={same_arr.min():.4f} mean={same_arr.mean():.4f} max={same_arr.max():.4f}")

    # ---- 冒名分布：完全用真实打分路径
    imp = np.load(config.ASSET_DIR / "calibration" / "impostor_embeddings.npy")
    labels_path = config.ASSET_DIR / "calibration" / "impostor_labels.json"
    labels = json.loads(labels_path.read_text("utf-8")) if labels_path.exists() else []
    if len(labels) != len(imp):
        print("   ! 缺少 impostor_labels.json，无法标出是谁（可跑 tools/rebuild_labels.py 重建）")
        labels = ["?"] * len(imp)
    imp_sims = np.array([matcher.raw_similarity(e)[0] for e in imp])
    p50 = float(np.percentile(imp_sims, 50))
    p95 = float(np.percentile(imp_sims, 95))
    print(f"== 冒名（top-{config.REFERENCE_TOP_K} 融合）n={len(imp_sims)} "
          f"min={imp_sims.min():.4f} mean={imp_sims.mean():.4f} p50={p50:.4f} "
          f"p95={p95:.4f} max={imp_sims.max():.4f}")
    top5 = sorted(zip(imp_sims, labels, strict=False), reverse=True)[:5]
    print("   最像的几个:", [(round(float(s), 4), who) for s, who in top5])

    # ---- 可分性
    overlap = float((imp_sims >= same_arr.min()).mean())
    margin = float(same_arr.min() - imp_sims.max())
    print(f"\n   同人下限 {same_arr.min():.4f} / 冒名上限 {imp_sims.max():.4f} "
          f"-> 间隔 {margin:+.4f}")
    print(f"   重叠率 {overlap:.2%}（越低越好，0 表示分布完全不重叠）")

    genuine_p05 = float(np.percentile(same_arr, 5))
    anchors = [
        {"label": "impostor_p50", "cos": p50, "score": 40.0, "note": "比一半陌生面孔更像"},
        {"label": "impostor_p95", "cos": p95, "score": 62.0, "note": "比 95% 陌生面孔更像"},
        {"label": "genuine_p05", "cos": genuine_p05, "score": 88.0,
         "note": "达到本人不同照片的典型水平"},
    ]
    print("\n== 锚点")
    for a in anchors:
        print(f"   {a['label']:14s} cos={a['cos']:.4f} -> {a['score']:.0f} 分  ({a['note']})")

    payload = {
        "version": 3,
        "method": "top-K gallery retrieval; anchors from real impostor population",
        "anchors": anchors,
        "stats": {
            "genuine_n": len(same_arr),
            "genuine_mean": float(same_arr.mean()),
            "genuine_min": float(same_arr.min()),
            "genuine_max": float(same_arr.max()),
            "impostor_n": len(imp_sims),
            "impostor_mean": float(imp_sims.mean()),
            "impostor_p50": p50,
            "impostor_p95": p95,
            "impostor_max": float(imp_sims.max()),
            "overlap_rate": overlap,
            "separation_margin": margin,
        },
        "reference_files": [r.name for r in entries],
        "reference_rejected": [{"name": n, "reason": w} for n, w in rejected],
        "top_k": config.REFERENCE_TOP_K,
    }
    config.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print("\n已写入:", config.CALIBRATION_FILE)

    cal = Calibrator(payload)
    print("\n== 标定自检")
    for label, cos in [
        ("冒名 p50", p50), ("冒名 max", float(imp_sims.max())),
        ("同人 min", float(same_arr.min())), ("同人 mean", float(same_arr.mean())),
    ]:
        print(f"   {label:10s} cos={cos:.4f} -> {cal.to_score(cos):5.1f} 分"
              f"   (等效欧氏距离 {cosine_to_euclidean(cos):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
