"""端到端冒烟测试：不依赖摄像头，验证 引擎 -> 会话 -> 打分 全链路。

用参考图库里的真人照模拟"用户画面"，再用人脸库照片模拟不同的人，
检查分数是否落在预期区间（本人应该明显高于陌生人）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from facecmp import config
from facecmp.engine import FaceEngine
from facecmp.reference import prepare_matcher
from facecmp.scoring import Calibrator, Matcher, cosine_to_euclidean, tier_for
from facecmp.session import LiveSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")


def banner(text: str) -> None:
    print(f"\n{'=' * 62}\n{text}\n{'=' * 62}")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    engine = FaceEngine()
    matcher = Matcher(Calibrator.load())
    print("标定:", matcher.calibrator.summary())

    banner("1) 参考图库加载 + 自检")
    prepare_matcher(matcher, engine, verbose=True)
    print(f"   合格 {len(matcher.references)} 张：")
    for r in matcher.references:
        print(f"     - {r.name}")
    assert matcher.references, "参考图库为空"

    banner("2) 拿参考照自己当'用户'，分数应当很高")
    session = LiveSession(engine=engine, matcher=matcher)
    for ref in matcher.references:
        img = ref.image
        # 显式推进时间戳，否则会撞上 EMBED_INTERVAL_MS 节流，采不满样本
        base = time.perf_counter()
        for i in range(config.LIVE_BUFFER_SIZE + 1):
            session.tick(img, now=base + i * (config.EMBED_INTERVAL_MS / 1000.0))
        ok = session.can_score()
        print(f"   {ref.name[:40]:42s} state={session.state.value:12s} "
              f"samples={session.sample_count} can_score={ok}")
        if ok:
            res = session.score()
            s = res.score
            print(f"      -> 分数 {s.score:5.1f}  余弦 {s.raw:.4f}  "
                  f"欧氏 {cosine_to_euclidean(s.raw):.3f}  {s.verdict}")
            cv2.imwrite(os.path.join(OUT, "selfcheck_aligned.jpg"), res.preview)
            cv2.imwrite(os.path.join(OUT, "selfcheck_frame.jpg"), res.frame)
        session.reset_samples()

    banner("3) 用冒名者样本当'用户'，分数应当明显更低")
    imp = np.load(config.ASSET_DIR / "calibration" / "impostor_embeddings.npy")
    labels_path = config.ASSET_DIR / "calibration" / "impostor_labels.json"
    labels = json.loads(labels_path.read_text("utf-8")) if labels_path.exists() else ["?"] * len(imp)
    scores = [matcher.score([e]).score for e in imp]
    arr = np.array(scores)
    print(f"   n={len(arr)} min={arr.min():.1f} mean={arr.mean():.1f} max={arr.max():.1f}")
    order = np.argsort(arr)[::-1]
    for i in order[:5]:
        print(f"     最高: {arr[i]:5.1f}  {labels[i]}")
    for i in order[-3:]:
        print(f"     最低: {arr[i]:5.1f}  {labels[i]}")

    banner("4) 分档与文案映射")
    for sc in (95, 80, 65, 50, 35, 20):
        tier, comment = tier_for(sc)
        print(f"   {sc:3d} 分 -> {tier} / {comment}")

    banner("5) 交互：劣化画面下的状态机")
    session2 = LiveSession(engine=engine, matcher=matcher)
    base = matcher.references[0].image
    cases = {
        "正常": base,
        "大幅压暗": (base * 0.18).astype(np.uint8),
        "强模糊": cv2.GaussianBlur(base, (0, 0), 9),
        "空画面": np.full_like(base, 128),
    }
    for name, img in cases.items():
        session2.reset_samples()
        # 时间戳每轮重置，否则会撞上节流
        session2.tick(img, now=time.perf_counter())
        print(f"   {name:8s} -> state={session2.state.value:12s} hint={session2.hint}")

    banner("结果")
    ok = True
    if matcher.references:
        session.reset_samples()
        base = time.perf_counter()
        for i in range(config.LIVE_BUFFER_SIZE + 1):
            session.tick(
                matcher.references[0].image,
                now=base + i * (config.EMBED_INTERVAL_MS / 1000.0),
            )
        if session.can_score():
            own = session.score().score.score
            print(f"   参考照自比 {own:.1f} 分，陌生人最高 {arr.max():.1f} 分")
            ok = own > arr.max() + 5
            print("   自评分显著高于陌生人:", ok)
        else:
            print("   ! 参考照自比无法评分")
            ok = False
    print("\n全部通过" if ok else "\n存在问题")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
