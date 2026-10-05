"""用真实摄像头跑一遍完整流程：抓帧 -> 检测 -> 采样 -> 打分。

不启动 GUI，只验证数据链路，附带产出可视化截图。
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from facecmp import config
from facecmp.camera import Camera
from facecmp.engine import FaceEngine
from facecmp.reference import prepare_matcher
from facecmp.scoring import Calibrator, Matcher, cosine_to_euclidean
from facecmp.session import LiveSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    engine = FaceEngine()
    matcher = Matcher(Calibrator.load())
    prepare_matcher(matcher, engine)
    session = LiveSession(engine=engine, matcher=matcher)
    print(f"参考图库 {len(matcher.references)} 张；标定：{matcher.calibrator.summary()}")

    cam = Camera()
    if not cam.start():
        print("摄像头启动失败:", cam.error)
        return 1
    print("摄像头已开启，等待画面…")
    time.sleep(1.5)

    collected = 0
    t_start = time.perf_counter()
    deadline = t_start + 25.0
    last_print = 0.0
    try:
        while time.perf_counter() < deadline and collected < config.MIN_SAMPLES_TO_SCORE:
            frame = cam.read()
            if frame is None:
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            session.tick(frame)
            dt = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()
            if now - last_print > 1.0:
                last_print = now
                print(f"  [{now - t_start:5.1f}s] {session.state.value:14s} "
                      f"samples={session.sample_count} tick={dt:.0f}ms "
                      f"cam={cam.current_fps():.1f}fps hint={session.hint}")
            if session.can_score():
                res = session.score()
                collected += 1
                s = res.score
                print(f"\n  == 第 {collected} 次评分")
                print(f"     分数   {s.score:.1f}")
                print(f"     余弦   {s.raw:.4f}  (欧氏 {cosine_to_euclidean(s.raw):.3f})")
                print(f"     命中   {s.best_reference}")
                print(f"     采样   {s.sample_count} 帧，一致性 {s.stability:.3f}")
                print(f"     评价   {s.verdict} / {s.comment}")
                print(f"     各参考：{[(n[:22], round(v, 4)) for n, v in s.per_reference]}")
                stamp = f"{int(time.time())}"
                cv2.imwrite(os.path.join(OUT, f"live_frame_{stamp}.jpg"), res.frame)
                cv2.imwrite(os.path.join(OUT, f"live_aligned_{stamp}.jpg"), res.preview)
                vis = res.frame.copy()
                if res.face.box:
                    x, y, w, h = res.face.box
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 3)
                    for p in res.face.landmarks:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(OUT, f"live_vis_{stamp}.jpg"), vis)
                session.reset_samples()
    finally:
        cam.stop()

    print(f"\n共完成 {collected} 次评分，截图在 {OUT}")
    return 0 if collected else 1


if __name__ == "__main__":
    raise SystemExit(main())
