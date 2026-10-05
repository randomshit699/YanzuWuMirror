"""回归测试：session 暴露的框 / 关键点必须真的是"显示帧坐标"。

背景（真实踩过的 bug）：
  检测跑在降采样分析帧上，而 session 早期把 boxes 留在分析帧坐标、
  face.box 留在原始帧坐标。界面照着"分析帧坐标"再乘一次缩放比，
  于是框被重复缩放，和画面里的人脸对不上（960 宽的摄像头配
  ANALYSIS_WIDTH=640 时 kx=1.5，框会大 1.5 倍并且错位）。

验证方法（不靠肉眼看截图）：
  拿 session 给出的显示帧坐标 face.box 去裁"显示帧"，
  再在这个裁剪块上跑一次关键点检测，得到的 68 点应当和
  face.landmarks 基本重合。同理 boxes 里应当也有一项和 face.box 吻合。
  只要坐标系错了（多乘或少乘了缩放比），这两条断言就会失败。
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from smoke_gui import make_camera_like_frame

from facecmp import config
from facecmp.engine import FaceEngine
from facecmp.reference import prepare_matcher
from facecmp.scoring import Calibrator, Matcher
from facecmp.session import LiveSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")
os.makedirs(OUT, exist_ok=True)


def main() -> int:
    engine = FaceEngine()
    matcher = Matcher(Calibrator.load())
    prepare_matcher(matcher, engine)
    ref = matcher.references[0]

    # 造一个比 ANALYSIS_WIDTH 大的"摄像头帧"，确保真的走了降采样分支
    frame = make_camera_like_frame(ref.image, ref.box, 960, 720)
    assert frame.shape[1] > config.ANALYSIS_WIDTH, "这个用例必须触发降采样才有意义"
    print(f"显示帧 {frame.shape[1]}x{frame.shape[0]}  "
          f"分析宽度 {config.ANALYSIS_WIDTH}  (kx={frame.shape[1] / config.ANALYSIS_WIDTH:.2f})")

    session = LiveSession(engine=engine, matcher=matcher)
    session.tick(frame, now=time.perf_counter())

    assert session.face is not None, "没有检测到人脸"
    face = session.face
    x, y, w, h = face.box
    print(f"face.box (显示帧坐标) = {face.box}  size={face.size}")

    ok = True

    # ---- 1) 框必须在显示帧范围内
    fh, fw = frame.shape[:2]
    in_bounds = (0 <= x < fw) and (0 <= y < fh) and (x + w <= fw + 1) and (y + h <= fh + 1)
    print(f"\n1) 框在显示帧内: {in_bounds}  (显示帧 {fw}x{fh})")
    ok = ok and in_bounds

    # ---- 2) 独立重算一遍坐标映射，和 session 的结果对比
    # 这里刻意自己写一遍裁剪+映射的数学（不调用 FaceCrop.to_source），
    # 只要 session 里的 scale 取反、写乘除搞错、offset 漏加，两边就会不一致。
    print("\n2) 独立复算裁剪映射，与 session.face.landmarks 对比")
    margin, target = config.FACE_CROP_MARGIN, config.FACE_ANALYSIS_SIZE
    sw, sh = w * (1 + 2 * margin), h * (1 + 2 * margin)
    cx, cy = x + w / 2, y + h / 2
    ix0 = round(min(max(0.0, cx - sw / 2), max(0.0, fw - sw)))
    iy0 = round(min(max(0.0, cy - sh / 2), max(0.0, fh - sh)))
    ix1 = round(min(ix0 + sw, fw))
    iy1 = round(min(iy0 + sh, fh))
    sub = frame[iy0:iy1, ix0:ix1]
    sc = target / min(sub.shape[0], sub.shape[1])
    if abs(sc - 1.0) > 0.05:
        import cv2 as _cv2

        sub = _cv2.resize(
            sub,
            (round(sub.shape[1] * sc), round(sub.shape[0] * sc)),
            interpolation=_cv2.INTER_CUBIC if sc > 1 else _cv2.INTER_AREA,
        )
    sub = np.ascontiguousarray(sub)
    ibox = (round((x - ix0) * sc), round((y - iy0) * sc),
            round(w * sc), round(h * sc))
    print(f"   复算: origin=({ix0},{iy0}) scale={sc:.4f} box={ibox} sub={sub.shape}")
    ref_face = engine.analyze(sub, ibox)
    if ref_face is None:
        print("   ! 复算时提关键点失败")
        return 1
    # 裁剪块坐标 -> 显示帧坐标：除以 scale 再加 offset
    expect = ref_face.landmarks / sc
    expect[:, 0] += ix0
    expect[:, 1] += iy0
    d = np.linalg.norm(expect - face.landmarks, axis=1)
    max_err, mean_err = float(d.max()), float(d.mean())
    # 阈值 8px：本测试用 face.box（已取整）复算，而 session 用的是未取整的
    # rbox，所以有 1~5px 的正常差异。真正把缩放取反 / offset 漏加的 bug
    # 会产生 100~500px 的偏差，这个阈值远小于它、又不会被取整误差误伤。
    good = max_err < 8.0
    print(f"   最大偏差 {max_err:.2f}px / 平均 {mean_err:.2f}px  阈值 8px  "
          f"{'OK' if good else 'FAIL'}")
    if not good:
        i = int(d.argmax())
        print(f"   偏差最大第 {i} 点: session {np.round(face.landmarks[i], 1)} "
              f"vs 复算 {np.round(expect[i], 1)}")
    ok = ok and good

    # ---- 3) 关键点的几何关系必须成立（与坐标系无关的独立校验）
    print("\n3) 关键点几何合理性")
    eye_r = face.landmarks[36:42].mean(axis=0)
    eye_l = face.landmarks[42:48].mean(axis=0)
    nose = face.landmarks[30]
    mouth = face.landmarks[48:55].mean(axis=0)
    chin = face.landmarks[8]
    eye_dist = float(np.linalg.norm(eye_l - eye_r))
    checks = {
        "双眼在鼻尖上方": eye_r[1] < nose[1] and eye_l[1] < nose[1],
        "鼻尖在嘴上方": nose[1] < mouth[1],
        "嘴在下巴上方": mouth[1] < chin[1],
        "眼距合理(0.2~0.6脸宽)": 0.20 * w < eye_dist < 0.60 * w,
        "关键点都在框附近": bool(
            np.all((face.landmarks[:, 0] > x - 0.3 * w)
                   & (face.landmarks[:, 0] < x + 1.3 * w)
                   & (face.landmarks[:, 1] > y - 0.3 * h)
                   & (face.landmarks[:, 1] < y + 1.3 * h))
        ),
    }
    for k, v in checks.items():
        print(f"   {k:24s} {v}")
        ok = ok and v
    print(f"   (眼距 {eye_dist:.0f}px / 脸宽 {w}px = {eye_dist / w:.2f})")

    # ---- 4) session.boxes 必须和 face.box 在同一套坐标系
    if session.boxes:
        ious = []
        for rbox, _ in session.boxes:
            bx, by, bw, bh = rbox
            ix = max(0, min(x + w, bx + bw) - max(x, bx))
            iy = max(0, min(y + h, by + bh) - max(y, by))
            inter = ix * iy
            union = w * h + bw * bh - inter
            ious.append(inter / union if union > 0 else 0.0)
        best_iou = max(ious)
        good = best_iou > 0.90
        print(f"\n4) boxes 与 face.box 最大 IoU = {best_iou:.3f}  "
              f"{'OK' if good else 'FAIL(坐标系不一致)'}")
        ok = ok and good
    else:
        print("\n4) session.boxes 为空，跳过")

    # ---- 5) 可视化，方便人工复核
    vis = frame.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 3)
    for p in face.landmarks[[30, 36, 39, 42, 45, 48, 54]]:
        cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
    for rbox, _ in session.boxes[:4]:
        bx, by, bw, bh = (int(v) for v in rbox)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)
    cv2.imwrite(os.path.join(OUT, "coords_check.png"), vis)
    print(f"\n可视化已保存: {OUT}/coords_check.png")

    print("\n全部通过" if ok else "\n存在失败用例")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
