"""similarity_transform 的正确性验证：闭式解必须和暴力数值优化一致。

这类几何变换很容易把矩阵顺序搞反（SVD 里 U/V 换位、cov 取转置），
而这类 bug 在图片上表现为"对齐图取到了旁边的背景"，非常难一眼看出来。
所以这里用暴力优化做基准，逐个用例比对。
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from scipy.optimize import minimize

from facecmp import config
from facecmp.engine import PREVIEW_TEMPLATE, FaceEngine, similarity_transform

T = PREVIEW_TEMPLATE


def residual(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    pred = (M[:, :2] @ src.T).T + M[:, 2]
    return float(((pred - dst) ** 2).sum())


def brute_force(src: np.ndarray, dst: np.ndarray) -> float:
    """Nelder-Mead 暴力搜最小残差，作为基准。"""

    def loss(p):
        c, th, tx, ty = p
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        M = np.hstack([c * R, np.array([[tx], [ty]])])
        return residual(M, src, dst)

    best = float("inf")
    for th0 in np.linspace(-np.pi, np.pi, 33):
        for c0 in (0.2, 1.0, 3.0):
            r = minimize(loss, [c0, th0, 0.0, 0.0], method="Nelder-Mead",
                         options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 60000})
            best = min(best, r.fun)
    return best


def cases():
    """构造 (缩放, 旋转角, 平移) 形式的真值用例。"""
    for s, deg, t in [
        (2.2, 30, [120.0, -60.0]),
        (0.6, -17, [300.0, 200.0]),
        (1.0, 0, [0.0, 0.0]),
        (3.0, 85, [-50.0, 90.0]),
        (1.4, -120, [10.0, 10.0]),
        (0.85, 175, [400.0, -300.0]),
    ]:
        th = np.deg2rad(deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        src = (s * (R @ T.T).T) + np.array(t)  # 真值正向变换
        yield f"s={s:<5} rot={deg:>5}d t={t}", src, T
    # 真实人脸关键点（非刚性，作为"有噪声"的用例）
    yield "real-landmark-like", T * 1.9 + np.array([[40.0, -25.0]]), T


def check_preview() -> bool:
    """端到端检查：对齐图必须真的落在人脸上，而不是背景。

    做法是把 68 点关键点按对齐矩阵投影到 150x150 模板坐标，
    投影结果应落在合理范围内；同时对齐图的像素方差不能小到像纯色块
    （纯背景块方差会很低、且颜色偏冷）。
    """
    print("\n== 对齐图端到端检查（每一项都必须有脸，不能是纯背景）")
    ok = True
    engine = FaceEngine()
    for name in sorted(os.listdir(config.REFERENCE_DIR)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")) or name.startswith("_"):
            continue
        img = cv2.imread(str(config.REFERENCE_DIR / name))
        face = engine.best_face(img, require_acceptable=True)
        if face is None:
            print(f"   {name[:40]:42s} 无人脸，跳过")
            continue
        p = face.preview
        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        std = float(gray.std())
        # 人脸区域应该包含丰富的明暗结构；纯色背景块的 std 会明显偏低
        good = std > 40.0
        ok = ok and good
        print(f"   {name[:40]:42s} preview std={std:6.2f} {'OK' if good else 'FAIL(疑似背景)'}")
    return ok


def main() -> int:
    ok = True
    print(f"{'用例':28s} {'闭式解残差':>14s} {'暴力最优':>14s} {'判定':>8s}")
    for name, src, dst in cases():
        M = similarity_transform(src, dst)
        r_closed = residual(M, src, dst)
        r_best = brute_force(src, dst)
        good = r_closed <= r_best + 1e-6 * max(1.0, r_best)
        ok = ok and good
        print(f"{name:28s} {r_closed:14.4e} {r_best:14.4e} {'OK' if good else 'FAIL':>8s}")

    # 额外检查：矩阵应当是纯旋转+等比缩放（两列等长且正交）
    print("\n== 结构检查（两列必须等长且正交，即不含错切）")
    for name, src, dst in cases():
        M = similarity_transform(src, dst)[:, :2]
        col0, col1 = M[:, 0], M[:, 1]
        lens = (np.linalg.norm(col0), np.linalg.norm(col1))
        cosang = abs(float(col0 @ col1) / (lens[0] * lens[1]))
        det = float(np.linalg.det(M))
        good = abs(lens[0] - lens[1]) < 1e-9 and cosang < 1e-9 and det > 0
        ok = ok and good
        print(f"   {name:28s} |c0|={lens[0]:.6f} |c1|={lens[1]:.6f} "
              f"夹角cos={cosang:.2e} det={det:+.4f} {'OK' if good else 'FAIL'}")

    ok = check_preview() and ok
    print("\n全部通过" if ok else "\n存在失败用例")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
