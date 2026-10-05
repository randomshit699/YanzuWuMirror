"""GUI 冒烟测试：不依赖真实摄像头画面，用参考照冒充摄像头帧喂给界面。

目的是把 界面构建 -> 绘制 -> 按钮回调 -> 评分展示 这条链路全部跑一遍，
并输出界面截图。
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk

import cv2
import numpy as np

from facecmp import config
from facecmp.app import FaceCompareApp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")
os.makedirs(OUT, exist_ok=True)

steps: list[str] = []


def make_camera_like_frame(image, box, out_w: int = 640, out_h: int = 480) -> np.ndarray:
    """裁出人脸附近区域再缩放到摄像头分辨率，模拟"人坐在镜头前"的画面。

    保持 4:3 宽高比，并且不让裁剪框超出原图范围，否则 HOG 会因为
    画面被拉伸变形而检不出人脸。
    """
    x, y, w, h = box
    ih, iw = image.shape[:2]
    target_h = int(max(w, h) * 1.9)
    target_w = int(target_h * out_w / out_h)
    if target_w > iw:
        target_w = iw
        target_h = int(target_w * out_h / out_w)
    if target_h > ih:
        target_h = ih
        target_w = int(target_h * out_w / out_h)
    cx, cy = x + w // 2, y + h // 2
    x0 = min(max(0, cx - target_w // 2), iw - target_w)
    y0 = min(max(0, cy - target_h // 2), ih - target_h)
    crop = image[y0:y0 + target_h, x0:x0 + target_w]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


class FakeCamera:
    """冒充 Camera，让 GUI 走真实的 _tick 路径而不用真的摄像头。"""

    def __init__(self, frame) -> None:
        self.frame = frame
        self.is_running = True
        self.error = ""
        self.fps = 30.0

    def start(self) -> bool:
        self.is_running = True
        return True

    def stop(self) -> None:
        self.is_running = False

    def read(self):
        return self.frame.copy() if self.is_running else None

    def current_fps(self) -> float:
        return self.fps


def run() -> int:
    root = tk.Tk()
    app = FaceCompareApp(root)
    root.update()

    def step(msg):
        steps.append(msg)
        print("  *", msg)

    # 1) 界面元素
    print("1) 界面元素")
    for name in ("video_canvas", "ref_canvas", "user_canvas", "progress", "btn_toggle", "btn_score"):
        widget = getattr(app, name, None)
        assert widget is not None, f"缺少控件 {name}"
        step(f"控件 {name} 就绪")
    root.update()
    step(f"参考图已绘出 {len(app.matcher.references)} 张")
    assert len(app.matcher.references) > 0

    # 2) 打开"摄像头"（用假帧注入，走真实 _tick 路径）
    print("\n2) 假摄像头帧注入")
    ref = app.matcher.references[0]
    frame = make_camera_like_frame(ref.image, ref.box)
    print(f"   假帧 {frame.shape}，原人脸框 {ref.box}")
    app.camera = FakeCamera(frame)
    app.running = True
    app.btn_toggle.configure(text="停止检测")

    # 反复调用真实 tick，让样本池攒够
    base = time.perf_counter()
    for i in range(40):
        app.session.tick(frame, now=base + i * (config.EMBED_INTERVAL_MS / 1000.0))
        if app.session.can_score():
            break
    app._tick()  # 走真实刷新路径（画框、按钮状态）
    root.update()
    step(f"state={app.session.state.value} samples={app.session.sample_count} "
         f"can_score={app.session.can_score()}")
    assert app.session.can_score(), "应当可以评分"
    assert str(app.btn_score.cget("state")) == "normal", (
        f"评分按钮应可用，实际 {app.btn_score.cget('state')}"
    )
    step("评分按钮已启用")

    # 3) 点评分
    print("\n3) 触发评分")
    app._do_score()
    root.update()
    score_text = app.score_var.get()
    verdict = app.verdict_var.get()
    detail = app.detail_var.get()
    step(f"分数={score_text} 评价={verdict}")
    print("   详情:", detail.replace("\n", " | "))
    print("   状态栏:", app.status_var.get())
    assert score_text not in ("", "--"), "分数未显示"
    assert app.session.last_result is not None, "没有产生评分结果"
    step("评分结果已生成")

    # 4) 截图
    print("\n4) 界面截图")
    root.update_idletasks()
    root.update()
    time.sleep(0.4)
    root.update()
    shot = os.path.join(OUT, "gui_screenshot.png")
    try:
        import pyautogui

        # 必须把窗口提到最前，否则截到的是压在上面的别的窗口
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        root.update()
        time.sleep(0.8)
        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        pyautogui.screenshot(shot, region=(x, y, w, h))
        root.attributes("-topmost", False)
        step(f"截图已保存 {shot} ({w}x{h})")
    except Exception as exc:
        print("   截图失败(不影响结论):", exc)

    # 5) 关闭摄像头按钮路径
    print("\n5) 关闭流程")
    app._toggle_camera()
    root.update()
    step("摄像头已关闭")
    app._on_close()
    step("窗口已销毁")

    print(f"\n共 {len(steps)} 步，全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
