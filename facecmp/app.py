"""Tkinter 图形界面：左边摄像头实时画面，右边吴彦祖参考照，下方相似度。

布局刻意做成"左看自己、右看目标"的对照式，评分用大号数字 + 进度条呈现。
"""

from __future__ import annotations

import math
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import config
from .camera import Camera
from .engine import FaceEngine
from .reference import prepare_matcher
from .scoring import Calibrator, Matcher
from .session import LiveSession, State

APP_TITLE = "吴彦祖相似度检测"
BG = "#14161c"
PANEL = "#1d2029"
FG = "#e8eaf0"
DIM = "#8b93a7"
ACCENT = "#4da3ff"
GOOD = "#3ddc84"
WARN = "#ffb020"
BAD = "#ff5c5c"


def cv_to_photo(image_bgr: np.ndarray, width: int, height: int, mirror: bool = False):
    """把 OpenCV 图像缩放并转成 PIL.Image（保持宽高比，居中留黑边）。

    注意用 ``Image.resize`` 而不是 ``thumbnail``：后者只会缩小不会放大，
    分析帧（480px）贴到 670px 宽的面板上会原样显示，画面看起来偏小。
    """
    if mirror:
        image_bgr = cv2.flip(image_bgr, 1)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    src_w, src_h = pil.size
    if src_w > 0 and src_h > 0:
        scale = min(width / src_w, height / src_h)
        pil = pil.resize(
            (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
            Image.LANCZOS,
        )
    canvas = Image.new("RGB", (width, height), (18, 20, 26))
    canvas.paste(pil, ((width - pil.width) // 2, (height - pil.height) // 2))
    return canvas


def crop_around_face(image_bgr: np.ndarray, box, margin: float = 1.9):
    """以人脸为中心裁一块区域，让参考图缩略图大小统一、也能看清脸。"""
    if box is None:
        return image_bgr
    x, y, w, h = box
    ih, iw = image_bgr.shape[:2]
    side = int(max(w, h) * margin)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(min(max(0, cx - side / 2), max(0, iw - side)))
    y0 = int(min(max(0, cy - side / 2), max(0, ih - side)))
    x1 = int(min(x0 + side, iw))
    y1 = int(min(y0 + side, ih))
    if x1 <= x0 or y1 <= y0:
        return image_bgr
    return image_bgr[y0:y1, x0:x1]


def fit_size(image_bgr: np.ndarray, box_w: int, box_h: int):
    h, w = image_bgr.shape[:2]
    scale = min(box_w / w, box_h / h)
    return max(1, int(w * scale)), max(1, int(h * scale))


class FaceCompareApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG)
        self.root.geometry("1280x860")
        self.root.minsize(1120, 700)

        self.camera = Camera()
        self.running = False
        self._closing = False
        self._score_busy = False
        self._photo_ids: dict[str, object] = {}

        print("正在加载 dlib 模型（首次约需几秒）…", file=sys.stderr)
        self.engine = FaceEngine()
        self.matcher = Matcher(Calibrator.load())
        self.session = LiveSession(engine=self.engine, matcher=self.matcher)
        self.reference_dir = str(config.REFERENCE_DIR)

        self._build_style()
        self._build_ui()
        self._load_references(initial=True)
        # 画布尺寸只有窗口显示出来之后才有效，所以首次布局完成再画一次
        self.root.after(120, self._redraw_static_panels)
        self.root.bind("<Configure>", self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._loop)

    def _redraw_static_panels(self) -> None:
        """窗口布局稳定后重绘静态面板（含占位提示）。"""
        if self._closing:
            return
        self._draw_reference()
        if self.session.last_result is not None:
            self._draw_user_preview(self.session.last_result.preview, "对齐后的人脸")
        else:
            self._draw_placeholder(self.user_canvas, "评分后这里会显示\n对齐后的人脸")
        if not self.running:
            self._draw_placeholder(self.video_canvas, "点击左下角「开始检测」\n调用摄像头")

    def _on_resize(self, _event=None) -> None:
        # 防抖：只在尺寸真正变化时重绘，避免拖拽窗口时反复刷新
        size = (self.root.winfo_width(), self.root.winfo_height())
        if size == getattr(self, "_last_size", None):
            return
        self._last_size = size
        if not self._closing:
            self._draw_reference()

    # ================================================================ 界面
    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Microsoft YaHei UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=FG,
                        font=("Microsoft YaHei UI", 10))
        style.configure("Dim.TLabel", background=BG, foreground=DIM,
                        font=("Microsoft YaHei UI", 9))
        style.configure("DimPanel.TLabel", background=PANEL, foreground=DIM,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=FG,
                        font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Big.TLabel", background=BG, foreground=FG,
                        font=("Microsoft YaHei UI", 44, "bold"))
        style.configure("Verdict.TLabel", background=BG, foreground=FG,
                        font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(14, 8))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=(18, 10))
        style.configure("TProgressbar", troughcolor="#262a36", background=ACCENT,
                        bordercolor="#262a36", lightcolor=ACCENT, darkcolor=ACCENT,
                        thickness=14)
        style.configure("Ready.Horizontal.TProgressbar", troughcolor="#262a36",
                        background=GOOD, lightcolor=GOOD, darkcolor=GOOD,
                        bordercolor="#262a36", thickness=14)
        style.configure("Warn.Horizontal.TProgressbar", troughcolor="#262a36",
                        background=WARN, lightcolor=WARN, darkcolor=WARN,
                        bordercolor="#262a36", thickness=14)
        style.configure("Bad.Horizontal.TProgressbar", troughcolor="#262a36",
                        background=BAD, lightcolor=BAD, darkcolor=BAD,
                        bordercolor="#262a36", thickness=14)
        style.configure("Sample.Horizontal.TProgressbar", troughcolor="#262a36",
                        background=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                        bordercolor="#262a36", thickness=10)

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=18, pady=(14, 8))
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        self.header_info = ttk.Label(header, text="", style="Dim.TLabel")
        self.header_info.pack(side="right")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        body.columnconfigure(0, weight=3, uniform="col")
        body.columnconfigure(1, weight=2, uniform="col")
        body.rowconfigure(0, weight=1)

        # ---------------- 左侧：实时画面
        left = ttk.Frame(body, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        head = ttk.Frame(left, style="Panel.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(head, text="你的实时画面", style="Panel.TLabel").pack(side="left")
        self.cam_state = ttk.Label(head, text="摄像头未开启", style="DimPanel.TLabel")
        self.cam_state.pack(side="right")

        self.video_canvas = tk.Canvas(left, bg="#0c0e13", highlightthickness=0)
        self.video_canvas.grid(row=1, column=0, sticky="nsew")

        self.hint_var = tk.StringVar(value="点击「开始检测」使用摄像头")
        ttk.Label(left, textvariable=self.hint_var, style="Panel.TLabel",
                  wraplength=520).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        # ---------------- 右侧：参考照
        right = ttk.Frame(body, style="Panel.TFrame", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        # 参考照条按内容高度固定，多余空间给下方的"你的抓拍"
        right.rowconfigure(1, weight=0)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        rhead = ttk.Frame(right, style="Panel.TFrame")
        rhead.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(rhead, text="参考目标：吴彦祖", style="Panel.TLabel").pack(side="left")
        self.ref_count = ttk.Label(rhead, text="", style="DimPanel.TLabel")
        self.ref_count.pack(side="right")

        self.ref_canvas = tk.Canvas(right, bg="#0c0e13", highlightthickness=0, height=300)
        self.ref_canvas.grid(row=1, column=0, sticky="nsew")

        self.user_head = ttk.Frame(right, style="Panel.TFrame")
        self.user_head.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        ttk.Label(self.user_head, text="你的抓拍", style="Panel.TLabel").pack(side="left")
        self.user_tag = ttk.Label(self.user_head, text="等待评分", style="DimPanel.TLabel")
        self.user_tag.pack(side="right")

        self.user_canvas = tk.Canvas(right, bg="#0c0e13", highlightthickness=0, height=200)
        self.user_canvas.grid(row=3, column=0, sticky="nsew")

        # ---------------- 底部：分数 + 按钮
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=18, pady=(0, 12))

        score_row = ttk.Frame(bottom)
        score_row.pack(fill="x")
        score_left = ttk.Frame(score_row)
        score_left.pack(side="left", padx=(0, 36))
        head_row = ttk.Frame(score_left)
        head_row.pack(anchor="w")
        ttk.Label(head_row, text="相似度", style="Dim.TLabel").pack(side="left", padx=(0, 10))
        self.score_var = tk.StringVar(value="--")
        ttk.Label(head_row, textvariable=self.score_var, style="Big.TLabel").pack(side="left")
        self.verdict_var = tk.StringVar(value="还没开始")
        ttk.Label(score_left, textvariable=self.verdict_var, style="Verdict.TLabel").pack(anchor="w")
        self.comment_var = tk.StringVar(value="")
        ttk.Label(score_left, textvariable=self.comment_var, style="Dim.TLabel").pack(anchor="w")

        detail = ttk.Frame(score_row)
        detail.pack(side="left", fill="x", expand=True)

        # 相似度条：常驻显示上一次评分结果
        self.score_bar = ttk.Progressbar(detail, maximum=100, style="Ready.Horizontal.TProgressbar")
        self.score_bar.pack(fill="x", pady=(4, 4))
        self.detail_var = tk.StringVar(value="")
        ttk.Label(detail, textvariable=self.detail_var, style="Dim.TLabel").pack(anchor="w")

        # 采样条：反映实时采样进度，与上面的分数条互不干扰
        sample_row = ttk.Frame(detail)
        sample_row.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(sample_row, maximum=config.LIVE_BUFFER_SIZE,
                                        style="Sample.Horizontal.TProgressbar",
                                        length=260)
        self.progress.pack(side="left", fill="x", expand=True)
        self.sample_var = tk.StringVar(value="")
        ttk.Label(sample_row, textvariable=self.sample_var, style="Dim.TLabel").pack(
            side="left", padx=(12, 0)
        )

        btns = ttk.Frame(bottom)
        btns.pack(fill="x", pady=(12, 0))
        self.btn_toggle = ttk.Button(btns, text="开始检测", command=self._toggle_camera,
                                     style="Accent.TButton")
        self.btn_toggle.pack(side="left")
        self.btn_score = ttk.Button(btns, text="立即评分", command=self._do_score, state="disabled")
        self.btn_score.pack(side="left", padx=8)
        ttk.Button(btns, text="更换参考图", command=self._change_reference).pack(side="left")
        ttk.Button(btns, text="重新加载参考图", command=lambda: self._load_references()).pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="")
        ttk.Label(btns, textvariable=self.status_var, style="Dim.TLabel").pack(side="right")

    # ================================================================ 参考图
    def _load_references(self, initial: bool = False) -> None:
        report = prepare_matcher(self.matcher, self.engine, self.reference_dir)
        n = len(self.matcher.references)
        self.ref_count.configure(text=f"{n} 张")

        if n == 0:
            self.status_var.set("参考图库为空，请点「更换参考图」选择照片")
            self.session.set_state(State.NO_CAMERA, "参考图库为空")
            return

        self._draw_reference()
        for name, why in report.rejected:
            print(f"  剔除 {name}: {why}")
        for w in report.warnings:
            print(f"  ! {w}")
        if not report.ok:
            self.status_var.set("；".join(report.warnings) or "参考图库存在异常")
        elif initial:
            self.status_var.set(self.matcher.calibrator.summary())

    def _draw_reference(self) -> None:
        self._clear(self.ref_canvas)
        refs = self.matcher.references
        if not refs:
            self.ref_canvas.create_text(
                max((self.ref_canvas.winfo_width() or 360) / 2, 60),
                max((self.ref_canvas.winfo_height() or 300) / 2, 40),
                text="参考图库为空\n点「更换参考图」选择吴彦祖的照片目录",
                fill=DIM, font=("Microsoft YaHei UI", 11), justify="center",
            )
            return
        w, h = self.ref_canvas.winfo_width() or 360, self.ref_canvas.winfo_height() or 300
        if w <= 1 or h <= 1:
            return
        # 横向排布所有参考照；统一裁成人脸附近，缩略图大小才一致
        cell_w = max(60, (w - 8 * (len(refs) - 1)) // len(refs))
        cell_h = max(60, min(h - 30, int(cell_w * 4 / 3)))
        # 画布高度贴合内容，避免右侧下方留出大片空白
        self.ref_canvas.configure(height=cell_h + 26)
        for i, ref in enumerate(refs):
            x = i * (cell_w + 8)
            thumb = crop_around_face(ref.image, ref.box, margin=1.8)
            photo = ImageTk.PhotoImage(cv_to_photo(thumb, cell_w, cell_h))
            self._photo_ids[f"ref{i}"] = photo
            self.ref_canvas.create_image(x, 0, image=photo, anchor="nw")
            self.ref_canvas.create_rectangle(
                x, 0, x + cell_w - 1, cell_h - 1, outline="#2b3040"
            )
            self.ref_canvas.create_text(
                x + cell_w // 2, cell_h + 10,
                text=os.path.splitext(ref.name)[0][:16],
                fill=DIM, font=("Microsoft YaHei UI", 8),
            )
        self.ref_canvas.create_rectangle(
            0, 0, w, cell_h + 30, outline="#2b3040",
        )

    def _change_reference(self) -> None:
        path = filedialog.askdirectory(
            title="选择参考图目录（可放多张吴彦祖的照片）",
            initialdir=self.reference_dir if os.path.isdir(self.reference_dir) else None,
        )
        if not path:
            return
        self.reference_dir = path
        self._load_references()

    # ================================================================ 主循环
    def _toggle_camera(self) -> None:
        if self.running:
            self.running = False
            self.camera.stop()
            self.session.reset_samples()
            self.btn_toggle.configure(text="开始检测")
            self.cam_state.configure(text="摄像头已关闭")
            self.btn_score.configure(state="disabled")
            self._clear(self.video_canvas)
            self._draw_placeholder(self.video_canvas, "摄像头已关闭")
            self.hint_var.set("点击「开始检测」重新开启摄像头")
            return

        if not self.matcher.references:
            self.status_var.set("请先加载参考图")
            return
        if not self.camera.start():
            self.status_var.set(self.camera.error)
            self.hint_var.set(self.camera.error)
            return
        self.running = True
        self.session.reset_samples()
        self.btn_toggle.configure(text="停止检测")
        self.cam_state.configure(text="摄像头运行中")
        self.status_var.set("摄像头已开启")
        self.hint_var.set("寻找人脸…")

    def _loop(self) -> None:
        if self._closing:
            return
        try:
            self._tick()
        except Exception as exc:  # 保证界面不会因为一次异常就死掉
            self.status_var.set(f"内部错误：{exc}")
        finally:
            if not self._closing:
                self.root.after(33, self._loop)

    def _tick(self) -> None:
        if not self.running:
            return
        frame = self.camera.read()
        if frame is None:
            self.hint_var.set("等待画面…")
            return

        self.session.tick(frame)
        # 预览用原始分辨率（更清晰）；session 暴露的框和关键点都是原始帧坐标，
        # 与这里传进去的 display_frame 同一套坐标系
        self._draw_video(frame)

        state = self.session.state
        self.hint_var.set(self.session.hint or state.value)
        self.cam_state.configure(text=f"{state.value} · 采集 {self.camera.current_fps():.0f}fps")

        n = self.session.sample_count
        self.sample_var.set(
            f"已采集 {n}/{config.LIVE_BUFFER_SIZE} 帧"
            f"（攒够 {config.MIN_SAMPLES_TO_SCORE} 帧即可评分）"
        )
        self.progress["value"] = n

        if self.session.can_score() and not self._score_busy:
            self.btn_score.configure(state="normal")
        elif not self._score_busy:
            self.btn_score.configure(state="disabled")

    def _draw_video(self, display_frame: np.ndarray) -> None:
        self._clear(self.video_canvas)
        w = self.video_canvas.winfo_width() or 640
        h = self.video_canvas.winfo_height() or 480
        if w <= 1 or h <= 1:
            return
        show = cv_to_photo(display_frame, w, h, mirror=True)
        photo = ImageTk.PhotoImage(show)
        self._photo_ids["video"] = photo
        self.video_canvas.create_image(0, 0, image=photo, anchor="nw")

        # 图像在画布里居中，上下/左右可能有黑边，坐标系要跟着这个偏移走
        fh, fw = display_frame.shape[:2]
        sx = show.width / fw
        sy = show.height / fh
        ox = (w - show.width) // 2
        oy = (h - show.height) // 2

        def to_canvas(x, y, bw=0.0, bh=0.0):
            """显示帧坐标 -> 画布坐标。

            session 暴露的 face.box / face.landmarks / boxes 已经全部是
            显示帧（= 原始帧）坐标，这里只做"缩放 + 居中偏移 + 左右镜像"，
            不要再乘任何分析帧的系数，否则会重复缩放。
            """
            return (
                ox + (fw - (x + bw)) * sx, oy + y * sy,
                ox + (fw - x) * sx, oy + (y + bh) * sy,
            )

        face = self.session.face
        boxes = self.session.boxes
        if face is not None and face.box:
            x, y, bw, bh = face.box
            color = GOOD if face.acceptable else WARN
            x0, y0, x1, y1 = to_canvas(x, y, bw, bh)
            self.video_canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=3)
            self.video_canvas.create_text(
                x0 + 4, y0 - 14, text=f"{min(bw, bh)}px", fill=color,
                font=("Consolas", 9, "bold"),
            )
            # 画 5 个关键点，直观展示对齐依据
            pts = [
                face.landmarks[30],                       # 鼻尖
                face.landmarks[36:42].mean(axis=0),      # 右眼
                face.landmarks[42:48].mean(axis=0),      # 左眼
                face.landmarks[48],                       # 右嘴角
                face.landmarks[54],                       # 左嘴角
            ]
            for p in pts:
                px, py = to_canvas(p[0], p[1])[:2]
                self.video_canvas.create_oval(px - 3, py - 3, px + 3, py + 3,
                                              outline="#ffffff", width=1)
        elif boxes:
            for rbox, _score in boxes[:4]:
                self.video_canvas.create_rectangle(
                    *to_canvas(*rbox),
                    outline=DIM, width=1, dash=(3, 3),
                )

    def _draw_placeholder(self, canvas: tk.Canvas, text: str) -> None:
        self._clear(canvas)
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 300
        canvas.create_text(w / 2, h / 2, text=text, fill=DIM, justify="center",
                           font=("Microsoft YaHei UI", 12))

    def _draw_user_preview(self, preview_bgr: np.ndarray, tag: str) -> None:
        self._clear(self.user_canvas)
        w = self.user_canvas.winfo_width() or 360
        h = self.user_canvas.winfo_height() or 200
        if w <= 1 or h <= 1:
            return
        size = min(h - 8, w - 8)
        photo = ImageTk.PhotoImage(cv_to_photo(preview_bgr, size, size))
        self._photo_ids["user"] = photo
        self.user_canvas.create_image(w / 2, 4, image=photo, anchor="n")
        self.user_canvas.create_text(w / 2, size + 14, text=tag, fill=DIM,
                                     font=("Microsoft YaHei UI", 8))

    # ================================================================ 评分
    def _do_score(self) -> None:
        if self._score_busy:
            return
        result = self.session.score()
        if result is None:
            self.hint_var.set("样本还不够，请正对摄像头保持不动")
            return

        s = result.score
        self._draw_user_preview(result.preview, "对齐后的人脸")
        self.score_var.set(f"{s.score:.0f}")
        self.verdict_var.set(s.verdict)
        self.comment_var.set(s.comment)

        detail = (
            f"融合余弦 {s.raw:.4f}（等效欧氏距离 {np.sqrt(max(0.0, 2 - 2 * s.raw)):.3f}，"
            f"dlib 同人阈值 0.6）\n"
            f"命中参考：{os.path.splitext(s.best_reference)[0]} · "
            f"采样 {s.sample_count} 帧 · 帧间一致性 {s.stability:.3f}"
        )
        self.detail_var.set(detail)

        style = "Ready" if s.score >= 70 else ("Warn" if s.score >= 40 else "Bad")
        self.score_bar["value"] = s.score
        self.score_bar.configure(style=f"{style}.Horizontal.TProgressbar")

        pct = self.matcher.percent_of_impostors(s.raw)
        if math.isfinite(pct):
            self.status_var.set(
                f"你比 {pct:.0%} 的陌生面孔更像 · {self.matcher.calibrator.summary()}"
            )
        else:
            self.status_var.set(self.matcher.calibrator.summary())
        self.user_tag.configure(text=f"{s.score:.0f} 分")

    # ================================================================ 杂项
    def _clear(self, canvas: tk.Canvas) -> None:
        canvas.delete("all")

    def _on_close(self) -> None:
        self._closing = True
        self.running = False
        self.camera.stop()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    FaceCompareApp(root)  # 实例必须保留引用，否则会被垃圾回收
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
