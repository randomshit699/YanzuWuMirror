"""实时会话：把摄像头、人脸引擎、打分器串成一个状态机。

关键约束：dlib 描述子约 265ms/次，UI 刷新约 33ms/帧。两者不可能每帧都跑，
所以这里分成两条独立节流：

  * **检测**（约 40ms）：每帧都跑，负责画人脸框
  * **描述子**（约 265ms）：只在"人脸合格"时按固定间隔跑，攒够 N 帧就出分

多帧取中位数融合，可以显著压掉摄像头噪声导致的分数抖动。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from . import config
from .engine import Face, FaceEngine, assess_quality
from .scoring import Matcher, ScoreResult


class State(Enum):
    NO_CAMERA = "摄像头未开启"
    SEARCHING = "寻找人脸"
    BAD_QUALITY = "姿态或画质不合格"
    SAMPLING = "正在采集样本"
    READY = "可以评分"
    SCORING = "评分中"


def _is_blank(frame: np.ndarray) -> bool:
    """判断画面是不是一片纯色（摄像头没真正在送图像）。

    实测遇到过：设备能被打开、能读到帧，但每帧都是同一个灰色值
    （均值 11、标准差 0）。这时不该提示"没看到人脸"，而要提示设备有问题。
    """
    if frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < 2.0


@dataclass
class SessionResult:
    score: ScoreResult
    face: Face
    frame: np.ndarray
    preview: np.ndarray


@dataclass
class LiveSession:
    engine: FaceEngine
    matcher: Matcher
    analysis_width: int = config.ANALYSIS_WIDTH

    state: State = State.NO_CAMERA
    hint: str = ""
    face: Face | None = None          # 最近一次完整分析（含描述子）
    # 检测框。**坐标一律是原始帧（= 界面显示帧）的像素坐标**，
    # 和 face.box / face.landmarks 保持同一套坐标系，界面直接拿去画即可。
    # 这里曾经把 boxes 留在分析帧坐标、face 留在原始帧坐标，
    # 导致界面重复缩放、框和人脸对不上——不要再引入第二套坐标系。
    boxes: list[tuple] = field(default_factory=list)
    frame: np.ndarray | None = None   # 分析用的降采样帧
    sample_count: int = 0
    last_score: ScoreResult | None = None
    last_result: SessionResult | None = None

    _samples: deque = field(default_factory=lambda: deque(maxlen=config.LIVE_BUFFER_SIZE))
    _last_embed_time: float = 0.0
    _scoring_until: float = 0.0

    # ------------------------------------------------------------------
    def reset_samples(self) -> None:
        self._samples.clear()
        self.sample_count = 0
        self.last_score = None
        self.face = None

    def set_state(self, state: State, hint: str = "") -> None:
        if self.state != state or self.hint != hint:
            self.state = state
            self.hint = hint

    # ------------------------------------------------------------------
    def tick(self, raw_frame: np.ndarray | None, now: float | None = None) -> None:
        """处理一帧：更新画面、检测人脸、按节流策略补采样本。

        ``now`` 显式传入时用于测试（可以跳过节流）；正常运行时留空用真实时钟。
        """
        if raw_frame is None:
            self.set_state(State.NO_CAMERA, "没读到画面")
            return
        now = now if now is not None else time.perf_counter()

        scale = self.analysis_width / raw_frame.shape[1]
        if scale < 1.0:
            frame = cv2.resize(
                raw_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            # 已经比分析分辨率还小（或正好相等），保持原图，避免无谓的降采样
            frame = raw_frame
        self.frame = frame

        detected = self.engine.detect(frame)
        if not detected:
            self.boxes = []
            self._samples.clear()
            self.sample_count = 0
            self.face = None
            # 画面本身就是一片纯色（摄像头被占用/被隐私遮挡/选错设备）时，
            # 光说"没看到人脸"会让人以为是自己的问题
            if _is_blank(frame):
                self.set_state(
                    State.SEARCHING, "摄像头画面是空的，请检查设备是否被其它程序占用"
                )
            else:
                self.set_state(State.SEARCHING, "没看到人脸，请正对摄像头")
            return

        # 检测在降采样帧上跑（快），但对外暴露的所有坐标都换算回原始帧，
        # 界面只需要一套坐标系。
        fh, fw = frame.shape[:2]
        kx, ky = raw_frame.shape[1] / fw, raw_frame.shape[0] / fh
        raw_boxes = [
            ((box[0] * kx, box[1] * ky, box[2] * kx, box[3] * ky), score)
            for box, score in detected
        ]
        self.boxes = raw_boxes

        # 描述子很贵：只在距上次 >= EMBED_INTERVAL 时才算
        due = (now - self._last_embed_time) * 1000.0 >= config.EMBED_INTERVAL_MS
        if not due:
            self.set_state(State.SAMPLING, "正在分析…")
            return

        self._last_embed_time = now

        # 从原始帧裁人脸并统一缩放——这样描述子的输入尺度与
        # 人脸在画面里的大小无关，用户离镜头远近变化时分数才稳。
        best: Face | None = None
        best_key = float("-inf")
        for rbox, score in raw_boxes:
            cropped = self.engine.normalize_face_crop(raw_frame, rbox)
            if cropped is None:
                continue
            face = self.engine.analyze(cropped.image, cropped.box, score)
            if face is None:
                continue
            # 关键点用裁剪块自己的 origin/scale 还原到原始帧坐标。
            # 注意不能用"分析帧 -> 原始帧"的缩放比来换算，两者毫无关系。
            face.landmarks = cropped.to_source(face.landmarks)
            face.box = tuple(round(v) for v in rbox)  # type: ignore[assignment]
            face.quality = assess_quality(raw_frame, face.landmarks, rbox)
            # 选脸优先级：姿态合格 > 面积大
            key = (1e6 if face.acceptable else 0.0) + face.size
            if key > best_key:
                best, best_key = face, key
            if face.acceptable:
                break

        if best is None:
            self._samples.clear()
            self.sample_count = 0
            self.face = None
            self.set_state(State.BAD_QUALITY, "这张脸不太行，换个姿势")
            return
        face = best

        if not face.acceptable:
            # 姿态/画质不合格：不进样本池
            self.face = face
            self.set_state(State.BAD_QUALITY, face.hint)
            return

        self.face = face
        self._samples.append(face.embedding)
        self.sample_count = len(self._samples)
        if self.sample_count >= config.MIN_SAMPLES_TO_SCORE:
            if self.sample_count >= config.LIVE_BUFFER_SIZE:
                self.set_state(State.READY, "样本已就绪，点「立即评分」")
            else:
                self.set_state(
                    State.READY,
                    f"已可评分（{self.sample_count}/{config.LIVE_BUFFER_SIZE}），点「立即评分」",
                )
        else:
            self.set_state(
                State.SAMPLING, f"正在采集 {self.sample_count}/{config.MIN_SAMPLES_TO_SCORE}"
            )

    # ------------------------------------------------------------------
    def can_score(self) -> bool:
        return (
            self.face is not None
            and self.face.acceptable
            and len(self._samples) >= config.MIN_SAMPLES_TO_SCORE
            and bool(self.matcher.references)
        )

    def score(self) -> SessionResult | None:
        """对当前样本池打分。会清空样本池，强制用户重新摆一次。"""
        if not self.can_score() or self.face is None or self.frame is None:
            return None
        self.set_state(State.SCORING, "正在计算相似度…")
        result = self.matcher.score(list(self._samples))
        out = SessionResult(
            score=result,
            face=self.face,
            frame=self.frame.copy(),
            preview=self.face.preview.copy(),
        )
        self.last_score = result
        self.last_result = out
        self._samples.clear()
        self.sample_count = 0
        return out
