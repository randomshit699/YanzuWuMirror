"""摄像头采集：后台线程持续抓帧，主线程取最新帧。

描述子计算要 ~265ms/帧，必须和视频采集解耦——否则界面会卡死。
采集线程只负责 grab，推理在 GUI 线程按预算节流执行。
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import config


class Camera:
    """线程安全的摄像头封装。"""

    def __init__(self, index: int = config.CAMERA_INDEX) -> None:
        self.index = index
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error: str = ""
        self.fps = 0.0
        self._frame_times: list[float] = []

    # ------------------------------------------------------------ 生命周期
    def start(self) -> bool:
        if self._running:
            return True
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
        if self._cap is None or not self._cap.isOpened():
            self._last_error = f"无法打开摄像头 {self.index}（设备被占用或不存在）"
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
        self.fps = 0.0
        self._frame_times.clear()

    @property
    def is_running(self) -> bool:
        return self._running and self._cap is not None

    @property
    def error(self) -> str:
        return self._last_error

    # ------------------------------------------------------------ 抓帧
    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
            self._frame_times.append(time.perf_counter())
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
        self._frame_times.clear()

    def read(self) -> np.ndarray | None:
        """取最新一帧（不阻塞）。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _update_fps(self) -> None:
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 1e-6:
                self.fps = (len(self._frame_times) - 1) / span

    def current_fps(self) -> float:
        self._update_fps()
        return self.fps

    def __enter__(self) -> Camera:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
