"""人脸检测、68 点关键点、128 维人脸特征提取。

技术选型（全部来自 dlib，随 ``dlib-bin`` + ``face-recognition-models`` 离线安装）：

  * ``get_frontal_face_detector``  HOG 实时人脸检测，480px 下一帧约 40ms
  * ``shape_predictor_68``         68 点关键点，约 2ms
  * ``face_recognition_model_v1``  ResNet 人脸描述子（128 维），约 265ms

关于对齐的一个要点：``compute_face_descriptor`` 内部会用 68 个关键点做一次
对齐（``get_alignment_proper``），所以**不要**再自己做仿射变换裁剪。
实测过：先用 5 点相似变换裁成 150x150 再算描述子，同一个人的两两余弦会从
0.996 掉到 0.98，自一致性反而变差。因此这里保持原图 + 68 点检测框直接送进
描述子，只在"给人看的预览"时才自己画对齐图。
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import dlib
import numpy as np

from . import config

# dlib 68 点模型中我们关心的几个索引
IDX_CHIN = 8
IDX_NOSE_TIP = 30
IDX_EYE_R = list(range(36, 42))  # 画面左侧那只眼（人物自己的右眼）
IDX_EYE_L = list(range(42, 48))  # 画面右侧那只眼
IDX_MOUTH_R = 48
IDX_MOUTH_L = 54
INNER_LIPS = list(range(62, 66))

# 给人预览用的对齐模板（5 点），仅用于画图，不参与打分
PREVIEW_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


def _model_dir() -> Path:
    import face_recognition_models

    return Path(face_recognition_models.__file__).resolve().parent / "models"


def _model_path(name: str) -> str:
    path = _model_dir() / name
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"缺少 dlib 模型文件 {path}\n请先执行: pip install dlib-bin face-recognition-models"
        )
    return str(path)


@dataclass
class Face:
    """一次人脸分析的完整结果。"""

    box: tuple[int, int, int, int]
    embedding: np.ndarray  # (128,) 已 L2 归一化
    landmarks: np.ndarray  # (68, 2)
    preview: np.ndarray  # 对齐后的小图，仅供显示
    det_score: float = 0.0
    quality: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return min(self.box[2], self.box[3])

    @property
    def acceptable(self) -> bool:
        return bool(self.quality.get("acceptable", False))

    @property
    def hint(self) -> str:
        return str(self.quality.get("hint", ""))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """最小二乘最优相似变换（Umeyama 1991）：旋转 + 等比缩放 + 平移，禁止镜像。

    推导要点（这里两个地方极易写错，写错了图片上表现为"对齐到了旁边的背景"）：

      * 协方差必须除以点数 n：``C = XᵀY / n``。漏掉 /n 会让尺度恰好放大 n 倍。
      * 旋转是 ``R = V Uᵀ`` 而不是 ``U Vᵀ``（C = U D Vᵀ）。
      * 尺度 ``c = tr(D·S) / σ_x²``，σ_x² 是源点云的平均离差平方和。

    返回 2x3 矩阵 M，满足 ``dst ≈ [M[:, :2] | M[:, 2]] @ [src, 1]``。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = (src_c**2).sum() / n
    if var_src < 1e-12:
        return np.hstack([np.eye(d), np.zeros((d, 1))])

    cov = (src_c.T @ dst_c) / n
    u, singular, vt = np.linalg.svd(cov)
    # S 纠正行列式，保证只有旋转、没有镜像
    correction = np.eye(d)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    scale = float((singular * np.diag(correction)).sum() / var_src)
    matrix = scale * rotation
    return np.hstack([matrix, (mu_dst - matrix @ mu_src).reshape(d, 1)])


@dataclass
class FaceCrop:
    """裁出来的人脸块，以及把坐标映射回原图所需的信息。

    之前这里只返回 (image, box)，结果关键点从裁剪块还原到原图时
    没有偏移/缩放信息可用，界面上画出来的关键点全是错位的。
    """

    image: np.ndarray
    box: tuple[int, int, int, int]  # 检测框（裁剪块坐标）
    origin: tuple[int, int]         # 裁剪块左上角在原图中的位置
    scale: float                    # 裁剪块相对原图的缩放倍数

    def to_source(self, points: np.ndarray) -> np.ndarray:
        """裁剪块坐标 -> 原图坐标。

        裁剪块是把原图区域按 ``scale`` 缩放得到的，所以还原要**除以** scale
        （scale < 1 表示被缩小，除回去才是原图坐标）。
        """
        pts = np.asarray(points, dtype=np.float64) / (self.scale or 1.0)
        pts[..., 0] += self.origin[0]
        pts[..., 1] += self.origin[1]
        return pts


class FaceEngine:
    """dlib 模型封装。检测很快，描述子较慢，调用方需要自己按需节流。"""

    def __init__(self, detector: str = config.DLIB_FACE_DETECTOR) -> None:
        self._lock = threading.Lock()
        self._detector_name = detector
        self._hog = dlib.get_frontal_face_detector()
        self._cnn = None
        if detector == "cnn":
            self._cnn = dlib.cnn_face_detection_model_v1(_model_path("mmod_human_face_detector.dat"))
        elif detector != "hog":
            raise ValueError(f"未知检测器: {detector}")
        self._predictor = dlib.shape_predictor(_model_path("shape_predictor_68_face_landmarks.dat"))
        self._resnet = dlib.face_recognition_model_v1(
            _model_path("dlib_face_recognition_resnet_model_v1.dat")
        )

    # ------------------------------------------------------------ 检测
    def detect(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """人脸框检测，按面积从大到小。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found: list[tuple[tuple[int, int, int, int], float]] = []
        if self._cnn is not None:
            for det in self._cnn(gray, 1):
                r = det.rect
                found.append(
                    (
                        (int(r.left()), int(r.top()), int(r.right() - r.left()), int(r.bottom() - r.top())),
                        float(getattr(det, "confidence", 0.0)),
                    )
                )
        else:
            rects, scores, _ = self._hog.run(gray, 1)
            for i, r in enumerate(rects):
                found.append(
                    (
                        (int(r.left()), int(r.top()), int(r.right() - r.left()), int(r.bottom() - r.top())),
                        float(scores[i]) if scores is not None and i < len(scores) else 0.0,
                    )
                )

        height, width = image.shape[:2]
        cleaned = []
        for box, score in found:
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w > 20 and h > 20:
                cleaned.append(((x, y, w, h), score))
        cleaned.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
        return cleaned

    # ------------------------------------------------------------ 关键点 + 描述子
    def _to_rect(self, box, shape) -> dlib.rectangle:
        height, width = shape[:2]
        x, y, w, h = box
        return dlib.rectangle(
            max(0, int(x)),
            max(0, int(y)),
            min(width - 1, int(x + w)),
            min(height - 1, int(y + h)),
        )

    def landmarks(self, image: np.ndarray, box) -> np.ndarray | None:
        rect = self._to_rect(box, image.shape)
        try:
            with self._lock:
                detection = self._predictor(image, rect)
        except RuntimeError:
            return None
        points = np.array([[p.x, p.y] for p in detection.parts()], dtype=np.float64)
        return points if points.shape == (68, 2) and np.isfinite(points).all() else None

    def embed(self, image: np.ndarray, box) -> np.ndarray | None:
        """整图 + 检测框 -> 128 维描述子。dlib 内部负责对齐。"""
        rect = self._to_rect(box, image.shape)
        try:
            with self._lock:
                detection = self._predictor(image, rect)
                if len(detection.parts()) != 68:
                    return None
                descriptor = self._resnet.compute_face_descriptor(image, detection, 0)
        except RuntimeError:
            return None
        return l2_normalize(np.array(descriptor, dtype=np.float64))

    def analyze(self, image: np.ndarray, box, det_score: float = 0.0) -> Face | None:
        """关键点 + 描述子 + 画质评估，一次跑完（描述子较贵，别每帧调）。"""
        rect = self._to_rect(box, image.shape)
        try:
            with self._lock:
                detection = self._predictor(image, rect)
                if len(detection.parts()) != 68:
                    return None
                descriptor = self._resnet.compute_face_descriptor(image, detection, 0)
        except RuntimeError:
            return None

        landmarks = np.array([[p.x, p.y] for p in detection.parts()], dtype=np.float64)
        return Face(
            box=tuple(int(v) for v in box),  # type: ignore[arg-type]
            embedding=l2_normalize(np.array(descriptor, dtype=np.float64)),
            landmarks=landmarks,
            preview=self._preview(image, landmarks),
            det_score=det_score,
            quality=assess_quality(image, landmarks, box),
        )

    def best_face(self, image: np.ndarray, require_acceptable: bool = True) -> Face | None:
        """从整张图里挑一张最适合打分的脸：够大、够正、够居中。"""
        width = image.shape[1]
        center_x = width / 2.0
        best: Face | None = None
        best_rank = float("-inf")
        for box, score in self.detect(image):
            if min(box[2], box[3]) < config.MIN_FACE_SIZE:
                continue
            face = self.analyze(image, box, score)
            if face is None:
                continue
            if require_acceptable and not face.acceptable:
                if best is None:
                    best = face
                continue
            offset = abs((box[0] + box[2] / 2) - center_x) / max(width, 1)
            rank = face.size * (1.2 - min(offset, 1.0)) * (1.0 if face.acceptable else 0.4) + 0.05 * score
            if rank > best_rank:
                best, best_rank = face, rank
        return best

    # ------------------------------------------------------------ 裁剪与归一化
    def normalize_face_crop(
        self,
        image: np.ndarray,
        box,
        margin: float = config.FACE_CROP_MARGIN,
        target: int = config.FACE_ANALYSIS_SIZE,
    ) -> FaceCrop | None:
        """从原图裁出人脸区域，并把短边缩放到 ``target``。

        为什么要单独做这一步：
        dlib 的关键点模型和描述子都是在"人脸大约 100-200px"这个尺度上训练的。
        如果直接把整帧丢进去，人脸在画面里的大小会随用户离镜头的远近
        剧烈变化，关键点精度跟着抖动。统一裁剪 + 缩放后，
        无论人脸在画面里是 300px 还是 80px，送进网络的尺度都一样。

        返回 :class:`FaceCrop`（含坐标映射信息），框无效时返回 None。
        """
        ih, iw = image.shape[:2]
        x, y, w, h = (float(v) for v in box)
        if w < 8 or h < 8:
            return None
        cx, cy = x + w / 2, y + h / 2
        side_w, side_h = w * (1 + 2 * margin), h * (1 + 2 * margin)
        x0 = round(min(max(0.0, cx - side_w / 2), max(0.0, iw - side_w)))
        y0 = round(min(max(0.0, cy - side_h / 2), max(0.0, ih - side_h)))
        x1 = round(min(x0 + side_w, iw))
        y1 = round(min(y0 + side_h, ih))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None

        crop = image[y0:y1, x0:x1]
        scale = target / min(crop.shape[0], crop.shape[1])
        if abs(scale - 1.0) > 0.05:
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            crop = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * scale)),
                 max(1, round(crop.shape[0] * scale))),
                interpolation=interp,
            )
        crop = np.ascontiguousarray(crop)
        # 检测框跟着裁剪块一起换算
        crop_box = (round((x - x0) * scale), round((y - y0) * scale),
                    round(w * scale), round(h * scale))
        return FaceCrop(image=crop, box=crop_box, origin=(x0, y0), scale=float(scale))

    # ------------------------------------------------------------ 预览图
    def _preview(self, image: np.ndarray, landmarks: np.ndarray, size: int = 150) -> np.ndarray:
        """仅用于展示的对齐小图（不参与打分）。

        注意 flag：``similarity_transform`` 返回的是 src->dst 的矩阵，而
        ``cv2.warpAffine`` 默认约定矩阵是 dst->src（内部会自己求逆）。
        所以这里**不能**加 WARP_INVERSE_MAP，否则会二次取逆，对齐到背景上去。
        """
        points = np.array(
            [
                landmarks[IDX_EYE_R].mean(axis=0),
                landmarks[IDX_EYE_L].mean(axis=0),
                landmarks[IDX_NOSE_TIP],
                landmarks[IDX_MOUTH_R],
                landmarks[IDX_MOUTH_L],
            ]
        )
        matrix = similarity_transform(points, PREVIEW_TEMPLATE)
        return cv2.warpAffine(
            image,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


def assess_quality(image: np.ndarray, landmarks: np.ndarray, box) -> dict:
    """用 68 点关键点 + 像素统计判断"这张脸能不能拿来评分"。"""
    eye_r = landmarks[IDX_EYE_R].mean(axis=0)
    eye_l = landmarks[IDX_EYE_L].mean(axis=0)
    nose = landmarks[IDX_NOSE_TIP]
    chin = landmarks[IDX_CHIN]

    eye_dist = float(np.linalg.norm(eye_l - eye_r))
    if eye_dist < 1e-6 or not np.isfinite(eye_dist):
        return {"acceptable": False, "hint": "关键点异常", "roll": 0.0, "yaw": 1.0, "pitch": 0.0}

    # 侧头 roll：双眼连线倾角
    roll = math.degrees(math.atan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0]))
    # 偏头 yaw：鼻尖相对双眼中点的水平偏移 / 眼距
    eye_mid = (eye_r + eye_l) / 2.0
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    # 仰俯 pitch：鼻尖在"眼中线 -> 下巴"这条竖线上的相对高度
    total = float(chin[1] - eye_mid[1])
    pitch = 0.0 if abs(total) < 1e-6 else float(nose[1] - eye_mid[1]) / total

    # 像素层面：亮度 + 清晰度
    x, y, w, h = (int(v) for v in box)
    pad_x, pad_y = int(w * 0.2), int(h * 0.2)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)
    crop = image[y0:y1, x0:x1]
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        brightness = sharpness = 0.0

    reasons: list[str] = []
    if abs(roll) > config.POSE_ROLL_MAX_DEG:
        reasons.append("头摆正一点，别侧着")
    if abs(yaw) > config.POSE_YAW_MAX:
        reasons.append("请正对摄像头")
    if not 0.25 <= pitch <= 0.70:
        reasons.append("下巴抬一点，视线看镜头")
    if brightness < config.MIN_FACE_BRIGHTNESS:
        reasons.append("光线太暗了")
    if sharpness < config.MIN_FACE_SHARPNESS:
        reasons.append("画面有点糊，靠近点")
    if min(w, h) < config.GOOD_FACE_SIZE:
        reasons.append("再靠近摄像头一点")

    return {
        "acceptable": not reasons,
        "hint": "、".join(reasons),
        "roll": roll,
        "yaw": yaw,
        "pitch": pitch,
        "brightness": brightness,
        "sharpness": sharpness,
    }


def median_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """逐维取中位数再归一化，比均值更抗单帧异常。"""
    if not embeddings:
        raise ValueError("没有可用的描述子")
    stacked = np.vstack(embeddings)
    return l2_normalize(np.median(stacked, axis=0))


def confidence(embeddings: list[np.ndarray]) -> float:
    """采样一致性：两两余弦的均值，越接近 1 说明这一帧序列越稳定。"""
    if len(embeddings) < 2:
        return 0.0
    stacked = np.vstack(embeddings)
    sims = stacked @ stacked.T
    n = len(embeddings)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def env_report() -> str:
    """给界面状态栏用的一行环境信息。"""
    import numpy

    return f"OpenCV {cv2.__version__} · dlib {getattr(dlib, '__version__', 'n/a')} · NumPy {numpy.__version__}"


if __name__ == "__main__":  # pragma: no cover
    print(env_report())
    print("模型目录:", _model_dir())
    for f in sorted(os.listdir(_model_dir())):
        print("  -", f)
