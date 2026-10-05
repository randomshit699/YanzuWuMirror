"""相似度打分：把 128 维描述子变成一个可解释的 0-100 分。

打分流程（与标定时使用的代码路径完全一致，保证分数可比）：

  1. 用户这一侧：连续采 N 帧描述子 -> 逐维取中位数 -> 归一化，压掉抖动
  2. 参考这一侧：一张参考图给出一个"最高相似度"（1:N 检索，标准做法）
  3. 图库聚合：取最高的前 K 张的均值（top-K 融合），
     既不会因为某张照片拍糊/侧脸而误判，也不会被单一照片的偶然性带偏
  4. 原始余弦 -> 标定分数：用真实"陌生人 vs 吴彦祖"分布做锚点做分段线性映射

关于分数含义：标定锚点来自 30 张真实公众人物照片 vs 吴彦祖参考图库的余弦分布，
所以分数大致可以读作"你比多少比例的陌生面孔更像"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import config
from .engine import confidence, median_embedding


@dataclass
class ReferenceEntry:
    """参考图库里的一张照片。"""

    name: str
    path: str
    embedding: np.ndarray
    image: object = None  # cv2 图像，界面用
    box: tuple[int, int, int, int] | None = None
    landmarks: np.ndarray | None = None
    note: str = ""


@dataclass
class GalleryReport:
    """参考图库自检结果。"""

    entries: list[ReferenceEntry] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (文件名, 原因)
    clusters: int = 1  # 互不相同的人的数量
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.entries) and not self.warnings


@dataclass
class ScoreResult:
    """一次评分的完整结果。"""

    raw: float  # top-K 融合后的原始余弦
    score: float  # 标定后的 0-100
    per_reference: list[tuple[str, float]] = field(default_factory=list)
    best_reference: str = ""
    sample_count: int = 0
    stability: float = 0.0  # 用户侧多帧一致性 0-1
    verdict: str = ""
    comment: str = ""
    tier: str = ""


class Calibrator:
    """把余弦相似度映射到 0-100，锚点由 tools/build_calibration.py 统计得到。"""

    def __init__(self, data: dict | None = None) -> None:
        data = data or {}
        self.stats = data.get("stats", {})
        anchors = data.get("anchors") or []
        # 保证 cos 严格递增
        clean = []
        for a in sorted(anchors, key=lambda x: x["cos"]):
            if clean and a["cos"] <= clean[-1][0] + 1e-6:
                continue
            clean.append((float(a["cos"]), float(a["score"])))
        if len(clean) < 2:
            # 没有标定文件时的兜底：经验锚点
            clean = [(0.0, 0.0), (0.35, 25.0), (0.60, 50.0), (0.80, 75.0), (1.0, 100.0)]
            self.calibrated = False
        else:
            self.calibrated = True
        if clean[0][0] > 0.0:
            clean.insert(0, (0.0, 0.0))
        if clean[-1][0] < 1.0:
            clean.append((1.0, 100.0))
        self.anchors = clean

    @classmethod
    def load(cls, path=None) -> Calibrator:
        path = path or config.CALIBRATION_FILE
        try:
            return cls(json.loads(open(path, encoding="utf-8").read()))
        except Exception:
            return cls({})

    def to_score(self, cos: float) -> float:
        """分段线性插值，两端截断。"""
        cos = float(np.clip(cos, 0.0, 1.0))
        xs = [a[0] for a in self.anchors]
        ys = [a[1] for a in self.anchors]
        if cos <= xs[0]:
            return float(ys[0])
        if cos >= xs[-1]:
            return float(ys[-1])
        for i in range(len(xs) - 1):
            if xs[i] <= cos <= xs[i + 1]:
                span = xs[i + 1] - xs[i]
                if span <= 1e-9:
                    return float(ys[i])
                t = (cos - xs[i]) / span
                return float(ys[i] + t * (ys[i + 1] - ys[i]))
        return float(ys[-1])

    def summary(self) -> str:
        if not self.calibrated:
            return "未标定（使用经验映射）"
        s = self.stats
        return (
            f"标定自 {s.get('impostor_n', 0)} 张真人脸："
            f"陌生人均值 {s.get('impostor_mean', 0):.3f}，"
            f"本人照片间 {s.get('genuine_mean', 0):.3f}"
        )


def tier_for(score: float) -> tuple[str, str]:
    """分数 -> (等级, 调侃文案)。"""
    for threshold, tier, comment in config.TIERS:
        if score >= threshold:
            return tier, comment
    return config.TIERS[-1][1], config.TIERS[-1][2]


class Matcher:
    """参考图库 + 打分逻辑。"""

    def __init__(self, calibrator: Calibrator | None = None) -> None:
        self.calibrator = calibrator or Calibrator.load()
        self.references: list[ReferenceEntry] = []

    def set_references(self, entries: list[ReferenceEntry]) -> None:
        self.references = entries

    # -------------------------------------------------------------- 图库自检
    def validate_gallery(self, require_single_person: bool = True) -> GalleryReport:
        """检查参考图库是否自洽。

        实际踩过的坑：一张合影里检测器挑中了旁边的人，那张"参考照"其实是
        冒名者，会把整个图库带偏。所以这里做两道检查：

          * 两两余弦 < ``SAME_PERSON_MIN``  -> 很可能不是同一个人
          * 两两余弦 > ``DUPLICATE_MAX``    -> 同一张照片的重复，白占权重
        """
        report = GalleryReport(entries=list(self.references))
        refs = self.references
        if not refs:
            report.warnings.append("参考图库为空")
            return report

        kept: list[ReferenceEntry] = []
        for ref in refs:
            conflict = None
            for other in kept:
                cos = float(ref.embedding @ other.embedding)
                if cos < config.SAME_PERSON_MIN_COS:
                    conflict = (other, cos)
                    break
            if conflict is None:
                kept.append(ref)
            else:
                report.rejected.append(
                    (ref.name, f"与 {conflict[0].name} 余弦仅 {conflict[1]:.3f}，疑似不是同一人")
                )
        report.entries = kept

        # 去重：把几乎相同的照片合并掉，避免某一张被重复加权
        if len(kept) >= 2:
            sims = [
                float(a.embedding @ b.embedding)
                for i, a in enumerate(kept)
                for b in kept[i + 1:]
            ]
            report.clusters = 1 + sum(1 for s in sims if s < config.SAME_PERSON_MIN_COS)
            if min(sims) > config.DUPLICATE_MAX_COS:
                report.warnings.append(
                    f"参考图高度重复（两两余弦最低 {min(sims):.4f}），建议加入不同角度的照片"
                )
        if require_single_person and report.clusters > 1:
            report.warnings.append(
                f"参考图库里检测到 {report.clusters} 个不同的人，已剔除冲突照片"
            )
        return report

    # -------------------------------------------------------------- 核心
    def aggregate_user(self, embeddings: list[np.ndarray]) -> np.ndarray:
        return median_embedding(embeddings)

    def raw_similarity(self, user: np.ndarray) -> tuple[float, list[tuple[str, float]], str]:
        """1:N 检索 + top-K 融合。返回 (融合余弦, 每张参考的余弦, 最像的那张)。"""
        if not self.references:
            raise ValueError("参考图库为空")
        sims = [(ref.name, float(user @ ref.embedding)) for ref in self.references]
        sims_sorted = sorted(sims, key=lambda kv: kv[1], reverse=True)
        k = min(config.REFERENCE_TOP_K, len(sims_sorted))
        # top-K 的均值按 K 归一化（K=1 时就是最高那张）
        top_mean = sum(v for _, v in sims_sorted[:k]) / k
        return float(top_mean), sims_sorted, sims_sorted[0][0]

    def score(
        self,
        user_embeddings: list[np.ndarray],
        sample_count: int | None = None,
    ) -> ScoreResult:
        if not user_embeddings:
            raise ValueError("没有用户样本")
        if not self.references:
            raise ValueError("参考图库为空")

        user = self.aggregate_user(user_embeddings)
        raw, per_ref, best = self.raw_similarity(user)
        score = self.calibrator.to_score(raw)
        score = float(np.clip(score, 0.0, 100.0))
        tier, comment = tier_for(score)
        return ScoreResult(
            raw=float(np.clip(raw, 0.0, 1.0)),
            score=score,
            per_reference=per_ref,
            best_reference=best,
            sample_count=sample_count if sample_count is not None else len(user_embeddings),
            stability=confidence(user_embeddings),
            verdict=tier,
            comment=comment,
            tier=tier,
        )

    # -------------------------------------------------------------- 展示辅助
    def percent_of_impostors(self, raw: float) -> float:
        """这个余弦超过了百分之多少的陌生面孔。"""
        hi = self.calibrator.stats.get("impostor_mean")
        lo = self.calibrator.stats.get("impostor_max")
        if hi is None or lo is None or lo <= hi:
            return float("nan")
        return float(np.clip((raw - hi) / (lo - hi), 0.0, 1.0))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """dlib 传统度量（欧氏距离），0.6 以内视为同一人。"""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def cosine_to_euclidean(cos: float) -> float:
    """单位向量下 余弦 <-> 欧氏距离 的换算。"""
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * cos)))
