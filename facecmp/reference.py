"""参考图库加载：读图、检测人脸、抽取描述子，并做自检。"""

from __future__ import annotations

import os

import cv2

from . import config
from .engine import FaceEngine
from .scoring import GalleryReport, Matcher, ReferenceEntry


def load_references(
    engine: FaceEngine,
    directory=None,
    require_frontal: bool = True,
) -> tuple[list[ReferenceEntry], list[tuple[str, str]]]:
    """扫描目录里的所有图片，返回 (合格条目, [(文件名, 剔除原因)])。

    require_frontal=True 时只接受姿态合格的正面照。侧脸照的描述子不稳定，
    混进图库会污染"同一人"分布。
    """
    directory = directory or config.REFERENCE_DIR
    entries: list[ReferenceEntry] = []
    rejected: list[tuple[str, str]] = []

    if not os.path.isdir(directory):
        return entries, [(str(directory), "参考图目录不存在")]

    for name in sorted(os.listdir(directory)):
        if name.startswith((".", "_")):
            continue
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        path = os.path.join(directory, name)
        image = cv2.imread(path)
        if image is None:
            rejected.append((name, "图片读取失败"))
            continue
        face = engine.best_face(image, require_acceptable=True)
        if face is None:
            rejected.append((name, "未检测到人脸"))
            continue
        if require_frontal and not face.acceptable:
            rejected.append((name, f"姿态/画质不合格：{face.hint}"))
            continue
        entries.append(
            ReferenceEntry(
                name=name,
                path=path,
                embedding=face.embedding,
                image=image,
                box=face.box,
                landmarks=face.landmarks,
            )
        )
    return entries, rejected


def prepare_matcher(
    matcher: Matcher,
    engine: FaceEngine,
    directory=None,
    verbose: bool = False,
) -> GalleryReport:
    """加载 + 自检 + 写回 matcher，返回自检报告。"""
    entries, rejected = load_references(engine, directory)
    matcher.set_references(entries)
    report = matcher.validate_gallery()
    for name, why in rejected:
        report.rejected.append((name, why))
    matcher.set_references(report.entries)
    if verbose:
        for name, why in report.rejected:
            print(f"  剔除 {name}: {why}")
        for w in report.warnings:
            print(f"  ! {w}")
    return report
