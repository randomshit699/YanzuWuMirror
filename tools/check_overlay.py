"""把 session 的检测框/关键点画到帧上，直接输出图片核对坐标。

比从 GUI 截图里目测靠谱得多。
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
from smoke_gui import make_camera_like_frame

from facecmp import config
from facecmp.engine import FaceEngine
from facecmp.reference import prepare_matcher
from facecmp.scoring import Calibrator, Matcher
from facecmp.session import LiveSession

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")
os.makedirs(OUT, exist_ok=True)

engine = FaceEngine()
matcher = Matcher(Calibrator.load())
prepare_matcher(matcher, engine)
session = LiveSession(engine=engine, matcher=matcher)

ref = matcher.references[0]
raw = make_camera_like_frame(ref.image, ref.box, 640, 480)
session.tick(raw, now=time.perf_counter())
print("raw frame      :", raw.shape)
print("analysis frame :", session.frame.shape, f"(ANALYSIS_WIDTH={config.ANALYSIS_WIDTH})")
print("boxes          :", session.boxes)
print("face.box       :", session.face.box if session.face else None)

frame = session.frame
vis = frame.copy()
for entry in session.boxes:
    bx, by, bw, bh = entry[0]
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (200, 200, 0), 1)

if session.face:
    x, y, bw, bh = session.face.box
    ok = session.face.acceptable
    color = (0, 220, 0) if ok else (0, 180, 255)
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
    cv2.putText(vis, f"{min(bw,bh)}px", (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    pts = {
        "R": session.face.landmarks[36:42].mean(0),
        "L": session.face.landmarks[42:48].mean(0),
        "N": session.face.landmarks[30],
        "MR": session.face.landmarks[48],
        "ML": session.face.landmarks[54],
    }
    for k, p in pts.items():
        cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        cv2.putText(vis, k, (int(p[0]) + 5, int(p[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
p = os.path.join(OUT, "overlay_check.png")
cv2.imwrite(p, vis)
print("written:", p, vis.shape)
