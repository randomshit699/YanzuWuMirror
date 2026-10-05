"""对比三条描述子提取路径的一致性，选出最稳的一条。"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import dlib
import face_recognition_models as m
import numpy as np

from facecmp import config
from facecmp.engine import FACE_TEMPLATE, l2_normalize, similarity_transform

BASE = os.path.join(os.path.dirname(m.__file__), "models")
res = dlib.face_recognition_model_v1(os.path.join(BASE, "dlib_face_recognition_resnet_model_v1.dat"))
sp68 = dlib.shape_predictor(os.path.join(BASE, "shape_predictor_68_face_landmarks.dat"))
sp5 = dlib.shape_predictor(os.path.join(BASE, "shape_predictor_5_face_landmarks.dat"))
hog = dlib.get_frontal_face_detector()


def order5(raw):
    pts = np.asarray(raw, float)
    o = np.argsort(pts[:, 1], kind="stable")
    eyes = pts[o[:2]]
    nose = pts[o[2:3]]
    mouth = pts[o[3:5]]
    eyes = eyes[np.argsort(eyes[:, 0], kind="stable")]
    mouth = mouth[np.argsort(mouth[:, 0], kind="stable")]
    return np.vstack([eyes, nose, mouth])


def detect(img):
    rects, _scores, _ = hog.run(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    r = max(rects, key=lambda x: (x.right() - x.left()) * (x.bottom() - x.top()))
    return dlib.rectangle(r.left(), r.top(), r.right(), r.bottom())


def path_68(img, box):
    d = sp68(img, box)
    return l2_normalize(np.array(res.compute_face_descriptor(img, d, 1)))


def path_5(img, box):
    d = sp5(img, box)
    return l2_normalize(np.array(res.compute_face_descriptor(img, d, 1)))


def path_manual(img, box):
    d = sp5(img, box)
    pts = order5([[p.x, p.y] for p in d.parts()])
    M = similarity_transform(pts, FACE_TEMPLATE)
    crop = cv2.warpAffine(img, M, (150, 150), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
    crop = np.ascontiguousarray(crop)
    d2 = sp68(crop, dlib.rectangle(0, 0, 149, 149))
    return l2_normalize(np.array(res.compute_face_descriptor(crop, d2, 1)))


files = [f for f in sorted(os.listdir(config.REFERENCE_DIR)) if f.lower().endswith(".jpg") and not f.startswith("_")]
embs = {"68pt": {}, "5pt": {}, "manual": {}}
for f in files:
    img = cv2.imread(str(config.REFERENCE_DIR / f))
    box = detect(img)
    for name, fn in [("68pt", path_68), ("5pt", path_5), ("manual", path_manual)]:
        t = time.time()
        embs[name][f] = fn(img, box)
        if f == files[0]:
            print(f"  {name:7s} t={ (time.time()-t)*1000:.0f}ms")

for name in embs:
    print(f"\n== {name} 两两余弦")
    for a, b in itertools.combinations(files, 2):
        c = float(embs[name][a] @ embs[name][b])
        print(f"   {c:.4f}  {a[:26]:28s} {b[:26]}")
