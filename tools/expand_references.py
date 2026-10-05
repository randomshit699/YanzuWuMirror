"""补充吴彦祖的正面参考照，用于构建"同一人"分布。

从 Wikimedia Commons 搜索并下载，只保留姿态合格（正面、清晰）的照片。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from facecmp import config
from facecmp.engine import FaceEngine

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "FaceCompareDemo/1.0 (educational, local)"}

TERMS = [
    "Daniel Wu actor portrait", "Daniel Wu 2015", "Daniel Wu press conference",
    "\u5433\u5f65\u7956", "Daniel Wu face", "Daniel Wu SDCC", "Daniel Wu WonderCon",
    "Daniel Wu 2019", "Daniel Wu 2011", "Daniel Wu premiere",
]


def api(params, retries=4):
    url = API + "?" + urllib.parse.urlencode(dict(params, format="json"))
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except Exception as exc:
            last = exc
    raise last


def fetch(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                buf = r.read()
            return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            if i == retries - 1:
                return None
    return None


import numpy as np


def main() -> int:
    engine = FaceEngine()
    existing = {
        p.name for p in config.REFERENCE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("_")
    }
    print("已有:", len(existing))

    titles: dict[str, None] = {}
    for term in TERMS:
        try:
            d = api({"action": "query", "list": "search", "srsearch": f"{term} filetype:bitmap",
                     "srnamespace": 6, "srlimit": 20})
        except Exception as exc:
            print("搜索失败", term, exc)
            continue
        for it in d.get("query", {}).get("search", []):
            t = it["title"]
            if t.lower().endswith((".jpg", ".jpeg", ".png")) and t not in titles:
                titles[t] = None
    print("候选:", len(titles))

    # 拿缩略图（更快）
    urls: dict[str, str] = {}
    keys = list(titles)
    for i in range(0, len(keys), 20):
        chunk = keys[i:i + 20]
        try:
            d = api({"action": "query", "titles": "|".join(chunk), "prop": "imageinfo",
                     "iiprop": "url|size", "iiurlwidth": 900})
        except Exception as exc:
            print("取链接失败", exc)
            continue
        for page in d.get("query", {}).get("pages", {}).values():
            ii = page.get("imageinfo")
            if ii and ii[0].get("width", 0) >= 400:
                urls[page["title"]] = ii[0].get("thumburl") or ii[0]["url"]

    added = 0
    for title, url in urls.items():
        if added >= 10:
            break
        slug = "".join(c for c in title.replace("File:", "") if c.isalnum() or c in "._-")[:60]
        if any(slug in e for e in existing):
            continue
        img = fetch(url)
        if img is None:
            continue
        face = engine.best_face(img, require_acceptable=True)
        if face is None or not face.acceptable:
            continue
        out = config.REFERENCE_DIR / f"daniel_wu_{len(existing) + added + 10:02d}_{slug}.jpg"
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        existing.add(out.name)
        added += 1
        print(f"  + {out.name}  size={face.size}  {img.shape}")

    print(f"\n新增 {added} 张，参考库共 {len(existing)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
