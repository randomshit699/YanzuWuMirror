"""下载一批"其他人"的人脸照片，用于标定相似度分布。

只取 Wikimedia Commons 上自由授权的肖像照，每张只用来提取 128 维描述子，
不保存原图。产出 assets/calibration/impostor_*.npy
"""
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from facecmp import config
from facecmp.engine import FaceEngine

OUT = config.ASSET_DIR / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "FaceCompareDemo/1.0 (educational, local)"}

# 一批知名人物，各取 2-3 张不同照片；这些人彼此之间就是"冒名者"样本
PEOPLE = [
    "Andy Lau", "Aaron Kwok", "Chow Yun-fat", "Tony Leung Chiu-wai", "Leslie Cheung",
    "Jackie Chan", "Jet Li", "Donnie Yen", "Nicholas Tse", "Eason Chan",
    "Jay Chou", "Wang Leehom", "Takeshi Kaneshiro", "Hiroshi Abe",
    "Rain South Korea", "Kim Soo-hyun", "Song Joong-ki", "Lee Min-ho", "Yoona",
    "Louis Koo", "Shawn Yue", "Ronald Cheng", "Ekin Cheng", "Wu Chun",
    "Ryo Nishikido", "Shota Matsuda", "Takayuki Yamada", "Shu Qi", "Faye Wong",
]


def api(params, retries=4):
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except Exception as exc:
            last = exc
    raise last


def search_files(person, limit=3):
    d = api({
        "action": "query", "list": "search",
        "srsearch": f'{person} filetype:bitmap', "srnamespace": 6, "srlimit": limit * 3,
    })
    return [it["title"] for it in d.get("query", {}).get("search", [])
            if it["title"].lower().endswith((".jpg", ".jpeg", ".png"))][:limit]


def image_urls(titles):
    out = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        if not chunk:
            continue
        d = api({"action": "query", "titles": "|".join(chunk), "prop": "imageinfo",
                 "iiprop": "url|size", "iiurlwidth": 800})
        for page in d.get("query", {}).get("pages", {}).values():
            ii = page.get("imageinfo")
            if ii and ii[0].get("width", 0) >= 400:
                out[page["title"]] = ii[0].get("thumburl") or ii[0]["url"]
    return out


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                return np.frombuffer(r.read(), np.uint8)
        except Exception as exc:
            if attempt == retries - 1:
                print("   下载失败:", exc)
                return None
    return None


def main():
    engine = FaceEngine()
    cache = OUT / "impostor_cache.npz"
    store: dict[str, list[np.ndarray]] = {}
    if cache.exists():  # 断点续跑
        data = np.load(cache, allow_pickle=True)
        for name in data.files:
            store[str(name)] = list(data[name])

    def flush():
        """保存断点续跑用的缓存 + 标签/统计（标签和描述子严格同序）。"""
        np.savez(cache, **{k: np.vstack(v) for k, v in store.items() if v})
        meta = {"people": {k: len(v) for k, v in store.items()}, "total": sum(len(v) for v in store.values())}
        (OUT / "impostor_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        (OUT / "impostor_labels.json").write_text(
            json.dumps([n for n, v in store.items() for _ in v], ensure_ascii=False, indent=1),
            "utf-8",
        )

    for person in PEOPLE:
        if len(store.get(person, [])) >= 3:
            print(f"== {person} (已有 {len(store[person])} 张，跳过)")
            continue
        print(f"== {person}")
        try:
            titles = search_files(person, 3)
            urls = image_urls(titles) if titles else {}
        except Exception as exc:
            print("   搜索失败:", exc)
            continue
        got = 0
        for title, url in urls.items():
            if got >= 3:
                break
            buf = fetch(url)
            if buf is None:
                continue
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                face = engine.best_face(img, require_acceptable=True)
            except Exception as exc:
                print("   分析失败:", exc)
                continue
            if face is None or not face.acceptable:
                continue
            store.setdefault(person, []).append(face.embedding)
            got += 1
            print(f"   + {title[:60]}  size={face.size}")
        if got:
            print(f"   本轮采到 {got} 张，累计 {len(store.get(person, []))}")
        flush()

    total = sum(len(v) for v in store.values())
    print(f"\n共 {len(store)} 人 / {total} 张")
    if not store:
        print("没抓到样本，退出")
        return 1

    flush()
    np.save(OUT / "impostor_embeddings.npy",
            np.vstack([e for v in store.values() for e in v]))
    print("已保存:", OUT / "impostor_embeddings.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
