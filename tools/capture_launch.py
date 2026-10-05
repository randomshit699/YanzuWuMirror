"""真实启动 run.py，截一张启动后的窗口图，确认界面能正常起来。"""
from __future__ import annotations

import subprocess
import sys
import time

import pyautogui

ROOT = r"C:\Users\Administrator\Documents\Project\OpenCodeProjects\YanzuWuMirror"
OUT = ROOT + r"\tools\_debug\run_launch.png"

proc = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT)
time.sleep(16)  # 等 dlib 加载 + 窗口布局完成

if proc.poll() is not None:
    print("进程已退出，code =", proc.returncode)
    raise SystemExit(1)

# 把主窗口置前（Windows 可能拒绝），失败就按窗口区域裁剪
import pygetwindow as gw

win = None
for w in gw.getAllWindows():
    if "吴彦祖相似度检测" in (w.title or ""):
        win = w
        break

full = pyautogui.screenshot()
if win is not None:
    try:
        if win.isMinimized:
            win.restore()
    except Exception:
        pass
    try:
        win.activate()
        time.sleep(1.2)
        full = pyautogui.screenshot()
    except Exception as exc:
        print("置前被系统拒绝，改为按窗口区域裁剪:", exc)
    if win.width > 200 and win.height > 200:
        full = full.crop((max(win.left, 0), max(win.top, 0),
                          win.left + win.width, win.top + win.height))
    else:
        print("窗口尺寸异常，使用全屏截图")

full.save(OUT)
print("窗口截图:", OUT, full.size)

# 看看有没有弹出多个窗口（应只有主窗口）
titles = [t for t in pyautogui.getAllTitles() if t.strip()]
print("当前窗口标题:", titles[:10])

proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
print("已关闭")
