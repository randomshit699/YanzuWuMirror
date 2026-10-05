"""程序入口：python run.py"""

from __future__ import annotations

import sys
import warnings

# face_recognition_models 还在用 pkg_resources，这里屏蔽它的噪音警告
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from facecmp.app import main

if __name__ == "__main__":
    sys.exit(main())
