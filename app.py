"""
Motorcycle DSS - Streamlit Application Entrypoint.
Chuyển tiếp điều khiển đến tầng giao diện web/app.py trong kiến trúc phân tầng.
Đảm bảo tính tương thích ngược khi người dùng chạy:
    streamlit run app.py
hoặc
    streamlit run web/app.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

TARGET_FILE = os.path.join(ROOT_DIR, "web", "app.py")

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    code = compile(f.read(), TARGET_FILE, "exec")
    exec(code, globals())
