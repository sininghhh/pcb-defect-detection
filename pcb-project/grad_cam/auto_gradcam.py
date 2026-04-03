import os
import time
from heatmap import yolo_heatmap, get_params

# =============================
# 路径配置
# =============================
WATCH_DIR = r"D:\deeppcb\10668PCB\grad-cam-picture"
SAVE_DIR = r"D:\deeppcb\10668PCB\grad-cam-result"

os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# 已处理记录文件（关键）
# =============================
RECORD_FILE = "processed.txt"

processed = set()

# 启动时读取历史记录
if os.path.exists(RECORD_FILE):
    with open(RECORD_FILE, "r") as f:
        processed = set(f.read().splitlines())

# =============================
# 加载模型
# =============================
model = yolo_heatmap(**get_params())

print("Watching folder...")

# =============================
# 主循环
# =============================
while True:
    files = os.listdir(WATCH_DIR)

    for f in files:

        # 只处理图片
        if not f.lower().endswith((".jpg", ".png")):
            continue

        # 已处理过跳过
        if f in processed:
            continue

        img_path = os.path.join(WATCH_DIR, f)
        save_path = os.path.join(SAVE_DIR, f)

        # 如果结果已经存在，也跳过（双保险）
        if os.path.exists(save_path):
            processed.add(f)
            continue

        # 防止文件还没写完
        if os.path.getsize(img_path) == 0:
            continue

        print("Processing:", f)

        try:
            # 🔥 生成热力图
            model(img_path, SAVE_DIR)

            # 记录已处理
            processed.add(f)

            with open(RECORD_FILE, "a") as rf:
                rf.write(f + "\n")

        except Exception as e:
            print("Error:", e)

    time.sleep(1)
