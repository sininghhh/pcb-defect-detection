import cv2
import numpy as np
import onnxruntime as ort
import sys
import time
import requests

# =============================
# 类别
# =============================
CLASS_NAMES = [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spurious_copper',
    'spur'
]

# =============================
# PC地址
# =============================
PC_URL = "http://10.104.88.87:5000/upload"
#PC_URL = "http://10.38.166.71:5000/upload"

# =============================
# 加载模型
# =============================
MODEL_PATH = sys.argv[1]

print("Loading model:", MODEL_PATH)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
IMG_SIZE = session.get_inputs()[0].shape[2]

print("Model input size:", IMG_SIZE)

# =============================
# 中心裁剪（正方形）
# =============================
def center_crop(img):
    h, w = img.shape[:2]
    size = min(h, w)

    start_x = (w - size) // 2
    start_y = (h - size) // 2

    return img[start_y:start_y+size, start_x:start_x+size]

# =============================
# Letterbox
# =============================
def letterbox(img, size):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)

    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    dx = (size - nw) // 2
    dy = (size - nh) // 2

    canvas[dy:dy+nh, dx:dx+nw] = resized

    return canvas, scale, dx, dy

# =============================
# 预处理
# =============================
def preprocess(img):
    img = center_crop(img)
    img, scale, dx, dy = letterbox(img, IMG_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    return img, scale, dx, dy

# =============================
# 后处理
# =============================
def postprocess(pred, w, h, scale, dx, dy):

    preds = pred[0].T

    boxes = []
    scores = []
    class_ids = []

    for det in preds:

        cx, cy, bw, bh = det[:4]
        class_scores = det[4:]

        cls_id = int(np.argmax(class_scores))
        score = class_scores[cls_id]

        if score < 0.55:
            continue

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        x1 = (x1 - dx) / scale
        y1 = (y1 - dy) / scale
        x2 = (x2 - dx) / scale
        y2 = (y2 - dy) / scale

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        boxes.append([x1, y1, x2-x1, y2-y1])
        scores.append(float(score))
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.15, 0.5)

    results = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            results.append((x, y, x+w_box, y+h_box, scores[i], class_ids[i]))

    return results

# =============================
# 摄像头（关键修复）
# =============================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

print("\nS: detect+save | A: upload | Q: quit\n")

last_saved = None

while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    # ⭐ 正方形显示
    square = center_crop(frame)
    square = cv2.resize(square, (720, 720))
    cv2.imshow("Camera", square)

    key = cv2.waitKey(1) & 0xFF

    # =============================
    # 检测
    # =============================
    if key == ord('s'):

        print(">>> Detect")

        img, scale, dx, dy = preprocess(frame)

        outputs = session.run(None, {input_name: img})
        pred = outputs[0]

        cropped = center_crop(frame)
        h, w = cropped.shape[:2]

        boxes = postprocess(pred, w, h, scale, dx, dy)

        result = cropped.copy()

        for x1, y1, x2, y2, score, cls_id in boxes:
            name = CLASS_NAMES[cls_id]

            cv2.rectangle(result, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(result, f"{name}:{score:.2f}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        # ⭐ 强制正方形显示
        result_show = cv2.resize(result, (720, 720))
        cv2.imshow("Detection", result_show)

        filename = f"/home/pi/pcb_project/capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

        last_saved = filename

        print("Saved:", filename)

    # =============================
    # 上传
    # =============================
    elif key == ord('a'):

        if last_saved is None:
            print("⚠️ 先按 S")
            continue

        try:
            with open(last_saved, 'rb') as f:
                files = {'file': f}
                response = requests.post(PC_URL, files=files)

            print("Upload:", response.text)

        except Exception as e:
            print("Upload failed:", e)

    # =============================
    # 退出
    # =============================
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
