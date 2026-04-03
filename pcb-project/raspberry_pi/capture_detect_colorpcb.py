import cv2
import numpy as np
import onnxruntime as ort
import sys
import time

CLASS_NAMES = [
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
    "mouse_bite",
    "spur"
]

MODEL = sys.argv[1]

session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
IMG_SIZE = session.get_inputs()[0].shape[2]

print("Model size:", IMG_SIZE)


# ===================== 图像增强 =====================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ===================== ROI裁剪 =====================
def crop_center(img):
    h, w = img.shape[:2]
    return img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]


# ===================== Letterbox =====================
def letterbox(img, size):
    h, w = img.shape[:2]

    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized

    return canvas, scale


# ===================== 预处理 =====================
def preprocess(img):
    img = apply_clahe(img)
    img, scale = letterbox(img, IMG_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    return img, scale


# ===================== 后处理 =====================
def postprocess(pred, w, h, scale):

    if pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(0, 2, 1)

    boxes = []
    scores = []
    class_ids = []

    for det in pred[0]:

        x, y, bw, bh = det[:4]

        cls_scores = det[4:]
        cls_id = np.argmax(cls_scores)
        score = cls_scores[cls_id]

        # ↓ 降低阈值（关键）
        if score < 0.45:
            continue

        # scale反变换
        x1 = int((x - bw / 2) / scale)
        y1 = int((y - bh / 2) / scale)
        x2 = int((x + bw / 2) / scale)
        y2 = int((y + bh / 2) / scale)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))
        class_ids.append(cls_id)

    idx = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.5)

    results = []

    if len(idx) > 0:
        for i in idx.flatten():
            x, y, bw, bh = boxes[i]
            results.append((x, y, x + bw, y + bh, scores[i], class_ids[i]))

    return results


# ===================== 摄像头 =====================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 固定曝光（非常关键）
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -8)

print("Press S detect  Q quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ROI裁剪（关键）
    #frame = crop_center(frame)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord("s"):

        start = time.time()

        img, scale = preprocess(frame)

        outputs = session.run(None, {input_name: img})
        pred = outputs[0]

        h, w = frame.shape[:2]

        boxes = postprocess(pred, w, h, scale)

        end = time.time()
        fps = 1 / (end - start)

        result = frame.copy()

        for x1, y1, x2, y2, score, cls_id in boxes:
            name = CLASS_NAMES[cls_id]

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{name}:{score:.2f}"
            cv2.putText(result, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.putText(result, f"FPS:{fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detection", result)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
