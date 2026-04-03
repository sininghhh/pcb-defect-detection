import cv2
import numpy as np
import onnxruntime as ort
import sys

# ✅ 已替换成你的手机IP
PHONE_STREAM = "http://10.38.166.145:8080/video"

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

MODEL_PATH = sys.argv[1]

print("Loading model:", MODEL_PATH)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
IMG_SIZE = session.get_inputs()[0].shape[2]

print("Model input size:", IMG_SIZE)


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


def preprocess(img):
    img, scale, dx, dy = letterbox(img, IMG_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]

    return img, scale, dx, dy


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

        if score < 0.25:
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

        # 过滤小误检
        if (x2 - x1) < 12 or (y2 - y1) < 12:
            continue

        boxes.append([x1, y1, x2-x1, y2-y1])
        scores.append(float(score))
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    results = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            results.append((
                x,
                y,
                x+w_box,
                y+h_box,
                scores[i],
                class_ids[i]
            ))

    return results


# 📱 打开手机摄像头流
cap = cv2.VideoCapture(PHONE_STREAM)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Cannot open phone stream")
    exit()

print("\n📱 Phone camera connected")
print("Press Q to quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame read failed")
        break

    h, w = frame.shape[:2]

    img, scale, dx, dy = preprocess(frame)

    outputs = session.run(None, {input_name: img})
    pred = outputs[0]

    boxes = postprocess(pred, w, h, scale, dx, dy)

    result = frame.copy()

    for x1, y1, x2, y2, score, cls_id in boxes:

        name = CLASS_NAMES[cls_id]

        cv2.rectangle(result, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            result,
            f"{name}:{score:.2f}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Mobile Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
