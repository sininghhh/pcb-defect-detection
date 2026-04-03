import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response

# ================= CONFIG =================

MODEL_PATH = "/home/pi/pcb_project/models/best_416.onnx"

SAVE_DIR = "/home/pi/pcb_project/results/screenshots"

CLASS_NAMES = [
    "copper",
    "mousebite",
    "open",
    "pin-hole",
    "short",
    "spur"
]

CONF_TH = 0.35
CAM_INDEX = 0
FRAME_SKIP = 2
JPEG_QUALITY = 70

# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)

print("Loading model:", MODEL_PATH)

# ===== ONNX Runtime Optimization =====

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

sess = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

inp = sess.get_inputs()[0]
inp_name = inp.name
IMG_SIZE = inp.shape[2]

print("Model input:", IMG_SIZE)

# ===== Statistics =====

frame_count = 0
start_time = time.time()
last_save = 0

# ===== Camera =====

cap = cv2.VideoCapture(CAM_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ===== Letterbox =====

def letterbox(im, new_shape):

    h, w = im.shape[:2]

    r = min(new_shape / h, new_shape / w)

    nh, nw = int(h * r), int(w * r)

    im = cv2.resize(im, (nw, nh))

    pad_h = new_shape - nh
    pad_w = new_shape - nw

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114,114,114)
    )

    return im, r, left, top

# ===== Preprocess =====

def preprocess(frame):

    img, r, pad_x, pad_y = letterbox(frame, IMG_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))[None]

    return img, r, pad_x, pad_y

# ===== Scale Box =====

def scale_box(box, r, pad_x, pad_y):

    x1,y1,x2,y2 = box

    x1 = (x1 - pad_x) / r
    x2 = (x2 - pad_x) / r
    y1 = (y1 - pad_y) / r
    y2 = (y2 - pad_y) / r

    return [x1,y1,x2,y2]

# ===== Inference =====

def infer(frame):

    img, r, pad_x, pad_y = preprocess(frame)

    pred = sess.run(None, {inp_name: img})[0]

    dets = []

    for x1,y1,x2,y2,score,cls_id in pred[0]:

        if score < CONF_TH:
            continue

        cls_id = int(cls_id)

        box = scale_box([x1,y1,x2,y2], r, pad_x, pad_y)

        dets.append([*box, score, cls_id])

    return dets

# ===== Draw =====

def draw(frame, dets):

    h, w = frame.shape[:2]

    for x1,y1,x2,y2,score,cls_id in dets:

        name = CLASS_NAMES[cls_id]

        x1 = int(max(0, min(w-1, x1)))
        y1 = int(max(0, min(h-1, y1)))
        x2 = int(max(0, min(w-1, x2)))
        y2 = int(max(0, min(h-1, y2)))

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

# ===== Frame Generator =====

last_dets = []

def gen_frames():

    global frame_count, last_dets, last_save

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1

        t0 = time.time()

        if frame_count % FRAME_SKIP == 0:
            last_dets = infer(frame)

        infer_time = (time.time() - t0) * 1000

        draw(frame, last_dets)

        runtime = time.time() - start_time
        fps = frame_count / runtime if runtime > 0 else 0

        cv2.putText(
            frame,
            f"FPS:{fps:.2f}",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            f"infer:{infer_time:.1f}ms",
            (10,55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        # ===== Screenshot =====

        if len(last_dets) > 0 and time.time() - last_save > 2:

            ts = int(time.time()*1000)

            path = f"{SAVE_DIR}/det_{ts}.jpg"

            cv2.imwrite(path, frame)

            last_save = time.time()

        # ===== Encode =====

        ret, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        frame_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')

# ===== Flask =====

@app.route('/')
def index():

    return "<h2>PCB Defect Detection System</h2><a href='/video'>Video Stream</a>"

@app.route('/video')
def video():

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ===== Run =====

if __name__ == "__main__":

    print("Starting server...")

    app.run(host="0.0.0.0", port=5000, threaded=True)
