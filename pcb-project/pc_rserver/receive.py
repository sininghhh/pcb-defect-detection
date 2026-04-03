from flask import Flask, request
import os
import time

app = Flask(__name__)

SAVE_DIR = r"D:\deeppcb\10668PCB\grad-cam-picture"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = f"pcb_{int(time.time())}.jpg"
    path = os.path.join(SAVE_DIR, filename)
    file.save(path)

    print("Saved:", path)
    return "OK"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
