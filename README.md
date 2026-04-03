# A PCB Defect Detection System Based on YOLOv11

This repository contains the implementation of a PCB defect detection system based on YOLOv11, including model training, ONNX deployment, Raspberry Pi inference, and Grad-CAM visualization.

---

## 🎓 Thesis Information

This project is developed as part of an undergraduate thesis:

**Title:** A Study on Deep Learning-Based PCB Defect Detection Methods
**Author:** Tianci Huang
**Institution:** Huzhou University
**Year:** 2026

---

## 📌 Project Overview

This project proposes a PCB defect detection system based on YOLOv11, aiming to improve detection accuracy, deployment efficiency, and model interpretability in industrial scenarios.

The system focuses on detecting six common PCB defects:

* Missing hole
* Mouse bite
* Open circuit
* Short circuit
* Spurious copper
* Spur

The system integrates:

* YOLOv11 model training (PC side)
* ONNX model export and optimization
* Raspberry Pi real-time detection
* Image transmission via HTTP
* Grad-CAM visualization for interpretability

---

## 🧠 System Architecture

The system follows a pipeline:

**Training → ONNX Export → Edge Detection → Image Upload → Grad-CAM Analysis**

---

## 📂 Project Structure

```
pcb-defect-detection/
├── raspberry_pi/      # Edge-side detection scripts
├── pc_server/         # Image receiving server
├── grad_cam/          # Visualization modules
├── train/             # Training & export scripts
├── models/            # Trained weights & ONNX models
```

> Note: Large datasets and trained model weights are not included due to size limitations.

---

## ⚙️ Environment

### PC (Training)

* Python 3.11
* PyTorch
* Ultralytics
* OpenCV

### Raspberry Pi

* Python 3.13
* onnxruntime
* opencv-python
* numpy
* requests

---

## 🚀 Usage

### 1. Model Training

```bash
yolo detect train model=yolo11n.pt data=data.yaml epochs=100 imgsz=800
```

### 2. Export ONNX

```bash
yolo export model=best.pt format=onnx opset=12 imgsz=512
```

### 3. Raspberry Pi Detection

```bash
python detect_gray.py best_512.onnx
```

### 4. Start PC Server

```bash
python server.py
```

### 5. Grad-CAM Visualization

```bash
python monitor.py
```

> Please refer to the scripts in each subdirectory for detailed usage.

---

## 📸 Results

The system is capable of:

* Real-time PCB defect detection on Raspberry Pi
* Multi-scale ONNX inference
* Grad-CAM heatmap visualization for model interpretability

---

## 📊 Features

* Lightweight YOLOv11 deployment
* Real-time PCB defect detection
* Multi-scale ONNX inference
* Edge-cloud collaborative system
* Grad-CAM interpretability

---

## 📎 Notes

* Recommended ONNX input size: **512 / 600**
* Input size 800 improves small defect detection but increases latency

---

## 📜 License

This project is intended for academic and research purposes only.
