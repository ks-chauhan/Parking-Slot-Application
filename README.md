# 🚗 Smart Parking Occupancy Detection System

## Overview

This project is an AI-powered Smart Parking Occupancy Detection System built using:

- PyTorch
- ResNet18
- Streamlit
- OpenCV

The system allows users to:

- Upload parking lot images
- Manually mark parking slots
- Run AI inference on each parking slot
- Detect whether slots are occupied or empty
- View parking statistics in real time

---

## Features

- Interactive parking slot annotation
- Deep learning-based occupancy classification
- Real-time parking statistics
- Scalable architecture for different parking layouts
- Modular inference pipeline
- Hugging Face hosted model weights

---

## Project Architecture

```text
Parking Lot Image
        ↓
User Draws Parking Slots
        ↓
Crop Individual Slots
        ↓
ResNet18 Classification Model
        ↓
Occupied / Empty Prediction
        ↓
Visualization + Statistics
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Deep Learning | PyTorch |
| Model | ResNet18 |
| Image Processing | OpenCV |
| Model Hosting | Hugging Face |

---

## Dataset

The model was trained using the CNR Parking Dataset containing parking slot images classified into:

- busy
- free

---

## Model Performance

| Metric | Result |
|---|---|
| Validation Accuracy | ~99% |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/ks-chauhan/Parking-Slot-Application/tree/main
cd Parking-Slot-Application
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
Python Version - 3.11

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Project Structure

```text
project/
│
├── app.py
├── parking_model.py
├── parking_slots.json
├── requirements.txt
└── README.md
```

---

## Model Weights

The trained ResNet18 model weights are hosted on Hugging Face:

https://huggingface.co/the-kshitij-chauhan/Parking-Slot-Model

---

## Future Improvements

- Real-time video stream support
- Automatic parking slot detection
- Multi-camera support
- Cloud deployment
- Mobile application integration
- Parking analytics dashboard