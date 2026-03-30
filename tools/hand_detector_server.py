"""
Hand Detector Sidecar Server for 指文字道場

Runs a local Flask server that:
1. Captures webcam feed via OpenCV
2. Detects hand landmarks via MediaPipe
3. Recognizes Japanese finger spelling (あいうえお)
4. Serves results via HTTP GET /landmarks

Usage:
    pip install -r requirements.txt
    python hand_detector_server.py

The Godot game polls http://127.0.0.1:8765/landmarks
"""

import json
import threading
import cv2
import mediapipe as mp
from flask import Flask, jsonify

app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Shared state (updated by camera thread)
current_state = {
    "detected": False,
    "landmarks": [],
    "recognized": "",
}
state_lock = threading.Lock()


def is_finger_extended(landmarks, tip_idx, pip_idx):
    """Check if a finger is extended (tip above pip in image coords)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_thumb_extended(landmarks):
    """Check if thumb is extended (tip left of IP joint for right hand)."""
    return landmarks[4].x < landmarks[3].x


def recognize_sign(landmarks):
    """
    Recognize Japanese finger spelling from hand landmarks.
    Phase 1: あいうえお only.

    JSL finger spelling reference:
    - あ (a): Fist (all fingers closed)
    - い (i): Pinky extended only
    - う (u): Index + middle extended (peace/victory)
    - え (e): Index extended only (pointing)
    - お (o): All fingers extended (open hand)
    """
    thumb = is_thumb_extended(landmarks)
    index = is_finger_extended(landmarks, 8, 6)
    middle = is_finger_extended(landmarks, 12, 10)
    ring = is_finger_extended(landmarks, 16, 14)
    pinky = is_finger_extended(landmarks, 20, 18)

    extended_count = sum([index, middle, ring, pinky])

    # あ: fist (no fingers extended)
    if extended_count == 0 and not thumb:
        return "あ"

    # い: only pinky
    if pinky and not index and not middle and not ring:
        return "い"

    # う: index + middle (peace sign)
    if index and middle and not ring and not pinky:
        return "う"

    # え: only index (pointing)
    if index and not middle and not ring and not pinky:
        return "え"

    # お: all fingers open
    if index and middle and ring and pinky:
        return "お"

    return ""


def camera_loop():
    """Background thread: capture webcam + detect hands."""
    global current_state

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️  カメラが開けません！")
        return

    print("📷 カメラ起動OK")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        with state_lock:
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm_list = []
                for lm in hand.landmark:
                    lm_list.append({"x": lm.x, "y": lm.y, "z": lm.z})

                recognized = recognize_sign(hand.landmark)

                current_state = {
                    "detected": True,
                    "landmarks": lm_list,
                    "recognized": recognized,
                }
            else:
                current_state = {
                    "detected": False,
                    "landmarks": [],
                    "recognized": "",
                }


@app.route("/landmarks")
def get_landmarks():
    with state_lock:
        return jsonify(current_state)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return """
    <h1>🤟 指文字道場 - Hand Detector Server</h1>
    <p>GET <a href="/landmarks">/landmarks</a> — current hand data</p>
    <p>GET <a href="/health">/health</a> — health check</p>
    """


if __name__ == "__main__":
    print("🥋 指文字道場 Hand Detector Server")
    print("   http://127.0.0.1:8765")
    print("   Ctrl+C で終了")
    print()

    # Start camera in background thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Start Flask server
    app.run(host="127.0.0.1", port=8765, debug=False)
