"""
Hand Detector Sidecar Server for 指文字道場

Runs a local Flask server that:
1. Captures webcam feed via OpenCV
2. Detects hand landmarks via MediaPipe Tasks API (0.10.x+)
3. Recognizes Japanese finger spelling (あいうえお)
4. Serves results via HTTP GET /landmarks
5. Serves camera frame as JPEG via GET /frame

Usage:
    pip install -r requirements.txt
    python3 hand_detector_server.py

The Godot game polls http://127.0.0.1:8765/landmarks and /frame
"""

import os
import threading
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, Response

app = Flask(__name__)

# ── MediaPipe Tasks API setup ────────────────────────────────────────────────

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Download hand landmarker model if not present
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded!")

# Shared state
current_state = {
    "detected": False,
    "landmarks": [],
    "recognized": "",
}
current_frame_jpg = None  # JPEG bytes of latest camera frame
state_lock = threading.Lock()
frame_lock = threading.Lock()


def is_finger_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_thumb_extended(landmarks):
    return landmarks[4].x < landmarks[3].x


def recognize_sign(landmarks):
    """
    JSL finger spelling (Phase 1: あいうえお)
    - あ: Fist (all closed)
    - い: Pinky only
    - う: Index + middle (peace)
    - え: Index only (pointing)
    - お: All fingers open
    """
    thumb = is_thumb_extended(landmarks)
    index = is_finger_extended(landmarks, 8, 6)
    middle = is_finger_extended(landmarks, 12, 10)
    ring = is_finger_extended(landmarks, 16, 14)
    pinky = is_finger_extended(landmarks, 20, 18)

    extended_count = sum([index, middle, ring, pinky])

    if extended_count == 0 and not thumb:
        return "あ"
    if pinky and not index and not middle and not ring:
        return "い"
    if index and middle and not ring and not pinky:
        return "う"
    if index and not middle and not ring and not pinky:
        return "え"
    if index and middle and ring and pinky:
        return "お"

    return ""


def camera_loop():
    global current_state, current_frame_jpg

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

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

        # Detect hands
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        # Draw hand landmarks on frame for visualization
        display_frame = frame.copy()

        with state_lock:
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand = result.hand_landmarks[0]
                lm_list = []
                h, w = frame.shape[:2]
                points = []
                for lm in hand:
                    lm_list.append({"x": lm.x, "y": lm.y, "z": lm.z})
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append((px, py))
                    cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)

                # Draw connections
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                for c in connections:
                    if c[0] < len(points) and c[1] < len(points):
                        cv2.line(display_frame, points[c[0]], points[c[1]], (0, 200, 0), 2)

                recognized = recognize_sign(hand)
                if recognized:
                    cv2.putText(display_frame, recognized, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

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

        # Encode frame as JPEG
        _, jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            current_frame_jpg = jpg.tobytes()


@app.route("/landmarks")
def get_landmarks():
    with state_lock:
        return jsonify(current_state)


@app.route("/frame")
def get_frame():
    """Return latest camera frame as JPEG image."""
    with frame_lock:
        if current_frame_jpg is None:
            return Response("No frame", status=503)
        return Response(current_frame_jpg, mimetype="image/jpeg")


@app.route("/stream")
def stream():
    """MJPEG stream for browser preview."""
    def generate():
        while True:
            with frame_lock:
                if current_frame_jpg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           current_frame_jpg + b"\r\n")
            import time
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return """
    <h1>🤟 指文字道場 - Hand Detector Server</h1>
    <p>GET <a href="/landmarks">/landmarks</a> — hand landmark JSON</p>
    <p>GET <a href="/frame">/frame</a> — latest camera frame (JPEG)</p>
    <p>GET <a href="/stream">/stream</a> — live MJPEG stream (open in browser)</p>
    <p>GET <a href="/health">/health</a> — health check</p>
    """


if __name__ == "__main__":
    print("🥋 指文字道場 Hand Detector Server")
    print("   http://127.0.0.1:8765")
    print("   ブラウザで /stream を開くとカメラプレビューが見れます")
    print("   Ctrl+C で終了")
    print()

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    app.run(host="127.0.0.1", port=8765, debug=False)
