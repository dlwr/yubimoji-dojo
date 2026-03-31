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


def finger_curl(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx):
    """How curled a finger is. Higher = more curled."""
    tip_y = landmarks[tip_idx].y
    dip_y = landmarks[dip_idx].y
    pip_y = landmarks[pip_idx].y
    mcp_y = landmarks[mcp_idx].y
    # If tip is below mcp (in image coords, higher y = lower), finger is curled
    return tip_y - mcp_y


def get_finger_states(landmarks):
    """Get detailed finger state info for debugging and recognition."""
    thumb = is_thumb_extended(landmarks)
    index = is_finger_extended(landmarks, 8, 6)
    middle = is_finger_extended(landmarks, 12, 10)
    ring = is_finger_extended(landmarks, 16, 14)
    pinky = is_finger_extended(landmarks, 20, 18)

    # Curl amounts (positive = curled, negative = extended)
    index_curl = finger_curl(landmarks, 8, 7, 6, 5)
    middle_curl = finger_curl(landmarks, 12, 11, 10, 9)
    ring_curl = finger_curl(landmarks, 16, 15, 14, 13)
    pinky_curl = finger_curl(landmarks, 20, 19, 18, 17)

    # Is index finger bent/hooked? (tip below DIP but above PIP)
    index_bent = landmarks[8].y > landmarks[7].y and landmarks[8].y < landmarks[5].y

    # Fingers spread? (horizontal distance between index and pinky tips)
    spread = abs(landmarks[8].x - landmarks[20].x)

    return {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
        "index_bent": index_bent,
        "index_curl": round(index_curl, 3),
        "middle_curl": round(middle_curl, 3),
        "ring_curl": round(ring_curl, 3),
        "pinky_curl": round(pinky_curl, 3),
        "spread": round(spread, 3),
    }


def fingertip_distance(landmarks):
    """Average distance between all fingertip pairs (how clustered they are)."""
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    total = 0
    count = 0
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dx = landmarks[tips[i]].x - landmarks[tips[j]].x
            dy = landmarks[tips[i]].y - landmarks[tips[j]].y
            total += (dx * dx + dy * dy) ** 0.5
            count += 1
    return total / count if count > 0 else 999


def recognize_sign(landmarks):
    """
    JSL finger spelling (Phase 1: あいうえお)

    Correct JSL yubimoji forms:
    - あ (a): Fist with thumb extended to the side, palm forward
    - い (i): Pinky extended up, rest closed
    - う (u): Index + middle extended together (close), rest closed
    - え (e): All 5 fingers extended and together, palm forward (like showing a card)
    - お (o): All fingers curled to form a circle (fingertips clustered together)
    """
    state = get_finger_states(landmarks)
    thumb = state["thumb"]
    index = state["index"]
    middle = state["middle"]
    ring = state["ring"]
    pinky = state["pinky"]
    spread = state["spread"]

    extended_count = sum([index, middle, ring, pinky])
    tip_dist = fingertip_distance(landmarks)

    # お: all fingertips clustered together (making a circle/ball shape)
    # Check this first — fingertips very close regardless of "extended" state
    if tip_dist < 0.08:
        return "お"

    # あ: fist + thumb extended to the side (4 fingers closed, thumb open)
    if extended_count == 0 and thumb:
        return "あ"

    # い: pinky only extended, rest closed
    if pinky and not index and not middle and not ring:
        return "い"

    # う: index + middle extended, close together, ring + pinky closed
    if index and middle and not ring and not pinky:
        return "う"

    # え: all 5 fingers extended (palm forward, like showing a card)
    if extended_count >= 3 and thumb and index and middle:
        return "え"

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
                finger_info = get_finger_states(hand)
                if recognized:
                    cv2.putText(display_frame, recognized, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

                # Show finger states on frame for debugging
                debug_y = 90
                for fname in ["thumb", "index", "middle", "ring", "pinky"]:
                    val = finger_info[fname]
                    color = (0, 255, 0) if val else (0, 0, 255)
                    cv2.putText(display_frame, f"{fname}: {'O' if val else 'X'}",
                                (20, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    debug_y += 20

                current_state = {
                    "detected": True,
                    "landmarks": lm_list,
                    "recognized": recognized,
                    "fingers": finger_info,
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
