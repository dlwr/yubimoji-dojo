"""
Calibration Mode for 指文字道場

Records hand landmark data for each sign, then uses it for recognition.
Data is saved to calibration_data.json.

Usage:
    python3 calibration_server.py

Endpoints:
    GET  /                  - Web UI for calibration
    POST /record            - Start recording a sign {char: "あ", samples: 10}
    GET  /status            - Current recording status
    GET  /signs             - List all calibrated signs
    DELETE /signs/<char>    - Delete a sign's calibration data
    GET  /landmarks         - Hand detection + recognition using calibrated data
    GET  /frame             - Camera frame as JPEG
    GET  /stream            - MJPEG stream
    GET  /health            - Health check
"""

import json
import os
import threading
import time
import urllib.request
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# ── MediaPipe setup ──────────────────────────────────────────────────────────

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded!")

DATA_PATH = os.path.join(os.path.dirname(__file__), "calibration_data.json")

# ── Shared state ─────────────────────────────────────────────────────────────

current_landmarks = None  # Latest raw landmarks (list of 21 {x,y,z})
current_frame_jpg = None
state_lock = threading.Lock()
frame_lock = threading.Lock()

# Recording state
recording = {
    "active": False,
    "char": "",
    "target_samples": 0,
    "collected": 0,
    "countdown": 0,  # seconds before recording starts
    "samples": [],
}
recording_lock = threading.Lock()

# Calibration data: {char: [sample, sample, ...]}
# Each sample is a list of 21 landmarks normalized relative to wrist
calibration_data = {}


def load_calibration():
    global calibration_data
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            calibration_data = json.load(f)
        print(f"📂 Loaded calibration: {list(calibration_data.keys())}")
    else:
        calibration_data = {}


def save_calibration():
    with open(DATA_PATH, "w") as f:
        json.dump(calibration_data, f, ensure_ascii=False, indent=2)


def normalize_landmarks(landmarks):
    """
    Normalize 21 landmarks relative to wrist (index 0).
    This makes recognition independent of hand position in frame.
    Also scale by hand size (distance from wrist to middle finger MCP).
    """
    wrist = landmarks[0]
    # Use distance from wrist to middle MCP as scale reference
    mid_mcp = landmarks[9]
    scale = ((mid_mcp.x - wrist.x) ** 2 + (mid_mcp.y - wrist.y) ** 2) ** 0.5
    if scale < 0.001:
        scale = 0.001  # Avoid division by zero

    normalized = []
    for lm in landmarks:
        normalized.append({
            "x": (lm.x - wrist.x) / scale,
            "y": (lm.y - wrist.y) / scale,
            "z": (lm.z - wrist.z) / scale,
        })
    return normalized


def normalize_landmarks_from_dict(landmarks):
    """Same as above but from dict format (stored calibration data)."""
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    scale = ((mid_mcp["x"] - wrist["x"]) ** 2 + (mid_mcp["y"] - wrist["y"]) ** 2) ** 0.5
    if scale < 0.001:
        scale = 0.001

    normalized = []
    for lm in landmarks:
        normalized.append({
            "x": (lm["x"] - wrist["x"]) / scale,
            "y": (lm["y"] - wrist["y"]) / scale,
            "z": (lm["z"] - wrist["z"]) / scale,
        })
    return normalized


def landmark_distance(a, b):
    """Euclidean distance between two normalized landmark sets."""
    if len(a) != 21 or len(b) != 21:
        return 999
    total = 0
    for i in range(21):
        dx = a[i]["x"] - b[i]["x"]
        dy = a[i]["y"] - b[i]["y"]
        total += (dx * dx + dy * dy) ** 0.5
    return total / 21


def recognize_from_calibration(landmarks):
    """
    Compare current hand against all calibrated signs.
    Returns (best_char, confidence) or ("", 0).
    """
    if not calibration_data:
        return "", 0

    normalized = normalize_landmarks(landmarks)

    best_char = ""
    best_dist = 999

    for char, samples in calibration_data.items():
        # Average distance to all samples for this char
        distances = []
        for sample in samples:
            norm_sample = normalize_landmarks_from_dict(sample)
            dist = landmark_distance(normalized, norm_sample)
            distances.append(dist)

        avg_dist = sum(distances) / len(distances) if distances else 999
        if avg_dist < best_dist:
            best_dist = avg_dist
            best_char = char

    # Confidence threshold - lower distance = better match
    # Typical good match: < 0.3, okay: < 0.5, bad: > 0.5
    if best_dist < 0.5:
        confidence = max(0, min(100, int((1 - best_dist / 0.5) * 100)))
        return best_char, confidence

    return "", 0


# ── Camera loop ──────────────────────────────────────────────────────────────

def camera_loop():
    global current_landmarks, current_frame_jpg

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

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        display_frame = frame.copy()
        detected = False
        recognized_char = ""
        confidence = 0

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            detected = True
            h, w = frame.shape[:2]

            # Draw landmarks
            points = []
            lm_list = []
            for lm in hand:
                lm_list.append({"x": lm.x, "y": lm.y, "z": lm.z})
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)

            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
            ]
            for c in connections:
                if c[0] < len(points) and c[1] < len(points):
                    cv2.line(display_frame, points[c[0]], points[c[1]], (0, 200, 0), 2)

            with state_lock:
                current_landmarks = hand

            # Recording mode
            with recording_lock:
                if recording["active"] and recording["countdown"] <= 0:
                    normalized = normalize_landmarks(hand)
                    raw = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
                    recording["samples"].append(raw)
                    recording["collected"] += 1

                    # Show recording indicator
                    cv2.putText(display_frame, f"REC {recording['collected']}/{recording['target_samples']}",
                                (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if recording["collected"] >= recording["target_samples"]:
                        # Save samples
                        char = recording["char"]
                        calibration_data[char] = recording["samples"]
                        save_calibration()
                        print(f"✅ Saved {len(recording['samples'])} samples for '{char}'")
                        recording["active"] = False

            # Recognition (when not recording)
            with recording_lock:
                is_recording = recording["active"]

            if not is_recording:
                recognized_char, confidence = recognize_from_calibration(hand)
                if recognized_char:
                    cv2.putText(display_frame, f"{recognized_char} ({confidence}%)",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        else:
            with state_lock:
                current_landmarks = None

        # Show countdown if active
        with recording_lock:
            if recording["active"] and recording["countdown"] > 0:
                cv2.putText(display_frame, str(int(recording["countdown"])),
                            (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)

        # Encode frame
        _, jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            current_frame_jpg = jpg.tobytes()


def countdown_thread():
    """Handles countdown before recording starts."""
    while True:
        time.sleep(1)
        with recording_lock:
            if recording["active"] and recording["countdown"] > 0:
                recording["countdown"] -= 1


# ── API endpoints ────────────────────────────────────────────────────────────

@app.route("/record", methods=["POST"])
def start_recording():
    data = request.json
    char = data.get("char", "")
    samples = data.get("samples", 15)
    countdown = data.get("countdown", 3)

    if not char:
        return jsonify({"error": "char is required"}), 400

    with recording_lock:
        recording["active"] = True
        recording["char"] = char
        recording["target_samples"] = samples
        recording["collected"] = 0
        recording["countdown"] = countdown
        recording["samples"] = []

    return jsonify({"ok": True, "char": char, "samples": samples, "countdown": countdown})


@app.route("/status")
def get_status():
    with recording_lock:
        return jsonify({
            "recording": recording["active"],
            "char": recording["char"],
            "collected": recording["collected"],
            "target": recording["target_samples"],
            "countdown": recording["countdown"],
        })


@app.route("/signs")
def get_signs():
    return jsonify({
        char: len(samples) for char, samples in calibration_data.items()
    })


@app.route("/signs/<char>", methods=["DELETE"])
def delete_sign(char):
    if char in calibration_data:
        del calibration_data[char]
        save_calibration()
        return jsonify({"ok": True, "deleted": char})
    return jsonify({"error": "not found"}), 404


@app.route("/landmarks")
def get_landmarks():
    with state_lock:
        if current_landmarks is None:
            return jsonify({"detected": False, "landmarks": [], "recognized": "", "confidence": 0})

        hand = current_landmarks
        lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]

        with recording_lock:
            is_rec = recording["active"]

        if is_rec:
            return jsonify({"detected": True, "landmarks": lm_list, "recognized": "", "confidence": 0, "recording": True})

        char, conf = recognize_from_calibration(hand)
        return jsonify({"detected": True, "landmarks": lm_list, "recognized": char, "confidence": conf})


@app.route("/frame")
def get_frame():
    with frame_lock:
        if current_frame_jpg is None:
            return Response("No frame", status=503)
        return Response(current_frame_jpg, mimetype="image/jpeg")


@app.route("/stream")
def stream():
    def generate():
        while True:
            with frame_lock:
                if current_frame_jpg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" +
                           current_frame_jpg + b"\r\n")
            time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "signs": list(calibration_data.keys())})


@app.route("/")
def index():
    signs_html = ""
    for char, samples in calibration_data.items():
        signs_html += f"<li><b>{char}</b> — {len(samples)} samples</li>"
    if not signs_html:
        signs_html = "<li>まだ登録されていません</li>"

    return f"""
    <html><head><meta charset="utf-8"><title>指文字道場 キャリブレーション</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #f4a261; }}
        input, button {{ font-size: 18px; padding: 8px 16px; margin: 4px; }}
        button {{ background: #f4a261; border: none; cursor: pointer; border-radius: 4px; }}
        button:hover {{ background: #e76f51; }}
        #status {{ font-size: 24px; margin: 20px 0; color: #e9c46a; }}
        img {{ border: 2px solid #333; border-radius: 8px; }}
        ul {{ font-size: 18px; }}
    </style></head><body>
    <h1>🥋 指文字道場 キャリブレーション</h1>

    <h2>カメラ</h2>
    <img src="/stream" width="640" height="480">

    <h2>指文字を録画</h2>
    <input type="text" id="char" placeholder="文字 (例: あ)" maxlength="2" style="width:80px">
    <button onclick="record()">録画開始 (3秒後)</button>
    <div id="status">待機中</div>

    <h2>登録済みの指文字</h2>
    <ul>{signs_html}</ul>
    <button onclick="location.reload()">更新</button>

    <script>
    async function record() {{
        const char = document.getElementById('char').value;
        if (!char) {{ alert('文字を入力してください'); return; }}
        document.getElementById('status').innerText = '準備中...';
        const res = await fetch('/record', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{char, samples: 15, countdown: 3}})
        }});
        // Poll status
        const poll = setInterval(async () => {{
            const s = await (await fetch('/status')).json();
            if (s.countdown > 0) {{
                document.getElementById('status').innerText = `${{s.countdown}}秒後に開始...`;
            }} else if (s.recording) {{
                document.getElementById('status').innerText = `録画中: ${{s.collected}}/${{s.target}}`;
            }} else {{
                document.getElementById('status').innerText = `✅ 「${{char}}」の録画完了！`;
                clearInterval(poll);
                setTimeout(() => location.reload(), 1000);
            }}
        }}, 500);
    }}
    </script></body></html>
    """


if __name__ == "__main__":
    load_calibration()

    print("🥋 指文字道場 キャリブレーション Server")
    print("   http://127.0.0.1:8765")
    print("   ブラウザで開いてキャリブレーションしてください")
    print("   Ctrl+C で終了")
    print()

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    cd_thread = threading.Thread(target=countdown_thread, daemon=True)
    cd_thread.start()

    app.run(host="127.0.0.1", port=8765, debug=False)
