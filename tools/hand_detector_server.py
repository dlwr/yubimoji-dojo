"""
Hand Detector Server for 指文字道場 (Time-Series Edition)

Records and recognizes hand signs using time-series landmark data.
Works for both static signs (finger spelling) and dynamic signs (sign language).

Uses DTW (Dynamic Time Warping) for motion-tolerant matching.

Usage:
    pip install -r requirements.txt
    python3 hand_detector_server.py

Endpoints:
    GET  /                  - Web UI for calibration
    POST /record            - Record a sign {char: "あ", duration: 2.0, countdown: 3}
    GET  /status            - Current recording status
    GET  /signs             - List all calibrated signs with metadata
    DELETE /signs/<char>    - Delete a sign
    GET  /landmarks         - Hand detection + recognition (Godot polls this)
    GET  /frame             - Camera frame as JPEG
    GET  /stream            - MJPEG stream
    GET  /health            - Health check
"""

import json
import math
import os
import threading
import time
import urllib.request

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

current_landmarks_raw = None  # Latest MediaPipe landmarks object
current_frame_jpg = None
state_lock = threading.Lock()
frame_lock = threading.Lock()

# Ring buffer of recent frames for recognition (last ~2 seconds)
HISTORY_MAX = 30  # ~2 seconds at 15fps
landmark_history = []  # list of normalized landmark snapshots
history_lock = threading.Lock()

# Recording state
recording = {
    "active": False,
    "char": "",
    "duration": 2.0,      # seconds to record
    "countdown": 3,        # seconds before start
    "start_time": 0,
    "frames": [],          # list of normalized landmark snapshots
    "done": False,
    "has_motion": False,   # auto-detected
}
recording_lock = threading.Lock()

# Calibration data: {char: {frames: [...], has_motion: bool}}
# Each frame is a list of 21 normalized {x,y,z} landmarks
calibration_data = {}


def load_calibration():
    global calibration_data
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            calibration_data = json.load(f)
        print(f"📂 Loaded: {list(calibration_data.keys())} ({len(calibration_data)} signs)")
    else:
        calibration_data = {}


def save_calibration():
    with open(DATA_PATH, "w") as f:
        json.dump(calibration_data, f, ensure_ascii=False)


# ── Normalization ────────────────────────────────────────────────────────────

def normalize(landmarks):
    """Normalize 21 landmarks relative to wrist, scaled by hand size."""
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    scale = math.sqrt(
        (mid_mcp.x - wrist.x) ** 2 +
        (mid_mcp.y - wrist.y) ** 2
    )
    if scale < 0.001:
        scale = 0.001

    return [{
        "x": round((lm.x - wrist.x) / scale, 4),
        "y": round((lm.y - wrist.y) / scale, 4),
        "z": round((lm.z - wrist.z) / scale, 4),
    } for lm in landmarks]


def normalize_dict(landmarks):
    """Same but from dict format (stored data)."""
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    scale = math.sqrt(
        (mid_mcp["x"] - wrist["x"]) ** 2 +
        (mid_mcp["y"] - wrist["y"]) ** 2
    )
    if scale < 0.001:
        scale = 0.001

    return [{
        "x": round((lm["x"] - wrist["x"]) / scale, 4),
        "y": round((lm["y"] - wrist["y"]) / scale, 4),
        "z": round((lm["z"] - wrist["z"]) / scale, 4),
    } for lm in landmarks]


# ── Distance / DTW ───────────────────────────────────────────────────────────

def frame_distance(a, b):
    """Euclidean distance between two 21-landmark frames."""
    total = 0
    for i in range(min(len(a), len(b), 21)):
        dx = a[i]["x"] - b[i]["x"]
        dy = a[i]["y"] - b[i]["y"]
        total += math.sqrt(dx * dx + dy * dy)
    return total / 21


def dtw_distance(seq_a, seq_b):
    """
    Dynamic Time Warping distance between two sequences of landmark frames.
    Handles different speeds of the same motion.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 999

    # Downsample longer sequences for performance
    MAX_LEN = 30
    if n > MAX_LEN:
        indices = [int(i * (n - 1) / (MAX_LEN - 1)) for i in range(MAX_LEN)]
        seq_a = [seq_a[i] for i in indices]
        n = MAX_LEN
    if m > MAX_LEN:
        indices = [int(i * (m - 1) / (MAX_LEN - 1)) for i in range(MAX_LEN)]
        seq_b = [seq_b[i] for i in indices]
        m = MAX_LEN

    # DTW matrix
    dtw = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = frame_distance(seq_a[i - 1], seq_b[j - 1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

    return dtw[n][m] / max(n, m)


def detect_motion(frames):
    """Detect if a recorded sequence has significant motion."""
    if len(frames) < 5:
        return False
    # Compare first and last frame
    dist = frame_distance(frames[0], frames[-1])
    # Also check max displacement of any frame from first
    max_dist = max(frame_distance(frames[0], f) for f in frames)
    return max_dist > 0.4


def recognize(history_frames):
    """
    Match recent landmark history against calibration data.
    Uses single-frame matching for static signs, DTW for dynamic signs.
    Returns (char, confidence) or ("", 0).
    """
    if not calibration_data or not history_frames:
        return "", 0

    best_char = ""
    best_score = 999

    for char, data in calibration_data.items():
        cal_frames = data["frames"]
        has_motion = data.get("has_motion", False)

        if has_motion:
            # Dynamic sign: use DTW on full history
            dist = dtw_distance(history_frames, cal_frames)
        else:
            # Static sign: compare latest frame against middle frame of calibration
            mid_idx = len(cal_frames) // 2
            if history_frames:
                dist = frame_distance(history_frames[-1], cal_frames[mid_idx])
            else:
                continue

        if dist < best_score:
            best_score = dist
            best_char = char

    # Thresholds
    threshold = 0.6 if calibration_data.get(best_char, {}).get("has_motion") else 0.4
    if best_score < threshold:
        confidence = max(0, min(100, int((1 - best_score / threshold) * 100)))
        return best_char, confidence

    return "", 0


# ── Camera loop ──────────────────────────────────────────────────────────────

def camera_loop():
    global current_landmarks_raw, current_frame_jpg

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
        h, w = frame.shape[:2]

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]

            # Draw landmarks
            points = []
            for lm in hand:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)

            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
            ]
            for a, b in connections:
                if a < len(points) and b < len(points):
                    cv2.line(display_frame, points[a], points[b], (0, 200, 0), 2)

            normalized = normalize(hand)

            with state_lock:
                current_landmarks_raw = hand

            # Add to history ring buffer
            with history_lock:
                landmark_history.append(normalized)
                while len(landmark_history) > HISTORY_MAX:
                    landmark_history.pop(0)

            # Recording
            with recording_lock:
                if recording["active"] and recording["countdown"] <= 0:
                    recording["frames"].append(normalized)
                    elapsed = time.time() - recording["start_time"]
                    remaining = recording["duration"] - elapsed

                    cv2.putText(display_frame,
                                f"REC [{recording['char']}] {len(recording['frames'])}f ({remaining:.1f}s)",
                                (w - 380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if elapsed >= recording["duration"]:
                        # Finish recording
                        frames = recording["frames"]
                        has_motion = detect_motion(frames)
                        calibration_data[recording["char"]] = {
                            "frames": frames,
                            "has_motion": has_motion,
                        }
                        save_calibration()
                        motion_str = "動き" if has_motion else "静止"
                        print(f"✅ '{recording['char']}' saved: {len(frames)} frames ({motion_str})")
                        recording["active"] = False
                        recording["done"] = True

            # Recognition (when not recording)
            with recording_lock:
                is_rec = recording["active"]

            if not is_rec:
                with history_lock:
                    hist_copy = list(landmark_history)
                char, conf = recognize(hist_copy)
                if char:
                    cv2.putText(display_frame, f"{char} ({conf}%)",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        else:
            with state_lock:
                current_landmarks_raw = None

        # Countdown display
        with recording_lock:
            if recording["active"] and recording["countdown"] > 0:
                cv2.putText(display_frame, str(int(math.ceil(recording["countdown"]))),
                            (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)

        _, jpg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            current_frame_jpg = jpg.tobytes()

        time.sleep(0.033)  # ~30fps cap


def countdown_thread():
    while True:
        time.sleep(0.5)
        with recording_lock:
            if recording["active"] and recording["countdown"] > 0:
                recording["countdown"] -= 0.5
                if recording["countdown"] <= 0:
                    recording["start_time"] = time.time()


# ── API ──────────────────────────────────────────────────────────────────────

@app.route("/record", methods=["POST"])
def start_recording():
    data = request.json
    char = data.get("char", "")
    duration = data.get("duration", 2.0)
    countdown = data.get("countdown", 3)

    if not char:
        return jsonify({"error": "char is required"}), 400

    with recording_lock:
        recording.update({
            "active": True,
            "char": char,
            "duration": duration,
            "countdown": countdown,
            "start_time": 0,
            "frames": [],
            "done": False,
            "has_motion": False,
        })

    return jsonify({"ok": True, "char": char, "duration": duration})


@app.route("/status")
def get_status():
    with recording_lock:
        return jsonify({
            "recording": recording["active"],
            "done": recording["done"],
            "char": recording["char"],
            "frames": len(recording["frames"]),
            "countdown": recording["countdown"],
        })


@app.route("/signs")
def get_signs():
    result = {}
    for char, data in calibration_data.items():
        result[char] = {
            "frames": len(data["frames"]),
            "has_motion": data.get("has_motion", False),
            "type": "動き" if data.get("has_motion") else "静止",
        }
    return jsonify(result)


@app.route("/signs/<char>", methods=["DELETE"])
def delete_sign(char):
    if char in calibration_data:
        del calibration_data[char]
        save_calibration()
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404


@app.route("/landmarks")
def get_landmarks():
    with state_lock:
        if current_landmarks_raw is None:
            return jsonify({"detected": False, "landmarks": [], "recognized": "", "confidence": 0})

        lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in current_landmarks_raw]

    with recording_lock:
        if recording["active"]:
            return jsonify({"detected": True, "landmarks": lm_list, "recognized": "", "confidence": 0, "recording": True})

    with history_lock:
        hist = list(landmark_history)

    char, conf = recognize(hist)
    return jsonify({"detected": True, "landmarks": lm_list, "recognized": char, "confidence": conf})


@app.route("/frame")
def get_frame():
    with frame_lock:
        if current_frame_jpg is None:
            return Response("No frame", status=503)
        return Response(current_frame_jpg, mimetype="image/jpeg")


@app.route("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                if current_frame_jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           current_frame_jpg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "signs": len(calibration_data)})


@app.route("/")
def index():
    signs_rows = ""
    for char, data in calibration_data.items():
        motion = "🏃 動き" if data.get("has_motion") else "✋ 静止"
        signs_rows += f"""<tr>
            <td style="font-size:28px">{char}</td>
            <td>{motion}</td>
            <td>{len(data['frames'])}f</td>
            <td><button onclick="del_sign('{char}')">🗑</button></td>
        </tr>"""
    if not signs_rows:
        signs_rows = '<tr><td colspan="4">まだ登録なし</td></tr>'

    return f"""
    <html><head><meta charset="utf-8"><title>指文字道場 キャリブレーション</title>
    <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 30px auto; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #f4a261; }}
        h2 {{ color: #e9c46a; }}
        input, button, select {{ font-size: 18px; padding: 8px 16px; margin: 4px; }}
        button {{ background: #f4a261; border: none; cursor: pointer; border-radius: 4px; color: #1a1a2e; }}
        button:hover {{ background: #e76f51; }}
        #status {{ font-size: 24px; margin: 20px 0; color: #e9c46a; min-height: 40px; }}
        img {{ border: 2px solid #333; border-radius: 8px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ padding: 8px 12px; border-bottom: 1px solid #333; text-align: left; }}
        .controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    </style></head><body>
    <h1>🥋 指文字道場 キャリブレーション</h1>

    <img src="/stream" width="640" height="480">

    <h2>録画</h2>
    <div class="controls">
        <input type="text" id="char" placeholder="文字" maxlength="4" style="width:100px">
        <select id="duration">
            <option value="1">1秒 (静止)</option>
            <option value="2" selected>2秒</option>
            <option value="3">3秒 (動きあり)</option>
        </select>
        <button onclick="record()">🔴 録画 (3秒後開始)</button>
    </div>
    <div id="status">待機中</div>

    <h2>登録済み ({len(calibration_data)}文字)</h2>
    <table>
        <tr><th>文字</th><th>タイプ</th><th>フレーム数</th><th></th></tr>
        {signs_rows}
    </table>

    <script>
    async function record() {{
        const char = document.getElementById('char').value;
        if (!char) {{ alert('文字を入力してください'); return; }}
        const duration = parseFloat(document.getElementById('duration').value);
        document.getElementById('status').innerText = '準備中...';
        await fetch('/record', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{char, duration, countdown: 3}})
        }});
        const poll = setInterval(async () => {{
            const s = await (await fetch('/status')).json();
            if (s.countdown > 0) {{
                document.getElementById('status').innerText = s.countdown.toFixed(0) + '秒後に開始... 手の形を準備！';
            }} else if (s.recording) {{
                document.getElementById('status').innerText = '🔴 録画中: ' + s.frames + 'フレーム';
            }} else if (s.done) {{
                document.getElementById('status').innerText = '✅「' + char + '」録画完了！';
                clearInterval(poll);
                setTimeout(() => location.reload(), 800);
            }}
        }}, 300);
    }}
    async function del_sign(char) {{
        if (!confirm(char + ' を削除しますか？')) return;
        await fetch('/signs/' + encodeURIComponent(char), {{method: 'DELETE'}});
        location.reload();
    }}
    </script></body></html>
    """


if __name__ == "__main__":
    load_calibration()

    print("🥋 指文字道場 Hand Detector Server (Time-Series Edition)")
    print("   http://127.0.0.1:8765")
    print("   ブラウザでキャリブレーション → Godotでゲーム")
    print("   Ctrl+C で終了")
    print()

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    cd_thread = threading.Thread(target=countdown_thread, daemon=True)
    cd_thread.start()

    app.run(host="127.0.0.1", port=8765, debug=False)
