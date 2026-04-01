// ── 指文字道場 Browser Edition ──────────────────────────────────────────────

// 50音表
const GOJUON = [
  // 清音
  ["あ","い","う","え","お"],
  ["か","き","く","け","こ"],
  ["さ","し","す","せ","そ"],
  ["た","ち","つ","て","と"],
  ["な","に","ぬ","ね","の"],
  ["は","ひ","ふ","へ","ほ"],
  ["ま","み","む","め","も"],
  ["や","　","ゆ","　","よ"],
  ["ら","り","る","れ","ろ"],
  ["わ","を","ん","ー","　"],
  // 濁音
  ["が","ぎ","ぐ","げ","ご"],
  ["ざ","じ","ず","ぜ","ぞ"],
  ["だ","ぢ","づ","で","ど"],
  ["ば","び","ぶ","べ","ぼ"],
  // 半濁音
  ["ぱ","ぴ","ぷ","ぺ","ぽ"],
  // 小書き
  ["ぁ","ぃ","ぅ","ぇ","ぉ"],
  ["っ","ゃ","ゅ","ょ","ゎ"],
];

const WORDS = [
  "さくら","やま","かわ","そら","うみ",
  "ねこ","いぬ","とり","さかな",
  "あめ","かぜ","ゆき","はな",
  "てがみ","ともだち","せんせい",
  "おはよう","ありがとう","こんにちは",
  "ヴァンフォーレ","こうふ",
];

// ── State ───────────────────────────────────────────────────────────────────

let calibrationData = {};  // char -> {frames: [...], hasMotion: bool}
let hands = null;
let camera = null;
let currentScreen = "title-screen";

// Game state
let gameMode = "random";  // "random" | "word"
let gameScore = 0;
let gameTotal = 0;
let gameCurrentChar = "";
let gameWord = "";
let gameWordIndex = 0;
let gameTimer = 10;
let gameTimerInterval = null;
let gameWaiting = false;

// Calibration state
let calRecording = false;
let calChar = "";
let calFrames = [];
let calCountdown = 0;
let calStartTime = 0;
let calDuration = 2;
let calRawFrames = [];

// Recognition
const HISTORY_MAX = 30;
let landmarkHistory = [];
let lastRecognized = "";
let lastRecognizedTime = 0;

// ── Persistence ─────────────────────────────────────────────────────────────

function saveCalibration() {
  localStorage.setItem("yubimoji-calibration", JSON.stringify(calibrationData));
}

function loadCalibration() {
  const data = localStorage.getItem("yubimoji-calibration");
  if (data) {
    calibrationData = JSON.parse(data);
    console.log("Loaded calibration:", Object.keys(calibrationData).length, "signs");
  }
}

function exportCalibration() {
  const blob = new Blob([JSON.stringify(calibrationData, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "calibration_data.json";
  a.click();
  URL.revokeObjectURL(url);
}

function importCalibration(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      // Support both formats: {char: {frames, hasMotion}} and {char: {frames, has_motion}}
      for (const [char, val] of Object.entries(data)) {
        if (val.frames) {
          calibrationData[char] = {
            frames: val.frames,
            hasMotion: val.hasMotion || val.has_motion || false,
          };
        }
      }
      saveCalibration();
      updateGojuonGrid();
      alert(`${Object.keys(data).length}文字インポートしました！`);
    } catch (err) {
      alert("読み込みエラー: " + err.message);
    }
  };
  reader.readAsText(file);
}

// ── Normalization ───────────────────────────────────────────────────────────

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const midMcp = landmarks[9];
  let scale = Math.sqrt(
    (midMcp.x - wrist.x) ** 2 + (midMcp.y - wrist.y) ** 2
  );
  if (scale < 0.001) scale = 0.001;

  return landmarks.map(lm => ({
    x: +((lm.x - wrist.x) / scale).toFixed(4),
    y: +((lm.y - wrist.y) / scale).toFixed(4),
    z: +((lm.z - wrist.z) / scale).toFixed(4),
  }));
}

// ── DTW ─────────────────────────────────────────────────────────────────────

function frameDistance(a, b) {
  let total = 0;
  for (let i = 0; i < Math.min(a.length, b.length, 21); i++) {
    const dx = a[i].x - b[i].x;
    const dy = a[i].y - b[i].y;
    total += Math.sqrt(dx * dx + dy * dy);
  }
  return total / 21;
}

function dtwDistance(seqA, seqB) {
  const MAX = 25;
  if (seqA.length > MAX) {
    const step = (seqA.length - 1) / (MAX - 1);
    seqA = Array.from({length: MAX}, (_, i) => seqA[Math.round(i * step)]);
  }
  if (seqB.length > MAX) {
    const step = (seqB.length - 1) / (MAX - 1);
    seqB = Array.from({length: MAX}, (_, i) => seqB[Math.round(i * step)]);
  }

  const n = seqA.length, m = seqB.length;
  if (n === 0 || m === 0) return 999;

  const dtw = Array.from({length: n + 1}, () => new Float32Array(m + 1).fill(Infinity));
  dtw[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = frameDistance(seqA[i-1], seqB[j-1]);
      dtw[i][j] = cost + Math.min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]);
    }
  }
  return dtw[n][m] / Math.max(n, m);
}

// ── Recognition ─────────────────────────────────────────────────────────────

function isHandMoving(history) {
  // Check if hand has significant movement in recent frames
  if (history.length < 5) return false;
  const recent = history.slice(-15);
  let maxDist = 0;
  for (let i = 1; i < recent.length; i++) {
    const d = frameDistance(recent[i - 1], recent[i]);
    if (d > maxDist) maxDist = d;
  }
  // Also check overall displacement
  const totalDist = frameDistance(recent[0], recent[recent.length - 1]);
  return maxDist > 0.1 || totalDist > 0.15;
}

function recognize(history) {
  if (!Object.keys(calibrationData).length || !history.length) return ["", 0];

  const handMoving = isHandMoving(history);

  // Track if hand moved recently (within last ~1 second)
  const recentlyMoved = history.length >= 15 && isHandMoving(history.slice(-20));

  let bestStaticChar = "", bestStaticDist = 999;
  let bestMotionChar = "", bestMotionDist = 999;

  for (const [char, data] of Object.entries(calibrationData)) {
    const calFrames = data.frames;

    if (data.hasMotion) {
      // Motion sign: match when moving OR recently moved
      if (handMoving || recentlyMoved) {
        const dist = dtwDistance(history.slice(-20), calFrames);
        if (dist < bestMotionDist) {
          bestMotionDist = dist;
          bestMotionChar = char;
        }
      }
    } else {
      // Static sign: match latest frame
      const midIdx = Math.floor(calFrames.length / 2);
      const dist = frameDistance(history[history.length - 1], calFrames[midIdx]);
      if (dist < bestStaticDist) {
        bestStaticDist = dist;
        bestStaticChar = char;
      }
    }
  }

  // When moving/recently moved: prefer motion match, fall back to static
  if ((handMoving || recentlyMoved) && bestMotionChar && bestMotionDist < 0.6) {
    const conf = Math.max(0, Math.min(100, Math.round((1 - bestMotionDist / 0.6) * 100)));
    return [bestMotionChar, conf];
  }

  if (bestStaticChar && bestStaticDist < 0.4) {
    const conf = Math.max(0, Math.min(100, Math.round((1 - bestStaticDist / 0.4) * 100)));
    return [bestStaticChar, conf];
  }

  return ["", 0];
}

// ── MediaPipe ───────────────────────────────────────────────────────────────

function initHands() {
  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
  });
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onHandResults);
}

function startCamera(videoEl, canvasEl) {
  if (camera) camera.stop();
  camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({image: videoEl}); },
    width: 640,
    height: 480,
  });
  camera.start();
  // Store current canvas for drawing
  window._currentCanvas = canvasEl;
  window._currentVideo = videoEl;
}

function onHandResults(results) {
  const canvas = window._currentCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    // Draw landmarks
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {color: "#0f0", lineWidth: 2});
    drawLandmarks(ctx, landmarks, {color: "#0f0", lineWidth: 1, radius: 3});

    const normalized = normalizeLandmarks(landmarks);

    // Add to history
    landmarkHistory.push(normalized);
    while (landmarkHistory.length > HISTORY_MAX) landmarkHistory.shift();

    // Calibration recording — save both normalized AND raw landmarks
    if (calRecording && calCountdown <= 0) {
      const raw = landmarks.map(lm => ({x: lm.x, y: lm.y, z: lm.z}));
      calFrames.push(normalized);
      if (!calRawFrames) calRawFrames = [];
      calRawFrames.push(raw);
      const elapsed = (Date.now() - calStartTime) / 1000;
      updateCalStatus(`🔴 録画中: ${calFrames.length}f (${(calDuration - elapsed).toFixed(1)}s)`);

      if (elapsed >= calDuration) {
        finishCalRecording();
      }
    }

    // Calibration test mode — show recognized sign live
    if (currentScreen === "calibration-screen" && !calRecording) {
      const [char, conf] = recognize(landmarkHistory);
      if (char && conf > 20) {
        updateCalStatus(`認識: ${char} (${conf}%)`);
      } else {
        updateCalStatus("✋ 手を出すと認識テスト");
      }
    }

    // Game recognition
    if (currentScreen === "game-screen" && !gameWaiting) {
      const [char, conf] = recognize(landmarkHistory);
      if (char && conf > 30) {
        // Require stable recognition (same char for 300ms)
        if (char === lastRecognized && Date.now() - lastRecognizedTime > 300) {
          onSignRecognized(char, conf);
        }
        if (char !== lastRecognized) {
          lastRecognized = char;
          lastRecognizedTime = Date.now();
        }
        updateGameStatus(`認識: ${char} (${conf}%)`);
      } else {
        lastRecognized = "";
        updateGameStatus("手を検出中... ✋");
      }
    }
  } else {
    if (currentScreen === "game-screen" && !gameWaiting) {
      updateGameStatus("手が見えません 👀");
    }
  }
}

// ── Screen Management ───────────────────────────────────────────────────────

function showScreen(id) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  currentScreen = id;

  if (id === "calibration-screen") {
    updateGojuonGrid();
    startCamera(
      document.getElementById("cal-video"),
      document.getElementById("cal-canvas")
    );
  } else if (id === "title-screen") {
    updateSignCount();
    if (camera) camera.stop();
  }
}

function updateSignCount() {
  const count = Object.keys(calibrationData).length;
  document.getElementById("sign-count").textContent =
    count > 0 ? `${count}文字 登録済み` : "まずキャリブレーションしてください";
}

// ── Calibration ─────────────────────────────────────────────────────────────

const SECTION_LABELS = {
  0: "清音",
  10: "濁音",
  14: "半濁音",
  15: "小書き",
};

function updateGojuonGrid() {
  const grid = document.getElementById("gojuon-grid");
  grid.innerHTML = "";

  for (let r = 0; r < GOJUON.length; r++) {
    // Section headers
    if (SECTION_LABELS[r]) {
      const label = document.createElement("div");
      label.className = "gojuon-section";
      label.textContent = SECTION_LABELS[r];
      grid.appendChild(label);
    }

    for (const char of GOJUON[r]) {
      const btn = document.createElement("button");
      btn.className = "gojuon-btn";
      btn.textContent = char;

      if (char === "　") {
        btn.style.visibility = "hidden";
      } else {
        if (calibrationData[char]) {
          btn.classList.add("recorded");
          const badge = document.createElement("span");
          badge.className = "type-badge";
          badge.textContent = calibrationData[char].hasMotion ? "🏃" : "✋";
          btn.appendChild(badge);
        }
        btn.onclick = () => startCalRecording(char);
      }
      grid.appendChild(btn);
    }
  }
}

function startCalRecording(char) {
  calChar = char;
  calFrames = [];
  calRawFrames = [];
  calDuration = parseFloat(document.getElementById("cal-duration").value);
  calCountdown = 3;
  calRecording = true;

  // Highlight button
  document.querySelectorAll(".gojuon-btn").forEach(b => b.classList.remove("recording"));
  document.querySelectorAll(".gojuon-btn").forEach(b => {
    if (b.textContent.startsWith(char)) b.classList.add("recording");
  });

  // Countdown
  updateCalStatus(`${calCountdown}... 「${char}」の手の形を準備！`);
  const countInterval = setInterval(() => {
    calCountdown--;
    if (calCountdown > 0) {
      updateCalStatus(`${calCountdown}... 「${char}」の手の形を準備！`);
    } else {
      clearInterval(countInterval);
      calStartTime = Date.now();
      updateCalStatus("🔴 録画中...");
    }
  }, 1000);
}

function finishCalRecording() {
  calRecording = false;
  // Detect motion using raw frames (position-aware)
  const hasMotion = detectMotionRaw(calRawFrames) || detectMotion(calFrames);
  calibrationData[calChar] = { frames: calFrames, hasMotion };
  saveCalibration();

  const typeStr = hasMotion ? "動き" : "静止";
  updateCalStatus(`✅「${calChar}」録画完了！ (${typeStr}, ${calFrames.length}f)`);

  document.querySelectorAll(".gojuon-btn").forEach(b => b.classList.remove("recording"));
  updateGojuonGrid();
}

function detectMotionRaw(rawFrames) {
  // Detect motion from raw (non-normalized) wrist position changes
  if (!rawFrames || rawFrames.length < 5) return false;
  let maxDist = 0;
  const first = rawFrames[0][0]; // wrist of first frame
  for (const f of rawFrames) {
    const wrist = f[0];
    const dx = wrist.x - first.x;
    const dy = wrist.y - first.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > maxDist) maxDist = d;
  }
  return maxDist > 0.05; // 5% of screen = significant movement
}

function detectMotion(frames) {
  if (frames.length < 5) return false;
  // Check normalized shape change
  let maxShapeDist = 0;
  for (const f of frames) {
    const d = frameDistance(frames[0], f);
    if (d > maxShapeDist) maxShapeDist = d;
  }
  // Also check wrist position change (raw position, landmark[0])
  // Wrist is always {x:0, y:0} in normalized, but we can check
  // the overall hand center movement by looking at landmark 9 (middle MCP)
  // relative to first frame
  let maxPosDist = 0;
  for (const f of frames) {
    const dx = f[9].x - frames[0][9].x;
    const dy = f[9].y - frames[0][9].y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > maxPosDist) maxPosDist = d;
  }
  // Motion if shape changes OR significant position change
  return maxShapeDist > 0.3 || maxPosDist > 0.3;
}

function updateCalStatus(text) {
  const el = document.getElementById("cal-status");
  if (el) el.textContent = text;
}

// ── Game ────────────────────────────────────────────────────────────────────

function startGame(mode) {
  const chars = Object.keys(calibrationData);
  if (chars.length < 2) {
    alert("2文字以上キャリブレーションしてください！");
    return;
  }

  gameMode = mode;
  gameScore = 0;
  gameTotal = 0;
  gameWaiting = false;
  landmarkHistory = [];
  lastRecognized = "";

  showScreen("game-screen");
  startCamera(
    document.getElementById("game-video"),
    document.getElementById("game-canvas")
  );

  if (mode === "word") {
    // Filter words to only use calibrated chars
    const available = WORDS.filter(w =>
      [...w].every(c => calibrationData[c] || c === "　")
    );
    if (available.length === 0) {
      alert("使える単語がありません。もっと文字を登録してください。");
      endGame();
      return;
    }
    gameWord = available[Math.floor(Math.random() * available.length)];
    gameWordIndex = 0;
    nextWordChar();
  } else {
    nextRandomChar();
  }
}

function nextRandomChar() {
  const chars = Object.keys(calibrationData);
  gameCurrentChar = chars[Math.floor(Math.random() * chars.length)];
  gameTotal++;
  document.getElementById("prompt-char").textContent = gameCurrentChar;
  document.getElementById("word-progress").innerHTML = "";
  document.getElementById("score").textContent = `${gameScore} / ${gameTotal}`;
  startTimer();
}

function nextWordChar() {
  if (gameWordIndex >= gameWord.length) {
    // Word complete!
    showBanner("🎉 " + gameWord, "correct");
    setTimeout(() => {
      const available = WORDS.filter(w =>
        [...w].every(c => calibrationData[c] || c === "　")
      );
      gameWord = available[Math.floor(Math.random() * available.length)];
      gameWordIndex = 0;
      nextWordChar();
    }, 2000);
    return;
  }

  gameCurrentChar = gameWord[gameWordIndex];
  gameTotal++;
  document.getElementById("prompt-char").textContent = gameCurrentChar;

  // Show word progress
  let html = "";
  for (let i = 0; i < gameWord.length; i++) {
    if (i < gameWordIndex) html += `<span class="done">${gameWord[i]}</span>`;
    else if (i === gameWordIndex) html += `<span class="current">${gameWord[i]}</span>`;
    else html += `<span>${gameWord[i]}</span>`;
  }
  document.getElementById("word-progress").innerHTML = html;
  document.getElementById("score").textContent = `${gameScore} / ${gameTotal}`;
  startTimer();
}

function startTimer() {
  gameTimer = 10;
  gameWaiting = false;
  document.getElementById("timer").textContent = gameTimer;
  if (gameTimerInterval) clearInterval(gameTimerInterval);
  gameTimerInterval = setInterval(() => {
    gameTimer--;
    document.getElementById("timer").textContent = gameTimer;
    if (gameTimer <= 0) {
      clearInterval(gameTimerInterval);
      onTimeout();
    }
  }, 1000);
}

function onSignRecognized(char, conf) {
  if (gameWaiting) return;

  if (char === gameCurrentChar) {
    gameScore++;
    document.getElementById("score").textContent = `${gameScore} / ${gameTotal}`;
    showBanner("正解！", "correct");
    gameWaiting = true;
    clearInterval(gameTimerInterval);
    setTimeout(() => {
      hideBanner();
      if (gameMode === "word") {
        gameWordIndex++;
        nextWordChar();
      } else {
        if (gameTotal >= 10) {
          showResults();
        } else {
          nextRandomChar();
        }
      }
    }, 1000);
  }
}

function onTimeout() {
  showBanner(`時間切れ → ${gameCurrentChar}`, "incorrect");
  gameWaiting = true;
  setTimeout(() => {
    hideBanner();
    if (gameMode === "word") {
      gameWordIndex++;
      nextWordChar();
    } else {
      if (gameTotal >= 10) {
        showResults();
      } else {
        nextRandomChar();
      }
    }
  }, 2000);
}

function showResults() {
  const pct = Math.round(gameScore / gameTotal * 100);
  document.getElementById("prompt-char").textContent = "終了！";
  document.getElementById("word-progress").innerHTML = "";
  updateGameStatus(`${gameTotal}問中${gameScore}問正解 (${pct}%)`);
  document.getElementById("timer").textContent = "";
  setTimeout(() => endGame(), 4000);
}

let gamePaused = false;

function endGame() {
  clearInterval(gameTimerInterval);
  gamePaused = false;
  if (camera) camera.stop();
  showScreen("title-screen");
}

function pauseToCalibration() {
  // Pause game and go to calibration
  clearInterval(gameTimerInterval);
  gamePaused = true;
  showScreen("calibration-screen");
}

function resumeGame() {
  if (gamePaused) {
    gamePaused = false;
    showScreen("game-screen");
    startCamera(
      document.getElementById("game-video"),
      document.getElementById("game-canvas")
    );
    startTimer();
  }
}

function showBanner(text, type) {
  const el = document.getElementById("result-banner");
  el.textContent = text;
  el.className = `result-banner ${type}`;
}

function hideBanner() {
  document.getElementById("result-banner").className = "result-banner hidden";
}

function updateGameStatus(text) {
  const el = document.getElementById("game-status");
  if (el) el.textContent = text;
}

// ── Init ────────────────────────────────────────────────────────────────────

// Load calibration: try localStorage first, then fetch calibration_data.json
loadCalibration();
if (Object.keys(calibrationData).length === 0) {
  // Try loading from file (for first-time migration from Godot version)
  fetch("calibration_data.json")
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (data && Object.keys(data).length > 0) {
        for (const [char, val] of Object.entries(data)) {
          if (val.frames) {
            calibrationData[char] = {
              frames: val.frames,
              hasMotion: val.hasMotion || val.has_motion || false,
            };
          }
        }
        saveCalibration();
        updateSignCount();
        console.log("Auto-imported calibration_data.json:", Object.keys(calibrationData).length, "signs");
      }
    })
    .catch(() => {});
}
initHands();
updateSignCount();
