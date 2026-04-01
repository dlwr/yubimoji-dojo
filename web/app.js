// ── 指文字道場 Browser Edition ──────────────────────────────────────────────

// 50音表
const GOJUON = [
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

function recognize(history) {
  if (!Object.keys(calibrationData).length || !history.length) return ["", 0];

  let bestChar = "", bestDist = 999;

  for (const [char, data] of Object.entries(calibrationData)) {
    const calFrames = data.frames;
    let dist;

    if (data.hasMotion) {
      dist = dtwDistance(history, calFrames);
    } else {
      const midIdx = Math.floor(calFrames.length / 2);
      dist = frameDistance(history[history.length - 1], calFrames[midIdx]);
    }

    if (dist < bestDist) {
      bestDist = dist;
      bestChar = char;
    }
  }

  const threshold = calibrationData[bestChar]?.hasMotion ? 0.6 : 0.4;
  if (bestDist < threshold) {
    const confidence = Math.max(0, Math.min(100, Math.round((1 - bestDist / threshold) * 100)));
    return [bestChar, confidence];
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

    // Calibration recording
    if (calRecording && calCountdown <= 0) {
      calFrames.push(normalized);
      const elapsed = (Date.now() - calStartTime) / 1000;
      updateCalStatus(`🔴 録画中: ${calFrames.length}f (${(calDuration - elapsed).toFixed(1)}s)`);

      if (elapsed >= calDuration) {
        finishCalRecording();
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

function updateGojuonGrid() {
  const grid = document.getElementById("gojuon-grid");
  grid.innerHTML = "";

  for (const row of GOJUON) {
    for (const char of row) {
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
  const hasMotion = detectMotion(calFrames);
  calibrationData[calChar] = { frames: calFrames, hasMotion };
  saveCalibration();

  const typeStr = hasMotion ? "動き" : "静止";
  updateCalStatus(`✅「${calChar}」録画完了！ (${typeStr}, ${calFrames.length}f)`);

  document.querySelectorAll(".gojuon-btn").forEach(b => b.classList.remove("recording"));
  updateGojuonGrid();
}

function detectMotion(frames) {
  if (frames.length < 5) return false;
  let maxDist = 0;
  for (const f of frames) {
    const d = frameDistance(frames[0], f);
    if (d > maxDist) maxDist = d;
  }
  return maxDist > 0.4;
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

function endGame() {
  clearInterval(gameTimerInterval);
  if (camera) camera.stop();
  showScreen("title-screen");
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
