// ── 指文字道場 Browser Edition (3-Axis Recognition) ─────────────────────────
//
// Recognition uses 3 independent axes:
//   1. Shape:       finger joint angles (relative positions between joints)
//   2. Orientation: palm normal vector (which direction the hand faces)
//   3. Motion:      wrist position change over time (relative to start)

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

const SECTION_LABELS = {
  0: "清音",
  10: "濁音",
  14: "半濁音",
  15: "小書き",
};

const WORDS = [
  // カタカナ語（指文字の実用）
  "あめりか","いぎりす","ふらんす","いたりあ","すぺいん",
  "どいつ","ぶらじる","かなだ","おーすとらりあ",
  "とうきょう","おおさか","こうふ","よこはま","なごや",
  // 外国人名
  "まいける","じょん","ありす","えみりー","でいびっど",
  "ろばーと","じぇーむず","おりばー","そふぃあ",
  // 日本人名（指文字で表すことがある）
  "たなか","すずき","やまだ","さとう","わたなべ",
  // カタカナ語（日常）
  "こーひー","ぱそこん","すまほ","いんたーねっと",
  "ぷろぐらみんぐ","げーむ","さっかー","ばすけっと",
  "ぎたー","ぴあの","かめら","てれび",
  // 手話・ろう関連
  "しゅわ","ゆびもじ","でふりんぴっく",
  // 日常の言葉
  "おはよう","ありがとう","こんにちは","こんばんは",
  "さようなら","すみません","おねがいします",
  // 食べ物
  "らーめん","すし","てんぷら","うどん","そば",
  "ぱすた","ぴざ","はんばーがー","かれー",
  // ヴァンフォーレ
  "ヴぁんふぉーれ","こうふ","じぇいりーぐ",
];

// ── 3-Axis Feature Extraction ───────────────────────────────────────────────

function extractShape(landmarks) {
  // Shape: joint angles + finger spread (camera-angle independent)
  // 
  // For each finger: 3 joint angles (MCP bend, PIP bend, DIP bend)
  // Plus: 4 finger spread angles (between adjacent fingers)
  // Plus: thumb position relative to palm
  //
  // Total: 15 joint angles + 4 spreads + 2 thumb = 21 values

  const angles = [];

  // Finger joint chains: [MCP, PIP, DIP, TIP]
  const fingers = [
    [1, 2, 3, 4],     // thumb
    [5, 6, 7, 8],     // index
    [9, 10, 11, 12],  // middle
    [13, 14, 15, 16], // ring
    [17, 18, 19, 20], // pinky
  ];

  // Joint angles for each finger (3 per finger = 15 total)
  for (const f of fingers) {
    for (let i = 0; i < 3; i++) {
      const a = landmarks[f[i]];
      const b = landmarks[f[i + 1]];
      let prev;
      if (i === 0) {
        prev = landmarks[0]; // wrist for MCP
      } else {
        prev = landmarks[f[i - 1]];
      }
      angles.push(angleBetween(prev, a, b));
    }
  }

  // Finger spread: angle between adjacent fingertips relative to MCP line
  const mcps = [5, 9, 13, 17]; // index, middle, ring, pinky MCP
  const tips = [8, 12, 16, 20];
  for (let i = 0; i < 3; i++) {
    const spread = angleBetween(landmarks[0], landmarks[tips[i]], landmarks[tips[i + 1]]);
    angles.push(spread);
  }
  // Thumb-index spread
  angles.push(angleBetween(landmarks[0], landmarks[4], landmarks[8]));

  // Thumb cross: is thumb tip near index MCP? (for signs like お)
  const thumbCross = dist3d(landmarks[4], landmarks[5]);
  const handSize = dist3d(landmarks[0], landmarks[9]);
  angles.push(handSize > 0.001 ? thumbCross / handSize : 0);

  // Thumb position: above or below index (for signs like あ)
  angles.push(landmarks[4].y - landmarks[8].y);

  return angles;
}

function angleBetween(a, b, c) {
  // Angle at point b formed by vectors b→a and b→c
  const bax = a.x - b.x, bay = a.y - b.y, baz = a.z - b.z;
  const bcx = c.x - b.x, bcy = c.y - b.y, bcz = c.z - b.z;
  const dot = bax * bcx + bay * bcy + baz * bcz;
  const magA = Math.sqrt(bax * bax + bay * bay + baz * baz);
  const magC = Math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz);
  if (magA < 0.0001 || magC < 0.0001) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dot / (magA * magC)));
  return Math.acos(cosAngle); // radians, 0 to PI
}

function dist3d(a, b) {
  return Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2);
}

function extractOrientation(landmarks) {
  // Orientation: palm normal vector using cross product
  // Vector A: wrist → middle MCP (landmark 0 → 9)
  // Vector B: wrist → index MCP (landmark 0 → 5)
  // Normal = A × B (gives direction palm faces)
  const w = landmarks[0];
  const mMcp = landmarks[9];
  const iMcp = landmarks[5];

  const ax = mMcp.x - w.x, ay = mMcp.y - w.y, az = mMcp.z - w.z;
  const bx = iMcp.x - w.x, by = iMcp.y - w.y, bz = iMcp.z - w.z;

  // Cross product
  let nx = ay * bz - az * by;
  let ny = az * bx - ax * bz;
  let nz = ax * by - ay * bx;

  // Normalize
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (len > 0.0001) {
    nx /= len; ny /= len; nz /= len;
  }

  // Also compute hand "up" direction (wrist → middle fingertip)
  const tip = landmarks[12];
  let ux = tip.x - w.x, uy = tip.y - w.y, uz = tip.z - w.z;
  const ulen = Math.sqrt(ux * ux + uy * uy + uz * uz);
  if (ulen > 0.0001) {
    ux /= ulen; uy /= ulen; uz /= ulen;
  }

  return { nx, ny, nz, ux, uy, uz };
}

function extractPosition(landmarks) {
  // Raw wrist position (for motion tracking)
  return { x: landmarks[0].x, y: landmarks[0].y, z: landmarks[0].z };
}

function extractFeatures(landmarks) {
  return {
    shape: extractShape(landmarks),
    orientation: extractOrientation(landmarks),
    position: extractPosition(landmarks),
  };
}

// ── Distance Functions (per axis) ───────────────────────────────────────────

function shapeDistance(a, b) {
  // a and b are arrays of numbers (joint angles)
  if (!a || !b || a.length === 0 || b.length === 0) return 999;
  const len = Math.min(a.length, b.length);
  let total = 0;
  for (let i = 0; i < len; i++) {
    const diff = a[i] - b[i];
    total += diff * diff;
  }
  return Math.sqrt(total / len);
}

function orientationDistance(a, b) {
  // Cosine distance of palm normal
  const dotN = a.nx * b.nx + a.ny * b.ny + a.nz * b.nz;
  // Cosine distance of hand "up" direction
  const dotU = a.ux * b.ux + a.uy * b.uy + a.uz * b.uz;
  // Average: 0 = identical, 2 = opposite
  return (1 - dotN) + (1 - dotU);
}

function motionDistance(seqA, seqB) {
  // Compare relative motion trajectories using DTW
  // Convert absolute positions to relative (delta from first frame)
  if (seqA.length < 2 || seqB.length < 2) return 0;

  const relA = seqA.map(p => ({
    x: p.x - seqA[0].x,
    y: p.y - seqA[0].y,
    z: p.z - seqA[0].z,
  }));
  const relB = seqB.map(p => ({
    x: p.x - seqB[0].x,
    y: p.y - seqB[0].y,
    z: p.z - seqB[0].z,
  }));

  return dtwPositions(relA, relB);
}

function dtwPositions(seqA, seqB) {
  const MAX = 15;
  if (seqA.length > MAX) {
    const step = (seqA.length - 1) / (MAX - 1);
    seqA = Array.from({length: MAX}, (_, i) => seqA[Math.round(i * step)]);
  }
  if (seqB.length > MAX) {
    const step = (seqB.length - 1) / (MAX - 1);
    seqB = Array.from({length: MAX}, (_, i) => seqB[Math.round(i * step)]);
  }

  const n = seqA.length, m = seqB.length;
  if (n === 0 || m === 0) return 0;

  const dtw = Array.from({length: n + 1}, () => new Float32Array(m + 1).fill(Infinity));
  dtw[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const dx = seqA[i-1].x - seqB[j-1].x;
      const dy = seqA[i-1].y - seqB[j-1].y;
      const cost = Math.sqrt(dx * dx + dy * dy);
      dtw[i][j] = cost + Math.min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]);
    }
  }
  return dtw[n][m] / Math.max(n, m);
}

// ── DTW for shape sequences ─────────────────────────────────────────────────

function dtwShape(seqA, seqB) {
  const MAX = 15;
  if (!seqA.length || !seqB.length) return 999;
  if (seqA.length > MAX) {
    const step = (seqA.length - 1) / (MAX - 1);
    seqA = Array.from({length: MAX}, (_, i) => seqA[Math.round(i * step)]);
  }
  if (seqB.length > MAX) {
    const step = (seqB.length - 1) / (MAX - 1);
    seqB = Array.from({length: MAX}, (_, i) => seqB[Math.round(i * step)]);
  }
  const n = seqA.length, m = seqB.length;
  const dtw = Array.from({length: n + 1}, () => new Float32Array(m + 1).fill(Infinity));
  dtw[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = shapeDistance(seqA[i-1], seqB[j-1]);
      dtw[i][j] = cost + Math.min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]);
    }
  }
  return dtw[n][m] / Math.max(n, m);
}

// ── State ───────────────────────────────────────────────────────────────────

let calibrationData = {};  // char -> {shapes, orientations, positions, hasMotion}
let hands = null;
let camera = null;
let currentScreen = "title-screen";

// Game state
let gameMode = "random";
let gameScore = 0;
let gameTotal = 0;
let gameCurrentChar = "";
let gameWord = "";
let gameWordIndex = 0;
let gameTimer = 10;
let gameTimerInterval = null;
let gameWaiting = false;
let gamePaused = false;

// Calibration state
let calRecording = false;
let calChar = "";
let calShapes = [];
let calOrientations = [];
let calPositions = [];
let calCountdown = 0;
let calStartTime = 0;
let calDuration = 2;

// Recognition history
const HISTORY_MAX = 30;
let featureHistory = [];  // list of {shape, orientation, position}
let lastRecognized = "";
let lastRecognizedTime = 0;
let recognizeCounter = 0;

// ── Persistence ─────────────────────────────────────────────────────────────

// ── IndexedDB Storage (no size limit) ───────────────────────────────────────

const API_URL = "https://yubimoji-dojo-api.yuta25.workers.dev";
const DB_NAME = "yubimoji-dojo";
const DB_STORE = "calibration";
const DB_VERSION = 1;

// Admin mode: ?admin=KEY in URL
const urlParams = new URLSearchParams(window.location.search);
const ADMIN_KEY = urlParams.get("admin") || "";
const IS_ADMIN = ADMIN_KEY.length > 0;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        db.createObjectStore(DB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveCalibration() {
  updateCalStatus("💾 保存中...");
  try {
    // Save to IndexedDB (local cache)
    const db = await openDB();
    const tx = db.transaction(DB_STORE, "readwrite");
    const store = tx.objectStore(DB_STORE);
    store.put(calibrationData, "data");
    await new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });

    // If admin, also save to KV (shared with all users)
    if (IS_ADMIN) {
      console.log("Saving to KV...", Object.keys(calibrationData).length, "signs");
      const res = await fetch(`${API_URL}/calibration?key=${ADMIN_KEY}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(calibrationData),
      });
      const result = await res.json();
      console.log("KV response:", result);
      if (!res.ok) throw new Error("KV save failed: " + JSON.stringify(result));
      updateCalStatus(`💾 保存完了 ✓ (${Object.keys(calibrationData).length}文字 サーバー同期済み)`);
    } else {
      updateCalStatus("💾 保存完了 ✓ (ローカルのみ)");
    }
    console.log("Saved:", Object.keys(calibrationData).length, "signs");
  } catch (e) {
    console.warn("Save failed:", e);
    updateCalStatus("⚠️ 保存失敗: " + e.message);
  }
}

async function loadCalibration() {
  // 1. Try loading from KV (server, shared data)
  try {
    const res = await fetch(`${API_URL}/calibration`);
    if (res.ok) {
      const data = await res.json();
      if (data && Object.keys(data).length > 0) {
        // Check if data is v3 format (shapes are arrays of numbers, not arrays of {x,y,z})
        const firstChar = Object.keys(data)[0];
        const firstShape = data[firstChar]?.shapes?.[0];
        const isV3 = typeof firstShape?.[0] === "number";
        if (isV3) {
          calibrationData = data;
          console.log("Loaded v3 from KV:", Object.keys(calibrationData).length, "signs");
        } else {
          console.log("KV has old format data, skipping");
        }
        // Cache locally
        try {
          const db = await openDB();
          const tx = db.transaction(DB_STORE, "readwrite");
          tx.objectStore(DB_STORE).put(calibrationData, "data");
        } catch (e) {}
        return;
      }
    }
  } catch (e) {
    console.warn("KV load failed, trying local:", e);
  }

  // 2. Fallback: IndexedDB (local cache)
  try {
    const db = await openDB();
    const tx = db.transaction(DB_STORE, "readonly");
    const store = tx.objectStore(DB_STORE);
    const req = store.get("data");
    const data = await new Promise((resolve, reject) => {
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    if (data && Object.keys(data).length > 0) {
      calibrationData = data;
      console.log("Loaded from IndexedDB:", Object.keys(calibrationData).length, "signs");
      return;
    }
  } catch (e) {
    console.warn("IndexedDB load failed:", e);
  }

  // 3. Fallback: localStorage migration
  const lsData = localStorage.getItem("yubimoji-calibration-v2") ||
                  localStorage.getItem("yubimoji-calibration");
  if (lsData) {
    try {
      const parsed = JSON.parse(lsData);
      for (const [char, val] of Object.entries(parsed)) {
        if (val.shapes) {
          calibrationData[char] = val;
        } else if (val.frames) {
          calibrationData[char] = migrateV1(char, val);
        }
      }
      console.log("Migrated from localStorage:", Object.keys(calibrationData).length, "signs");
      localStorage.removeItem("yubimoji-calibration-v2");
      localStorage.removeItem("yubimoji-calibration");
    } catch (e) {}
  }
}

function exportCalibration() {
  const blob = new Blob([JSON.stringify(calibrationData, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "calibration_data_v2.json";
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
      let count = 0;
      for (const [char, val] of Object.entries(data)) {
        if (val.shapes) {
          // v2 format
          calibrationData[char] = val;
          count++;
        } else if (val.frames) {
          // v1 format — migrate
          calibrationData[char] = migrateV1(char, val);
          count++;
        }
      }
      saveCalibration();
      updateGojuonGrid();
      alert(`${count}文字インポートしました！`);
    } catch (err) {
      alert("読み込みエラー: " + err.message);
    }
  };
  reader.readAsText(file);
}

function migrateV1(char, v1data) {
  // Convert v1 (single frames array) to v2 (separate axes)
  // v1 frames are already normalized shapes
  return {
    shapes: v1data.frames,
    orientations: [],  // not available in v1
    positions: [],     // not available in v1
    hasMotion: v1data.hasMotion || v1data.has_motion || false,
  };
}

// ── Motion Detection ────────────────────────────────────────────────────────

function detectMotionFromPositions(positions) {
  if (positions.length < 5) return false;
  let maxDist = 0;
  const first = positions[0];
  for (const p of positions) {
    const dx = p.x - first.x;
    const dy = p.y - first.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > maxDist) maxDist = d;
  }
  return maxDist > 0.04;  // ~4% of screen
}

function detectMotionFromShapes(shapes) {
  if (shapes.length < 5) return false;
  let maxDist = 0;
  for (const s of shapes) {
    const d = shapeDistance(shapes[0], s);
    if (d > maxDist) maxDist = d;
  }
  return maxDist > 0.3;
}

function isHandCurrentlyMoving(history) {
  if (history.length < 5) return false;
  const recent = history.slice(-12);
  let maxDist = 0;
  for (let i = 1; i < recent.length; i++) {
    const dx = recent[i].position.x - recent[i-1].position.x;
    const dy = recent[i].position.y - recent[i-1].position.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > maxDist) maxDist = d;
  }
  const totalDx = recent[recent.length-1].position.x - recent[0].position.x;
  const totalDy = recent[recent.length-1].position.y - recent[0].position.y;
  const totalDist = Math.sqrt(totalDx * totalDx + totalDy * totalDy);
  return maxDist > 0.008 || totalDist > 0.03;
}

// ── Recognition (3-Axis) ────────────────────────────────────────────────────

function recognizeDebug(history) {
  // Returns sorted array of {char, score, type} for debugging
  if (!Object.keys(calibrationData).length || !history.length) return [];

  const moving = isHandCurrentlyMoving(history);
  const recentlyMoved = history.length >= 10 && isHandCurrentlyMoving(history.slice(-20));
  const results = [];

  for (const [char, cal] of Object.entries(calibrationData)) {
    if (cal.hasMotion) {
      if (!moving && !recentlyMoved) continue;
      const recentShapes = history.slice(-20).map(h => h.shape);
      const recentPositions = history.slice(-20).map(h => h.position);

      let shapeSim = 0;
      if (cal.shapes.length > 0) {
        const d = dtwShape(recentShapes, cal.shapes);
        shapeSim = Math.max(0, 1 - d / 1.2);
      }
      let motionSim = 0;
      if (cal.positions.length > 0) {
        const d = motionDistance(recentPositions, cal.positions);
        motionSim = Math.max(0, 1 - d / 0.1);
      }
      let orientSim = 0;
      if (cal.orientations.length > 0) {
        const calMid = cal.orientations[Math.floor(cal.orientations.length / 2)];
        const curOr = history[history.length - 1].orientation;
        orientSim = Math.max(0, 1 - orientationDistance(curOr, calMid) / 1.5);
      }
      const hasOrient = cal.orientations && cal.orientations.length > 0;
      const hasPos = cal.positions && cal.positions.length > 0;
      let score;
      if (hasOrient && hasPos) {
        score = shapeSim * 0.3 + motionSim * 0.4 + orientSim * 0.3;
      } else if (hasPos) {
        score = shapeSim * 0.5 + motionSim * 0.5;
      } else {
        score = shapeSim;
      }
      results.push({char, score: Math.round(score * 100), type: "動"});
    } else {
      const latest = history[history.length - 1];
      const midIdx = Math.floor(cal.shapes.length / 2);
      // Joint angle distance (smaller = better match, typically 0-1.5 range)
      const d = shapeDistance(latest.shape, cal.shapes[midIdx]);
      const shapeSim = Math.max(0, 1 - d / 1.0);
      let orientSim = 0;
      if (cal.orientations.length > 0) {
        const midOr = cal.orientations[Math.floor(cal.orientations.length / 2)];
        orientSim = Math.max(0, 1 - orientationDistance(latest.orientation, midOr) / 1.5);
      }
      const hasOrient = cal.orientations && cal.orientations.length > 0;
      const score = hasOrient ? shapeSim * 0.6 + orientSim * 0.4 : shapeSim;
      results.push({char, score: Math.round(score * 100), type: "静"});
    }
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

function recognize(history) {
  if (!Object.keys(calibrationData).length || !history.length) return ["", 0];

  const moving = isHandCurrentlyMoving(history);
  const recentlyMoved = history.length >= 10 &&
    isHandCurrentlyMoving(history.slice(-20));

  let bestStaticChar = "", bestStaticScore = -1;
  let bestMotionChar = "", bestMotionScore = -1;

  for (const [char, cal] of Object.entries(calibrationData)) {
    if (cal.hasMotion) {
      if (!moving && !recentlyMoved) continue;

      const recentShapes = history.slice(-20).map(h => h.shape);
      const recentPositions = history.slice(-20).map(h => h.position);

      let shapeSim = 0;
      if (cal.shapes.length > 0) {
        const d = dtwShape(recentShapes, cal.shapes);
        shapeSim = Math.max(0, 1 - d / 1.2);
      }

      let motionSim = 0;
      if (cal.positions.length > 0) {
        const d = motionDistance(recentPositions, cal.positions);
        motionSim = Math.max(0, 1 - d / 0.1);
      }

      let orientSim = 0;
      if (cal.orientations.length > 0 && history.length > 0) {
        const calMid = cal.orientations[Math.floor(cal.orientations.length / 2)];
        const curOr = history[history.length - 1].orientation;
        orientSim = Math.max(0, 1 - orientationDistance(curOr, calMid) / 1.5);
      }

      const hasOrient = cal.orientations && cal.orientations.length > 0;
      const hasPos = cal.positions && cal.positions.length > 0;
      let score;
      if (hasOrient && hasPos) {
        score = shapeSim * 0.3 + motionSim * 0.4 + orientSim * 0.3;
      } else if (hasPos) {
        score = shapeSim * 0.5 + motionSim * 0.5;
      } else {
        score = shapeSim;
      }
      if (score > bestMotionScore) {
        bestMotionScore = score;
        bestMotionChar = char;
      }
    } else {
      const latest = history[history.length - 1];

      let shapeSim = 0;
      if (cal.shapes.length > 0) {
        const midIdx = Math.floor(cal.shapes.length / 2);
        const d = shapeDistance(latest.shape, cal.shapes[midIdx]);
        shapeSim = Math.max(0, 1 - d / 1.0);
      }

      let orientSim = 0;
      if (cal.orientations.length > 0) {
        const midIdx = Math.floor(cal.orientations.length / 2);
        const d = orientationDistance(latest.orientation, cal.orientations[midIdx]);
        orientSim = Math.max(0, 1 - d / 1.5);
      }

      const hasOrient = cal.orientations && cal.orientations.length > 0;
      // Shape (joint angles) 60%, Orientation 40% — orientation matters more now
      const score = hasOrient
        ? shapeSim * 0.6 + orientSim * 0.4
        : shapeSim;
      if (score > bestStaticScore) {
        bestStaticScore = score;
        bestStaticChar = char;
      }
    }
  }

  if ((moving || recentlyMoved) && bestMotionChar && bestMotionScore > 0.3) {
    const conf = Math.round(bestMotionScore * 100);
    return [bestMotionChar, conf];
  }

  if (bestStaticChar && bestStaticScore > 0.35) {
    const conf = Math.round(bestStaticScore * 100);
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

    // Extract 3-axis features
    const features = extractFeatures(landmarks);

    // Add to history
    featureHistory.push(features);
    while (featureHistory.length > HISTORY_MAX) featureHistory.shift();

    // Calibration recording
    if (calRecording && calCountdown <= 0) {
      calShapes.push(features.shape);
      calOrientations.push(features.orientation);
      calPositions.push(features.position);
      const elapsed = (Date.now() - calStartTime) / 1000;
      updateCalStatus(`🔴 録画中: ${calShapes.length}f (${(calDuration - elapsed).toFixed(1)}s)`);

      if (elapsed >= calDuration) {
        finishCalRecording();
      }
    }

    // Throttle recognition (every 3rd frame to reduce DTW load)
    recognizeCounter++;
    const shouldRecognize = recognizeCounter % 2 === 0;

    // Calibration test mode — show top 3 candidates for debugging
    if (currentScreen === "calibration-screen" && !calRecording && shouldRecognize) {
      const result = recognizeDebug(featureHistory);
      if (result.length > 0) {
        const top = result.slice(0, 3).map(r => `${r.char}(${r.score}%)`).join(" ");
        updateCalStatus(`認識: ${top}`);
      } else {
        updateCalStatus("✋ 手を出すと認識テスト");
      }
    }

    // Game recognition
    if (currentScreen === "game-screen" && !gameWaiting && shouldRecognize) {
      const result = recognizeDebug(featureHistory);
      const now = Date.now();

      if (result.length > 0) {
        // Check if current char is in top 3 with decent score
        const top3 = result.slice(0, 3);
        const directHit = top3.find(r => r.char === gameCurrentChar && r.score > 30);

        if (directHit) {
          // Correct answer found in top 3
          if (lastRecognized === gameCurrentChar && now - lastRecognizedTime > 150) {
            onSignRecognized(gameCurrentChar, directHit.score);
          }
          if (lastRecognized !== gameCurrentChar) {
            lastRecognizedTime = now;
          }
          lastRecognized = gameCurrentChar;
        } else {
          // Show what's being recognized
          lastRecognized = result[0].char;
          lastRecognizedTime = now;
        }

        updateGameStatus(`認識: ${result[0].char}(${result[0].score}%)`);
      } else {
        if (now - lastRecognizedTime > 600) {
          lastRecognized = "";
        }
        updateGameStatus("手を検出中... ✋");
      }
    }
  } else {
    if (currentScreen === "game-screen" && !gameWaiting) {
      updateGameStatus("手が見えません 👀");
    }
    if (currentScreen === "calibration-screen" && !calRecording) {
      updateCalStatus("手が見えません 👀");
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

  for (let r = 0; r < GOJUON.length; r++) {
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
  calShapes = [];
  calOrientations = [];
  calPositions = [];
  calDuration = parseFloat(document.getElementById("cal-duration").value);
  calCountdown = 3;
  calRecording = true;

  document.querySelectorAll(".gojuon-btn").forEach(b => b.classList.remove("recording"));
  document.querySelectorAll(".gojuon-btn").forEach(b => {
    if (b.textContent.startsWith(char)) b.classList.add("recording");
  });

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
  const hasMotion = detectMotionFromPositions(calPositions) || detectMotionFromShapes(calShapes);
  const typeStr = hasMotion ? "動き" : "静止";
  updateCalStatus(`📦「${calChar}」${calShapes.length}f (${typeStr}) 保存中...`);

  document.querySelectorAll(".gojuon-btn").forEach(b => b.classList.remove("recording"));

  requestAnimationFrame(() => {
    calibrationData[calChar] = {
      shapes: calShapes,
      orientations: calOrientations,
      positions: calPositions,
      hasMotion,
    };
    saveCalibration();
    // Update grid after save
    setTimeout(() => {
      updateCalStatus(`✅「${calChar}」録画完了！ (${typeStr}, ${calShapes.length}f)`);
      updateGojuonGrid();
    }, 50);
  });
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
  gamePaused = false;
  featureHistory = [];
  lastRecognized = "";

  showScreen("game-screen");
  startCamera(
    document.getElementById("game-video"),
    document.getElementById("game-canvas")
  );

  if (mode === "word") {
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
    showBanner("🎉 " + gameWord, "correct");
    setTimeout(() => {
      hideBanner();
      const available = WORDS.filter(w =>
        [...w].every(c => calibrationData[c] || c === "　")
      );
      gameWord = available[Math.floor(Math.random() * available.length)];
      gameWordIndex = 0;
      nextWordChar();
    }, 800);
    return;
  }

  gameCurrentChar = gameWord[gameWordIndex];
  gameTotal++;
  document.getElementById("prompt-char").textContent = gameCurrentChar;

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
  if (char !== gameCurrentChar) return;

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
    }, 400);
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
  }, 1000);
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
  gamePaused = false;
  if (camera) camera.stop();
  showScreen("title-screen");
}

function pauseToCalibration() {
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

async function init() {
  await loadCalibration();

  // Auto-import from file if empty
  if (Object.keys(calibrationData).length === 0) {
    try {
      const r = await fetch("calibration_data.json");
      if (r.ok) {
        const data = await r.json();
        for (const [char, val] of Object.entries(data)) {
          if (val.shapes) {
            calibrationData[char] = val;
          } else if (val.frames) {
            calibrationData[char] = migrateV1(char, val);
          }
        }
        await saveCalibration();
      }
    } catch (e) {}
  }

  // Show admin buttons
  if (IS_ADMIN) {
    const calMenuBtn = document.getElementById("cal-menu-btn");
    const calGameBtn = document.getElementById("cal-game-btn");
    if (calMenuBtn) calMenuBtn.style.display = "";
    if (calGameBtn) calGameBtn.style.display = "";
    const randomBtn = document.getElementById("random-btn");
    if (randomBtn) randomBtn.style.display = "";
  }

  initHands();
  updateSignCount();
}

init();
