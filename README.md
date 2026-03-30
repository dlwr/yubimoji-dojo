# 指文字道場 (Yubimoji Dojo)

カメラで日本語の指文字を練習するゲーム 🤟

## セットアップ

### 1. 手認識サーバーを起動

```bash
cd tools
pip install -r requirements.txt
python hand_detector_server.py
```

カメラが起動して `http://127.0.0.1:8765` でサーバーが立ち上がる。

### 2. Godotでゲームを起動

```bash
godot --path . --editor
# またはシーンを直接実行:
godot --path . scenes/main_menu.tscn
```

## 遊び方

1. 手認識サーバーを先に起動
2. ゲームを起動して「始める」を押す
3. 画面右に表示される文字の指文字をカメラに向かって作る
4. 正解すると次の問題へ（10問で終了）
5. 制限時間は1問10秒

## Phase 1 対応文字

| 文字 | 指文字 |
|------|--------|
| あ | グー（全部閉じる） |
| い | 小指だけ伸ばす |
| う | 人差し指＋中指（ピース） |
| え | 人差し指だけ伸ばす |
| お | パー（全部開く） |

## 技術構成

- **Godot 4.x** — ゲームエンジン（GDScript）
- **Python + MediaPipe** — 手のランドマーク検出
- **Flask** — Godot⇔Python間のHTTP通信
- **OpenCV** — カメラキャプチャ

## ディレクトリ構成

```
yubimoji-dojo/
├── project.godot          # Godotプロジェクト設定
├── scenes/
│   ├── main_menu.tscn/gd  # タイトル画面
│   └── game.tscn/gd       # ゲーム画面
├── scripts/
│   ├── game_manager.gd    # ゲーム状態管理（Autoload）
│   ├── hand_detector.gd   # Python サーバーとの通信
│   └── finger_sign_data.gd # 指文字パターン定義
└── tools/
    ├── hand_detector_server.py  # MediaPipe手認識サーバー
    └── requirements.txt
```

## 今後の予定

- [ ] Phase 2: 50音全対応
- [ ] ドット絵のお手本表示
- [ ] 演出・SE・BGM
- [ ] Steam配信
