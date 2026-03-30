extends Control

# Main game scene
# Left: camera feed, Right: prompt character, Top: score, Bottom: status

@onready var prompt_label: Label = $GameLayout/RightPanel/PromptChar
@onready var score_label: Label = $GameLayout/TopBar/ScoreLabel
@onready var status_label: Label = $GameLayout/BottomBar/StatusLabel
@onready var timer_label: Label = $GameLayout/TopBar/TimerLabel
@onready var camera_texture: TextureRect = $GameLayout/LeftPanel/CameraView
@onready var hand_detector: HandDetector = $HandDetector
@onready var question_timer: Timer = $QuestionTimer

const TIME_PER_QUESTION: float = 10.0
var time_remaining: float = 0.0
var waiting_for_next: bool = false

func _ready() -> void:
	hand_detector.sign_recognized.connect(_on_sign_recognized)
	hand_detector.hand_detected.connect(_on_hand_detected)
	hand_detector.hand_lost.connect(_on_hand_lost)
	question_timer.timeout.connect(_on_question_timeout)

	# Start camera feed
	_setup_camera()

	# Start hand detection polling
	hand_detector.start()

	# Start first question
	_next_question()

func _setup_camera() -> void:
	# Access system camera via CameraServer
	var feeds = CameraServer.feeds()
	if feeds.size() > 0:
		var feed: CameraFeed = feeds[0]
		feed.set_active(true)
		var cam_tex = CameraTexture.new()
		cam_tex.camera_feed_id = feed.get_id()
		cam_tex.which_feed = CameraServer.FEED_RGBA_IMAGE
		camera_texture.texture = cam_tex
		status_label.text = "カメラ接続済み ✓"
	else:
		status_label.text = "カメラが見つかりません..."

func _next_question() -> void:
	waiting_for_next = false
	var char = GameManager.next_question()
	prompt_label.text = char
	score_label.text = "スコア: " + GameManager.get_score_text()
	time_remaining = TIME_PER_QUESTION
	question_timer.start(1.0)
	status_label.text = "「%s」を指文字で作ってください" % char

func _process(_delta: float) -> void:
	if GameManager.is_playing and not waiting_for_next:
		timer_label.text = "残り: %d秒" % int(time_remaining)

func _on_sign_recognized(character: String) -> void:
	if waiting_for_next or not GameManager.is_playing:
		return

	if character == GameManager.current_char:
		# Correct!
		GameManager.answer_correct()
		score_label.text = "スコア: " + GameManager.get_score_text()
		status_label.text = "正解！ ✨"
		_wait_and_next(1.5)
	else:
		status_label.text = "惜しい！「%s」→「%s」" % [GameManager.current_char, character]

func _on_hand_detected() -> void:
	if not waiting_for_next:
		status_label.text = "手を検出中... ✋"

func _on_hand_lost() -> void:
	if not waiting_for_next:
		status_label.text = "手が見えません 👀"

func _on_question_timeout() -> void:
	if waiting_for_next:
		return
	time_remaining -= 1.0
	if time_remaining <= 0:
		question_timer.stop()
		status_label.text = "時間切れ！ 答えは「%s」" % GameManager.current_char
		_wait_and_next(2.0)

func _wait_and_next(delay: float) -> void:
	waiting_for_next = true
	question_timer.stop()
	await get_tree().create_timer(delay).timeout
	if GameManager.total_questions >= 10:
		_show_results()
	else:
		_next_question()

func _show_results() -> void:
	GameManager.is_playing = false
	prompt_label.text = "終了！"
	status_label.text = "結果: %s （%d問中%d問正解）" % [
		GameManager.get_score_text(),
		GameManager.total_questions,
		GameManager.score
	]
	timer_label.text = ""

	# Return to menu after delay
	await get_tree().create_timer(5.0).timeout
	get_tree().change_scene_to_file("res://scenes/main_menu.tscn")

func _on_back_button_pressed() -> void:
	hand_detector.stop()
	get_tree().change_scene_to_file("res://scenes/main_menu.tscn")
