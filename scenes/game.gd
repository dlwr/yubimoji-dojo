extends Control

# Main game scene
# Left: camera feed from Python server, Right: prompt character

@onready var prompt_label: Label = $GameLayout/MainArea/RightPanel/PromptChar
@onready var score_label: Label = $GameLayout/TopBar/ScoreLabel
@onready var status_label: Label = $GameLayout/BottomBar/StatusLabel
@onready var timer_label: Label = $GameLayout/TopBar/TimerLabel
@onready var camera_texture: TextureRect = $GameLayout/MainArea/LeftPanel/CameraView
@onready var hand_detector: HandDetector = $HandDetector

const TIME_PER_QUESTION: float = 10.0
var time_remaining: float = 0.0
var waiting_for_next: bool = false
var game_active: bool = false

# Camera frame fetching
var _frame_request: HTTPRequest
var _frame_timer: Timer

func _ready() -> void:
	hand_detector.sign_recognized.connect(_on_sign_recognized)
	hand_detector.hand_detected.connect(_on_hand_detected)
	hand_detector.hand_lost.connect(_on_hand_lost)

	# Setup camera frame fetcher
	_frame_request = HTTPRequest.new()
	_frame_request.timeout = 2.0
	add_child(_frame_request)
	_frame_request.request_completed.connect(_on_frame_received)

	_frame_timer = Timer.new()
	_frame_timer.wait_time = 0.1  # ~10 FPS
	_frame_timer.timeout.connect(_fetch_frame)
	add_child(_frame_timer)
	_frame_timer.start()

	# Start hand detection polling
	hand_detector.start()

	# Start first question
	_next_question()

func _fetch_frame() -> void:
	if _frame_request.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return
	_frame_request.request("http://127.0.0.1:8765/frame")

func _on_frame_received(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		return
	# Load JPEG image
	var img = Image.new()
	var err = img.load_jpg_from_buffer(body)
	if err == OK:
		var tex = ImageTexture.create_from_image(img)
		camera_texture.texture = tex

func _next_question() -> void:
	waiting_for_next = false
	game_active = true
	var char = GameManager.next_question()
	prompt_label.text = char
	score_label.text = "スコア: " + GameManager.get_score_text()
	time_remaining = TIME_PER_QUESTION
	status_label.text = "「%s」を指文字で作ってください ✋" % char

func _process(delta: float) -> void:
	if game_active and not waiting_for_next:
		time_remaining -= delta
		timer_label.text = "残り: %d秒" % int(ceil(time_remaining))
		if time_remaining <= 0:
			_on_timeout()

func _on_sign_recognized(character: String) -> void:
	if waiting_for_next or not game_active:
		return

	if character == GameManager.current_char:
		# Correct!
		GameManager.answer_correct()
		score_label.text = "スコア: " + GameManager.get_score_text()
		status_label.text = "正解！ ✨"
		_wait_and_next(1.5)
	elif character != "":
		status_label.text = "惜しい！ それは「%s」ずら" % character

func _on_hand_detected() -> void:
	if not waiting_for_next and game_active:
		status_label.text = "「%s」を指文字で作ってください ✋" % GameManager.current_char

func _on_hand_lost() -> void:
	if not waiting_for_next and game_active:
		status_label.text = "手が見えません 👀"

func _on_timeout() -> void:
	if waiting_for_next:
		return
	status_label.text = "時間切れ！ 答えは「%s」" % GameManager.current_char
	_wait_and_next(2.0)

func _wait_and_next(delay: float) -> void:
	waiting_for_next = true
	await get_tree().create_timer(delay).timeout
	if GameManager.total_questions >= 10:
		_show_results()
	else:
		_next_question()

func _show_results() -> void:
	game_active = false
	prompt_label.text = "終了！"
	var pct = 0
	if GameManager.total_questions > 0:
		pct = GameManager.score * 100 / GameManager.total_questions
	status_label.text = "結果: %d問中%d問正解 (%d%%)" % [
		GameManager.total_questions,
		GameManager.score,
		pct,
	]
	timer_label.text = ""

	await get_tree().create_timer(5.0).timeout
	get_tree().change_scene_to_file("res://scenes/main_menu.tscn")

func _on_back_button_pressed() -> void:
	hand_detector.stop()
	_frame_timer.stop()
	get_tree().change_scene_to_file("res://scenes/main_menu.tscn")
