extends Node
class_name HandDetector

# Polls the Python MediaPipe sidecar server for hand landmarks.
# The sidecar runs on localhost:8765 and returns 21 hand landmarks as JSON.

signal landmarks_updated(landmarks: Array)
signal hand_detected()
signal hand_lost()
signal sign_recognized(character: String)

const SERVER_URL = "http://127.0.0.1:8765"
const POLL_INTERVAL = 0.15  # seconds between polls

var _http_request: HTTPRequest
var _poll_timer: Timer
var _has_hand: bool = false
var _landmarks: Array = []

func _ready() -> void:
	_http_request = HTTPRequest.new()
	_http_request.timeout = 1.0
	add_child(_http_request)
	_http_request.request_completed.connect(_on_request_completed)

	_poll_timer = Timer.new()
	_poll_timer.wait_time = POLL_INTERVAL
	_poll_timer.timeout.connect(_poll_landmarks)
	add_child(_poll_timer)

func start() -> void:
	_poll_timer.start()

func stop() -> void:
	_poll_timer.stop()

func _poll_landmarks() -> void:
	if _http_request.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return  # Still waiting for previous request
	_http_request.request(SERVER_URL + "/landmarks")

func _on_request_completed(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		if _has_hand:
			_has_hand = false
			hand_lost.emit()
		return

	var json = JSON.new()
	var err = json.parse(body.get_string_from_utf8())
	if err != OK:
		return

	var data = json.data
	if not data is Dictionary:
		return

	if data.get("detected", false):
		_landmarks = data.get("landmarks", [])
		if not _has_hand:
			_has_hand = true
			hand_detected.emit()
		landmarks_updated.emit(_landmarks)

		# Check for recognized sign
		var recognized = data.get("recognized", "")
		if recognized != "":
			sign_recognized.emit(recognized)
	else:
		if _has_hand:
			_has_hand = false
			hand_lost.emit()
