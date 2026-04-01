extends Node

# Game state autoload singleton
# Fetches available characters from the calibration server

var score: int = 0
var total_questions: int = 0
var current_char: String = ""
var is_playing: bool = false
var characters: Array[String] = []

const SERVER_URL = "http://127.0.0.1:8765"

func _ready() -> void:
	reset()

func reset() -> void:
	score = 0
	total_questions = 0
	current_char = ""
	is_playing = false

func fetch_characters(callback: Callable) -> void:
	"""Fetch calibrated signs from server."""
	var http = HTTPRequest.new()
	add_child(http)
	http.timeout = 3.0
	http.request_completed.connect(func(result, code, _headers, body):
		if result == HTTPRequest.RESULT_SUCCESS and code == 200:
			var json = JSON.new()
			if json.parse(body.get_string_from_utf8()) == OK:
				var data = json.data
				if data is Dictionary:
					characters.clear()
					for key in data.keys():
						characters.append(key)
		# Fallback if server unreachable
		if characters.is_empty():
			characters = ["あ", "い", "う", "え", "お"]
		http.queue_free()
		callback.call()
	)
	http.request(SERVER_URL + "/signs")

func next_question() -> String:
	current_char = characters.pick_random()
	total_questions += 1
	is_playing = true
	return current_char

func answer_correct() -> void:
	score += 1

func get_score_text() -> String:
	return "%d / %d" % [score, total_questions]
