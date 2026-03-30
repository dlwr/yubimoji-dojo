extends Node

# Game state autoload singleton

var score: int = 0
var total_questions: int = 0
var current_char: String = ""
var is_playing: bool = false

# Phase 1: あいうえお only
var characters: Array[String] = ["あ", "い", "う", "え", "お"]

func _ready() -> void:
	reset()

func reset() -> void:
	score = 0
	total_questions = 0
	current_char = ""
	is_playing = false

func next_question() -> String:
	current_char = characters.pick_random()
	total_questions += 1
	is_playing = true
	return current_char

func answer_correct() -> void:
	score += 1

func get_score_text() -> String:
	return "%d / %d" % [score, total_questions]
