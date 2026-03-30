extends Control

# Main menu / title screen

func _ready() -> void:
	# Reset game state when returning to menu
	GameManager.reset()

func _on_start_button_pressed() -> void:
	get_tree().change_scene_to_file("res://scenes/game.tscn")
