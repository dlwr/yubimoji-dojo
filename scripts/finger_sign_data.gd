extends Node
class_name FingerSignData

# Japanese Sign Language (JSL) finger spelling patterns
# Based on MediaPipe hand landmark positions (21 points)
#
# Landmark indices:
#   0: WRIST
#   1-4: THUMB (CMC, MCP, IP, TIP)
#   5-8: INDEX (MCP, PIP, DIP, TIP)
#   9-12: MIDDLE (MCP, PIP, DIP, TIP)
#   13-16: RING (MCP, PIP, DIP, TIP)
#   17-20: PINKY (MCP, PIP, DIP, TIP)
#
# For Phase 1, we use simple heuristics:
# - Which fingers are extended (tip y < pip y for extended)
# - Relative finger positions

# Returns which fingers are extended based on landmarks
static func get_extended_fingers(landmarks: Array) -> Dictionary:
	if landmarks.size() < 21:
		return {}

	# Each landmark is {x, y, z}
	var thumb_extended = landmarks[4]["x"] < landmarks[3]["x"]  # Thumb tip left of IP (right hand)
	var index_extended = landmarks[8]["y"] < landmarks[6]["y"]
	var middle_extended = landmarks[12]["y"] < landmarks[10]["y"]
	var ring_extended = landmarks[16]["y"] < landmarks[14]["y"]
	var pinky_extended = landmarks[20]["y"] < landmarks[18]["y"]

	return {
		"thumb": thumb_extended,
		"index": index_extended,
		"middle": middle_extended,
		"ring": ring_extended,
		"pinky": pinky_extended,
	}

# Recognize which character is being signed
# Returns empty string if no match
static func recognize(landmarks: Array) -> String:
	var fingers = get_extended_fingers(landmarks)
	if fingers.is_empty():
		return ""

	var t = fingers["thumb"]
	var i = fingers["index"]
	var m = fingers["middle"]
	var r = fingers["ring"]
	var p = fingers["pinky"]

	# JSL Finger spelling (simplified Phase 1):
	#
	# あ (a): Fist with thumb on the side (no fingers extended)
	# → All fingers closed, thumb may be slightly out
	if not i and not m and not r and not p and not t:
		return "あ"

	# い (i): Pinky extended, rest closed
	if not t and not i and not m and not r and p:
		return "い"

	# う (u): Index and middle extended together, rest closed
	if not t and i and m and not r and not p:
		return "う"

	# え (e): Index extended and bent, rest closed
	# Simplified: only index extended
	if not t and i and not m and not r and not p:
		return "え"

	# お (o): All fingers slightly curved (all extended but close together)
	# Simplified: all fingers extended
	if i and m and r and p:
		return "お"

	return ""
