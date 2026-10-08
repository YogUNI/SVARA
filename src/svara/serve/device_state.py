"""Simulated residential smart home state and pure functional state transitions.

Rooms: kitchen, bedroom, washroom, and global (for location 'none').
Objects: lights, lamp, heat, music, volume, assistant tasks (juice, newspaper, socks, shoes), language badges.

Task: P8-02
Reference: docs/05 §3, docs/11 V-43
"""

import copy
from typing import Any, Dict, List, Tuple


def get_initial_home_state() -> Dict[str, Any]:
    """Return default simulated home state."""
    return {
        "rooms": {
            "kitchen": {
                "lights": {"state": "off", "brightness": 60},
                "lamp": {"state": "off"},
                "heat": {"state": "off", "target_temp": 22},
            },
            "bedroom": {
                "lights": {"state": "off", "brightness": 60},
                "lamp": {"state": "off"},
                "heat": {"state": "off", "target_temp": 22},
            },
            "washroom": {
                "lights": {"state": "off", "brightness": 60},
                "lamp": {"state": "off"},
                "heat": {"state": "off", "target_temp": 22},
            },
        },
        "global": {
            "music": {"state": "stopped"},
            "volume": {"level": 5},
            "ui_language": "English",
        },
    }


def transition_device_state(
    current_state: Dict[str, Any],
    action: str,
    obj: str,
    location: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pure functional state transition: (state, intent) -> (new_state, effect).

    Returns:
        new_state: Updated state copy.
        effect: Action effect dictionary with type, changes, and message.
    """
    new_state = copy.deepcopy(current_state)
    target_rooms = [location] if location in new_state["rooms"] else list(new_state["rooms"].keys())

    # 1. Lights
    if obj == "lights":
        changes = []
        if action == "activate":
            for r in target_rooms:
                new_state["rooms"][r]["lights"]["state"] = "on"
                changes.append({"device": "lights", "room": r, "state": "on"})
            msg = f"Turned on lights in {location if location != 'none' else 'all rooms'}."
        elif action == "deactivate":
            for r in target_rooms:
                new_state["rooms"][r]["lights"]["state"] = "off"
                changes.append({"device": "lights", "room": r, "state": "off"})
            msg = f"Turned off lights in {location if location != 'none' else 'all rooms'}."
        elif action == "increase":
            for r in target_rooms:
                b = min(100, new_state["rooms"][r]["lights"]["brightness"] + 20)
                new_state["rooms"][r]["lights"]["brightness"] = b
                new_state["rooms"][r]["lights"]["state"] = "on"
                changes.append({"device": "lights", "room": r, "state": f"{b}%"})
            msg = f"Increased lights brightness in {location if location != 'none' else 'all rooms'}."
        elif action == "decrease":
            for r in target_rooms:
                b = max(10, new_state["rooms"][r]["lights"]["brightness"] - 20)
                new_state["rooms"][r]["lights"]["brightness"] = b
                changes.append({"device": "lights", "room": r, "state": f"{b}%"})
            msg = f"Decreased lights brightness in {location if location != 'none' else 'all rooms'}."
        else:
            return current_state, {"type": "none", "changes": [], "message": "Unsupported action on lights."}

        return new_state, {"type": "device", "changes": changes, "message": msg}

    # 2. Lamp
    elif obj == "lamp":
        changes = []
        target_state = "on" if action == "activate" else "off"
        for r in target_rooms:
            new_state["rooms"][r]["lamp"]["state"] = target_state
            changes.append({"device": "lamp", "room": r, "state": target_state})
        msg = f"Turned {target_state} lamp in {location if location != 'none' else 'all rooms'}."
        return new_state, {"type": "device", "changes": changes, "message": msg}

    # 3. Heat
    elif obj == "heat":
        changes = []
        if action == "activate":
            for r in target_rooms:
                new_state["rooms"][r]["heat"]["state"] = "on"
                changes.append({"device": "heat", "room": r, "state": "on"})
            msg = f"Turned on heat in {location if location != 'none' else 'all rooms'}."
        elif action == "deactivate":
            for r in target_rooms:
                new_state["rooms"][r]["heat"]["state"] = "off"
                changes.append({"device": "heat", "room": r, "state": "off"})
            msg = f"Turned off heat in {location if location != 'none' else 'all rooms'}."
        elif action == "increase":
            for r in target_rooms:
                temp = min(30, new_state["rooms"][r]["heat"]["target_temp"] + 1)
                new_state["rooms"][r]["heat"]["target_temp"] = temp
                new_state["rooms"][r]["heat"]["state"] = "on"
                changes.append({"device": "heat", "room": r, "state": f"{temp}°C"})
            msg = f"Increased temperature in {location if location != 'none' else 'all rooms'}."
        elif action == "decrease":
            for r in target_rooms:
                temp = max(16, new_state["rooms"][r]["heat"]["target_temp"] - 1)
                new_state["rooms"][r]["heat"]["target_temp"] = temp
                changes.append({"device": "heat", "room": r, "state": f"{temp}°C"})
            msg = f"Decreased temperature in {location if location != 'none' else 'all rooms'}."
        else:
            return current_state, {"type": "none", "changes": [], "message": "Unsupported action on heat."}

        return new_state, {"type": "device", "changes": changes, "message": msg}

    # 4. Music
    elif obj == "music":
        state_str = "playing" if action == "activate" else "stopped"
        new_state["global"]["music"]["state"] = state_str
        return new_state, {
            "type": "device",
            "changes": [{"device": "music", "room": "global", "state": state_str}],
            "message": f"Music is now {state_str}.",
        }

    # 5. Volume
    elif obj == "volume":
        cur_vol = new_state["global"]["volume"]["level"]
        delta = 1 if action == "increase" else -1
        new_vol = max(0, min(10, cur_vol + delta))
        new_state["global"]["volume"]["level"] = new_vol
        return new_state, {
            "type": "device",
            "changes": [{"device": "volume", "room": "global", "state": new_vol}],
            "message": f"Volume set to {new_vol}.",
        }

    # 6. Assistant Tasks (bring)
    elif action == "bring" and obj in ("juice", "newspaper", "socks", "shoes"):
        return new_state, {
            "type": "assistant_task",
            "changes": [],
            "message": f"Request received: Bringing {obj}.",
        }

    # 7. Language UI badge
    elif action == "change language":
        lang_target = obj if obj != "none" else "English"
        new_state["global"]["ui_language"] = lang_target
        return new_state, {
            "type": "ui_language",
            "changes": [{"device": "ui_language", "room": "global", "state": lang_target}],
            "message": f"Interface language set to {lang_target}." if obj != "none" else "Opened language selection menu.",
        }

    # Fallback unrecognized
    return new_state, {
        "type": "none",
        "changes": [],
        "message": f"Command recognized ({action} {obj} {location}), but has no simulated effect.",
    }
