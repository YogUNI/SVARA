"""Unit tests for simulated smart home device state transitions (Task P8-02).

Reference: docs/05 §3, docs/05 §8
"""

import json
import pytest

from svara.serve.device_state import get_initial_home_state, transition_device_state


def test_initial_device_state():
    state = get_initial_home_state()
    assert "kitchen" in state["rooms"]
    assert "bedroom" in state["rooms"]
    assert "washroom" in state["rooms"]
    assert state["rooms"]["kitchen"]["lights"]["state"] == "off"
    assert state["global"]["music"]["state"] == "stopped"


def test_lights_state_transitions():
    state = get_initial_home_state()

    # Turn on kitchen lights
    new_state, effect = transition_device_state(state, "activate", "lights", "kitchen")
    assert new_state["rooms"]["kitchen"]["lights"]["state"] == "on"
    assert effect["type"] == "device"
    assert any(c["device"] == "lights" and c["room"] == "kitchen" and c["state"] == "on" for c in effect["changes"])

    # Turn off kitchen lights
    off_state, off_effect = transition_device_state(new_state, "deactivate", "lights", "kitchen")
    assert off_state["rooms"]["kitchen"]["lights"]["state"] == "off"


def test_location_none_broadcasts_to_all_rooms():
    state = get_initial_home_state()

    # Turn on lights in all rooms (location='none')
    new_state, effect = transition_device_state(state, "activate", "lights", "none")
    assert new_state["rooms"]["kitchen"]["lights"]["state"] == "on"
    assert new_state["rooms"]["bedroom"]["lights"]["state"] == "on"
    assert new_state["rooms"]["washroom"]["lights"]["state"] == "on"
    assert len(effect["changes"]) == 3


def test_heat_temperature_adjustment():
    state = get_initial_home_state()
    cur_temp = state["rooms"]["bedroom"]["heat"]["target_temp"]

    # Increase heat in bedroom
    new_state, _ = transition_device_state(state, "increase", "heat", "bedroom")
    assert new_state["rooms"]["bedroom"]["heat"]["target_temp"] == cur_temp + 1

    # Decrease heat in bedroom
    dec_state, _ = transition_device_state(new_state, "decrease", "heat", "bedroom")
    assert dec_state["rooms"]["bedroom"]["heat"]["target_temp"] == cur_temp


def test_assistant_bring_tasks():
    state = get_initial_home_state()
    new_state, effect = transition_device_state(state, "bring", "juice", "none")

    assert effect["type"] == "assistant_task"
    assert "juice" in effect["message"].lower()


def test_all_fsc_intents_have_handlers():
    """Verify all 31 intents defined in configs/intent_map.json produce valid effects (docs/05 §8)."""
    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        imap = json.load(f)

    state = get_initial_home_state()
    for intent_id, details in imap["intent_map"].items():
        _, effect = transition_device_state(
            state,
            action=details["action"],
            obj=details["object"],
            location=details["location"],
        )
        assert effect["type"] in ("device", "assistant_task", "ui_language"), (
            f"Intent {intent_id} ({details}) returned unhandled effect type: {effect['type']}"
        )
