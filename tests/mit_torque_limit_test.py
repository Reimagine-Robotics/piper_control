"""Tests for the firmware-version-aware MIT torque wire limit."""

from piper_control import piper_control


def test_mit_wire_torque_limit_matches_firmware_frame() -> None:
  assert piper_control.mit_wire_torque_limit("1.8.post7") == 8.0
  assert piper_control.mit_wire_torque_limit("1.8.post8") == 16.0
  assert piper_control.mit_wire_torque_limit("1.8.post9") == 16.0


def test_mit_wire_torque_limit_handles_device_string_formats() -> None:
  assert piper_control.mit_wire_torque_limit("1.8-8") == 16.0
  assert piper_control.mit_wire_torque_limit(None) == 8.0
