"""Tests for the firmware-version-aware MIT torque limits and gear ratios."""

from piper_control import piper_control


def test_gear_ratio_band_scales_j1_to_j3() -> None:
  # <= S-V1.8-2: firmware applies the gear ratio, so J1-3 reach +/-32 Nm and
  # commands are divided by the ratio before encoding.
  limits, divisors = piper_control.mit_torque_limits("1.8.post2")
  assert limits == [32.0, 32.0, 32.0, 8.0, 8.0, 8.0]
  assert divisors == [4.0, 4.0, 4.0, 1.0, 1.0, 1.0]


def test_v183_band_drops_gear_ratio() -> None:
  # S-V1.8-3 .. S-V1.8-7: 8-bit field, no gear ratio, +/-8 Nm.
  limits, divisors = piper_control.mit_torque_limits("1.8.post5")
  assert limits == [8.0] * 6
  assert divisors == [1.0] * 6


def test_12bit_band_widens_to_16nm() -> None:
  limits, divisors = piper_control.mit_torque_limits("1.8.post8")
  assert limits == [16.0] * 6
  assert divisors == [1.0] * 6


def test_handles_device_string_and_unknown_firmware() -> None:
  # "1.8-8" parses as post8 -> 12-bit band.
  assert piper_control.mit_torque_limits("1.8-8")[0] == [16.0] * 6
  # Unknown firmware falls back to the gear-ratio band (safe under-command).
  _, divisors = piper_control.mit_torque_limits(None)
  assert divisors == [4.0, 4.0, 4.0, 1.0, 1.0, 1.0]
