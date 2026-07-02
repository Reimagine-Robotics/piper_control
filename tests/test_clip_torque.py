"""Tests for CAN torque clipping in piper_control.

Torques beyond the CAN message limit overflow the field and flip sign, so
``_clip_torque`` hard-clips every commanded torque and warns (once per
excursion) when it does. These tests pin that behaviour; the function is pure
and hardware-free, so no arm is required.
"""

import logging

import pytest

from piper_control import piper_control as pc


@pytest.fixture(autouse=True)
def _reset_warn_state():
  """Reset the module-level warn-latch so tests don't leak state into others."""
  pc._torque_clip_warned = [False] * len(pc._MIT_TORQUE_LIMITS)
  yield


def test_can_limit_is_8nm():
  # The CAN field saturates at 8 Nm; a higher limit would let the sign flip.
  assert pc._MIT_TORQUE_LIMITS == [8.0] * 6


def test_in_range_torque_is_unchanged():
  assert pc._clip_torque(5.0, 0) == 5.0
  assert pc._clip_torque(-8.0, 3) == -8.0
  assert pc._clip_torque(0.0, 5) == 0.0


def test_over_limit_is_clipped_to_limit():
  assert pc._clip_torque(12.0, 0) == 8.0
  assert pc._clip_torque(-100.0, 2) == -8.0


def test_warns_when_clipping(caplog):
  with caplog.at_level(logging.WARNING):
    pc._clip_torque(12.0, 1)
  warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
  assert len(warnings) == 1
  assert "joint 1" in warnings[0].getMessage()


def test_does_not_warn_in_range(caplog):
  with caplog.at_level(logging.WARNING):
    pc._clip_torque(5.0, 0)
  assert not caplog.records


def test_warns_once_per_excursion(caplog):
  # Simulate a sustained over-limit command at the control rate: only the first
  # tick should warn.
  with caplog.at_level(logging.WARNING):
    for _ in range(200):
      pc._clip_torque(12.0, 0)
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_rearms_after_returning_in_range(caplog):
  with caplog.at_level(logging.WARNING):
    pc._clip_torque(12.0, 0)  # first excursion -> warns
    pc._clip_torque(5.0, 0)  # back in range -> re-arms
    pc._clip_torque(12.0, 0)  # new excursion -> warns again
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_per_joint_warn_state_is_independent(caplog):
  with caplog.at_level(logging.WARNING):
    pc._clip_torque(12.0, 0)
    pc._clip_torque(12.0, 1)
  # Both joints warn on their own first excursion.
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
