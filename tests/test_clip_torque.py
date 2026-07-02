"""Tests for CAN torque clipping in MitJointPositionController.

Torques beyond the CAN message limit overflow the field and flip sign, so
``_clip_torque`` hard-clips every commanded torque and warns (once per
excursion) when it does. NaN torques are rejected outright. These tests pin
that behaviour; the method is pure aside from its per-instance warn latch and
needs no arm, so instances are built without running __init__ (which would talk
to hardware).

Run with plugin autoload disabled to avoid unrelated system pytest plugins
(e.g. a ROS install on the PYTHONPATH) failing to import:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
"""

import logging

import pytest

from piper_control import piper_control as pc


def _make_controller():
  """Build a controller with just the state _clip_torque needs.

  ``__init__`` queries the arm's firmware over the wire, so we bypass it and set
  only the per-instance warn latch the method reads and writes.
  """
  controller = pc.MitJointPositionController.__new__(pc.MitJointPositionController)
  controller._torque_clip_warned = [False] * len(pc._MIT_TORQUE_LIMITS)
  return controller


def test_can_limit_is_8nm():
  # The CAN field saturates at 8 Nm; a higher limit would let the sign flip.
  assert pc._MIT_TORQUE_LIMITS == [8.0] * 6


def test_in_range_torque_is_unchanged():
  c = _make_controller()
  assert c._clip_torque(5.0, 0) == 5.0
  assert c._clip_torque(-8.0, 3) == -8.0
  assert c._clip_torque(0.0, 5) == 0.0


def test_over_limit_is_clipped_to_limit():
  c = _make_controller()
  assert c._clip_torque(12.0, 0) == 8.0
  assert c._clip_torque(-100.0, 2) == -8.0


def test_nan_torque_is_rejected():
  c = _make_controller()
  with pytest.raises(ValueError, match="NaN"):
    c._clip_torque(float("nan"), 4)
  # A NaN must not silently latch the warn state as if it were a clip.
  assert c._torque_clip_warned[4] is False


def test_warns_when_clipping(caplog):
  c = _make_controller()
  with caplog.at_level(logging.WARNING):
    c._clip_torque(12.0, 1)
  warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
  assert len(warnings) == 1
  assert "joint 1" in warnings[0].getMessage()


def test_does_not_warn_in_range(caplog):
  c = _make_controller()
  with caplog.at_level(logging.WARNING):
    c._clip_torque(5.0, 0)
  assert not caplog.records


def test_warns_once_per_excursion(caplog):
  # Simulate a sustained over-limit command at the control rate: only the first
  # tick should warn.
  c = _make_controller()
  with caplog.at_level(logging.WARNING):
    for _ in range(200):
      c._clip_torque(12.0, 0)
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_rearms_after_returning_in_range(caplog):
  c = _make_controller()
  with caplog.at_level(logging.WARNING):
    c._clip_torque(12.0, 0)  # first excursion -> warns
    c._clip_torque(5.0, 0)  # back in range -> re-arms
    c._clip_torque(12.0, 0)  # new excursion -> warns again
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_per_joint_warn_state_is_independent(caplog):
  c = _make_controller()
  with caplog.at_level(logging.WARNING):
    c._clip_torque(12.0, 0)
    c._clip_torque(12.0, 1)
  # Both joints warn on their own first excursion.
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_instances_do_not_share_warn_state(caplog):
  # Two controllers (e.g. two arms) must not suppress each other's first
  # warning — the latch is per-instance.
  a = _make_controller()
  b = _make_controller()
  with caplog.at_level(logging.WARNING):
    a._clip_torque(12.0, 0)  # latches on instance a
    b._clip_torque(12.0, 0)  # instance b must still warn
  assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
