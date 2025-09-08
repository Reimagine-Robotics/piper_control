"""Functions for enabling, disabling, and resetting the arm and gripper.

These helper methods are blocking calls as they perform multiple retries and
sleeps, using piper_interface under the hood.
"""

import time
from collections.abc import Callable

from piper_control import piper_interface as pi

_SHORT_WAIT = 0.1
_LONG_WAIT = 0.5


def _create_timeout(
    seconds: float,
    message: str = "Timeout",
) -> Callable[[], None]:
  """Creates a timeout trigger that raises a TimeoutError after a given time."""
  start_time = time.time()

  def timeout_trigger() -> None:
    if time.time() - start_time > seconds:
      raise TimeoutError(message)

  return timeout_trigger


def disable_gripper(
    piper: pi.PiperInterface,
    *,
    timeout_seconds: float = 10.0,
) -> None:
  """
  Disables the gripper.
  """
  timeout_trigger = _create_timeout(timeout_seconds, "disable_gripper")
  while True:
    piper.disable_gripper()
    time.sleep(_SHORT_WAIT)
    if not piper.is_gripper_enabled():
      break

    timeout_trigger()
    time.sleep(_LONG_WAIT)


def enable_gripper(
    piper: pi.PiperInterface,
    *,
    timeout_seconds: float = 10.0,
) -> None:
  """
  Enables the gripper.
  """
  timeout_trigger = _create_timeout(timeout_seconds, "enable_gripper")
  while True:
    piper.enable_gripper()
    time.sleep(_SHORT_WAIT)
    if piper.is_gripper_enabled():
      break

    timeout_trigger()
    # Try disabling the gripper to clear any error state, then try again.
    piper.disable_gripper()
    time.sleep(_LONG_WAIT)


def reset_gripper(
    piper: pi.PiperInterface,
    *,
    timeout_seconds: float = 10.0,
) -> None:
  """
  Resets the gripper.
  """

  # First disable the gripper to clear any errors then enable.
  disable_gripper(piper, timeout_seconds=timeout_seconds)
  time.sleep(_SHORT_WAIT)
  enable_gripper(piper, timeout_seconds=timeout_seconds)


def disable_arm(
    piper: pi.PiperInterface,
    *,
    timeout_seconds: float = 10.0,
) -> None:
  """
  Disables the arm.

  WARNING: This powers down the arm and it will drop if not supported.
  """
  timeout_trigger = _create_timeout(timeout_seconds, "disable_arm")
  while True:
    piper.set_emergency_stop(pi.EmergencyStop.RESUME)
    time.sleep(_SHORT_WAIT)

    if (
        piper.control_mode == pi.ControlMode.STANDBY
        and piper.arm_status == pi.ArmStatus.NORMAL
        and piper.teach_status == pi.TeachStatus.OFF
    ):
      break

    timeout_trigger()
    time.sleep(_LONG_WAIT)


def enable_arm(
    piper: pi.PiperInterface,
    arm_controller: pi.ArmController = pi.ArmController.POSITION_VELOCITY,
    move_mode: pi.MoveMode = pi.MoveMode.JOINT,
    *,
    timeout_seconds: float = 10.0,
):
  """
  Enables the arm.

  Args:
    arm_controller (ArmController): The arm controller to use. This can be
      either MIT mode which provides raw motor commands, or POSITION_VELOCITY
      which uses inbuilt controllers for motion.
    move_mode (MoveMode): The move mode to use. This indicates whether the arm
      is to be commanded by specifying target joint angles, end-effector
      cartesian poses, or the raw MIT mode.
  """
  timeout_trigger = _create_timeout(timeout_seconds, "enable_arm")
  while True:
    piper.enable_arm()
    time.sleep(_SHORT_WAIT)
    if piper.is_arm_enabled():
      break

    timeout_trigger()
    time.sleep(_LONG_WAIT)

  # enable motion
  # Don't loop on this because sometimes (eg when coming out of teach mode) it
  # fails to set the control mode. If that happens you need to call
  # disable_arm() and then call this function again. This is not done
  # automatically because disable_arm() will cause the arm to lose power and
  # crash down if it is not supported, so it's pretty rude to do that
  # unexpectedly.
  piper.piper.MotionCtrl_2(
      pi.ControlMode.CAN_COMMAND.value,
      move_mode.value,
      100,  # move speed
      arm_controller.value,
  )
  time.sleep(_LONG_WAIT)


def reset_arm(
    piper: pi.PiperInterface,
    arm_controller: pi.ArmController = pi.ArmController.POSITION_VELOCITY,
    move_mode: pi.MoveMode = pi.MoveMode.JOINT,
    *,
    timeout_seconds: float = 10.0,
) -> None:
  """
  Resets the arm.

  WARNING: This depowers the arm and it will drop if not supported.

  Args:
    arm_controller (ArmController): The arm controller to use when enabling.
    move_mode (MoveMode): The move mode to use when enabling.
  """
  timeout_trigger = _create_timeout(timeout_seconds, "reset_arm")
  while True:
    disable_arm(
        piper,
        timeout_seconds=timeout_seconds,
    )

    enable_arm(
        piper,
        arm_controller=arm_controller,
        move_mode=move_mode,
        timeout_seconds=timeout_seconds,
    )

    # Sometimes (eg when coming out of teach mode) the first round of resets
    # don't work, keep going until they do.
    if piper.control_mode == pi.ControlMode.CAN_COMMAND:
      break

    timeout_trigger()
