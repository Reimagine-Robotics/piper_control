"""High level controllers for the piper robot.

These controllers are provided as a convenient high-level way of controlling the
Piper robot. They provide two main benefits:
- a simplified usage interface that hides much of the underlying Piper SDK
  complexity,
- a context that upon exit will park the arm in a safe rest position.
"""

import abc
import dataclasses
import time
from collections.abc import Sequence

import numpy as np
from packaging import version as packaging_version

from piper_control import piper_interface as pi


@dataclasses.dataclass(frozen=True)
class ArmOrientation:
  """Represents an arm mounting orientation with associated data."""

  name: str
  rest_position: tuple[float, ...]  # Joint angles in radians
  mounting_quaternion: tuple[float, float, float, float]  # [w, i, j, k]


@dataclasses.dataclass(frozen=True)
class ArmOrientations:
  """Registry of standard arm orientations.

  Coordinate system: Default upright arm has +x forward, +y left, +z up.
  """

  upright: ArmOrientation = ArmOrientation(
      name="upright",
      rest_position=(0.0, 0.0, 0.0, 0.02, 0.5, 0.0),
      mounting_quaternion=(1.0, 0.0, 0.0, 0.0),  # Identity - no rotation
  )

  left: ArmOrientation = ArmOrientation(
      name="left",
      rest_position=(1.71, 2.96, -2.65, 1.41, -0.081, -0.190),
      mounting_quaternion=(0.7071068, -0.7071068, 0.0, 0.0),  # -90° around X
  )

  right: ArmOrientation = ArmOrientation(
      name="right",
      rest_position=(-1.66, 2.91, -2.74, 0.0545, -0.271, 0.0979),
      mounting_quaternion=(0.7071068, 0.7071068, 0.0, 0.0),  # +90° around X
  )

  @classmethod
  def from_string(cls, orientation_name: str) -> ArmOrientation:
    """Get ArmOrientation instance from string name.

    Args:
      orientation_name: Name of the orientation ('upright', 'left', 'right')

    Returns:
      ArmOrientation instance

    Raises:
      ValueError: If orientation_name is not recognized
    """
    orientation_name = orientation_name.lower()
    for orientation in cls.__dict__.values():
      if (
          isinstance(orientation, ArmOrientation)
          and orientation.name.lower() == orientation_name
      ):
        return orientation

    available = [
        o.name for o in cls.__dict__.values() if isinstance(o, ArmOrientation)
    ]
    raise ValueError(
        f"Unknown arm orientation: {orientation_name}. Available: {available}"
    )


DEFAULT_GRIPPER_EFFORT = 1.0  # Ostensibly in Nm.

# Control rate in Hz. Frequency at which to send joint commands to the robot.
_CONTROL_RATE = 200.0

# For some reason, in older versions of the Piper firmware, in MIT mode, some of
# the joints behave in a "reverse" manner, whether for direct torque commands or
# for setting a desired position reference. This mapping tells us which joints
# are 'flipped'.
_MIT_FLIP_FIX_VERSION = packaging_version.Version("1.7-3")
_PRE_V1_7_3_MIT_JOINT_FLIP = [True, True, False, True, False, True]
_POST_V1_7_3_MIT_JOINT_FLIP = [False, False, False, False, False, False]

_MIT_TORQUE_LIMITS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

# Allowed gains range allowed for the Mit controller.
_MAX_KP_GAIN = 100.0
_MIN_KP_GAIN = 0.0

_MAX_KD_GAIN = 10.0
_MIN_KD_GAIN = 0.0


def _joints_within_target_threshold(
    cur_joints: Sequence[float],
    target: Sequence[float],
    threshold: Sequence[float] | float = 0.001,
):
  assert len(cur_joints) == len(target)
  diffs = np.abs(np.array(cur_joints) - np.array(target))
  return np.all(diffs < np.array(threshold))


class JointPositionController(abc.ABC):
  """Joint position controller base class."""

  def __init__(self, piper: pi.PiperInterface):
    self._piper = piper

  @property
  def piper(self) -> pi.PiperInterface:
    return self._piper

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exit_type, value, traceback):
    del exit_type, value, traceback
    self.stop()

  def move_to_position(
      self,
      target: Sequence[float],
      threshold: Sequence[float] | float = 0.001,
      timeout: float = 1.0,
  ) -> bool:
    """Moves the arm to a particular target pose. This is a blocking call.

    Args:
      target: list of joint angles (in radians) to move the arm to.
      threshold: a threshold error for when we consider the target reached.
      timeout: timeout in seconds, after which we finish blocking.

    Returns whether the target joint pose was reached (error within threshold).
    """

    assert len(target) == 6

    start_time = time.time()
    while time.time() - start_time < timeout:
      self.command_joints(target)
      cur_joints = self.piper.get_joint_positions()

      if _joints_within_target_threshold(cur_joints, target, threshold):
        return True

      time.sleep(1.0 / _CONTROL_RATE)

    return False

  @abc.abstractmethod
  def start(self) -> None:
    pass

  @abc.abstractmethod
  def stop(self) -> None:
    pass

  @abc.abstractmethod
  def command_joints(self, target: Sequence[float]) -> None:
    pass


class BuiltinJointPositionController(JointPositionController):
  """Joint position controller that uses the inbuilt position commands."""

  def __init__(
      self,
      piper: pi.PiperInterface,
      rest_position: (
          Sequence[float] | None
      ) = ArmOrientations.upright.rest_position,
  ):
    """Controller constructor

    Args:
      piper: The piper robot interface.
      rest_position: An optional joint angles (6) in radians that the robot will
        go to upon stopping. If None is set, then the rest behaviour is not
        executed.
    """

    super().__init__(piper)
    self._rest_position = rest_position

  def start(self) -> None:
    self.piper.set_arm_mode(
        arm_controller=pi.ArmController.POSITION_VELOCITY,
        move_mode=pi.MoveMode.JOINT,
    )

  def stop(self) -> None:
    if self._rest_position:
      self.move_to_position(self._rest_position, timeout=3.0)

  def command_joints(self, target: Sequence[float]) -> None:
    self.piper.command_joint_positions(target)


class MitJointPositionController(JointPositionController):
  """Joint position controller that uses MIT-mode position commands.

  By using MIT mode we can specify the P and D gains for the underlying
  controller.
  """

  def __init__(
      self,
      piper: pi.PiperInterface,
      kp_gains: Sequence[float] | float,
      kd_gains: Sequence[float] | float,
      rest_position: (
          Sequence[float] | None
      ) = ArmOrientations.upright.rest_position,
  ):
    """Controller constructor

    Args:
      piper: The piper robot interface.
      kp_gains: Either individual joint p-gains (6), or a single shared p-gain.
      kd_gains: Either individual joint d-gains (6), or a single shared d-gain.
      rest_position: An optional joint angles (6) in radians that the robot will
        go to upon stopping. If None is set, then the rest behaviour is not
        executed.
    """

    super().__init__(piper)

    if isinstance(kp_gains, float):
      self._kp_gains = (kp_gains,) * 6
    else:
      self._kp_gains = tuple(kp_gains)  # type: ignore
      assert len(self._kp_gains) == 6

    if isinstance(kd_gains, float):
      self._kd_gains = (kd_gains,) * 6
    else:
      self._kd_gains = tuple(kd_gains)  # type: ignore
      assert len(self._kd_gains) == 6

    if any([p < _MIN_KP_GAIN or p > _MAX_KP_GAIN for p in self._kp_gains]):
      raise ValueError(f"KP gains outside valid range: {self._kp_gains}")

    if any([d < _MIN_KD_GAIN or d > _MAX_KD_GAIN for d in self._kd_gains]):
      raise ValueError(f"KD gains outside valid range: {self._kd_gains}")

    self._rest_position = rest_position
    current_firmware = self.piper.get_piper_firmware_version()
    try:
      firmware_version = packaging_version.parse(current_firmware)
      if firmware_version < _MIT_FLIP_FIX_VERSION:
        self._joint_flip_map = _PRE_V1_7_3_MIT_JOINT_FLIP
      else:
        self._joint_flip_map = _POST_V1_7_3_MIT_JOINT_FLIP
    except packaging_version.InvalidVersion as e:
      raise ValueError(
          f"Invalid firmware version string: {current_firmware}"
      ) from e

  def start(self) -> None:
    self.piper.set_arm_mode(
        arm_controller=pi.ArmController.MIT,
        move_mode=pi.MoveMode.MIT,
    )

  def stop(self) -> None:
    # Move to the rest position if one is specified.
    if self._rest_position:
      self._smoothly_move_to_position(
          self._rest_position,
          threshold=0.1,  # No need to be precise.
          timeout=2.0,
      )

    # Over a few seconds relax all of the joints.
    self.relax_joints(2.0)

  def command_joints(
      self,
      target: Sequence[float],
      kp_gains: Sequence[float] | None = None,
      kd_gains: Sequence[float] | None = None,
  ) -> None:
    if not kp_gains:
      kp_gains = self._kp_gains

    if not kd_gains:
      kd_gains = self._kd_gains

    assert len(target) == 6
    assert len(kp_gains) == 6
    assert len(kd_gains) == 6

    cur_joints = self._piper.get_joint_positions()
    for ji, pos in enumerate(target):
      joint_error = np.abs(cur_joints[ji] - pos)

      # Scale the p-gain as a function of the error. When far away, have a lower
      # gain than when closer to the target. This has the beneficit of the arm
      # being less jerky on initial movement when the target is far away.
      kp = np.clip(1.5 / (joint_error + 0.0001), 0.75, kp_gains[ji])
      kd = kd_gains[ji]

      # Clip the position to limits so that we don't send invalid commands.
      min_rad = pi.JOINT_LIMITS_RAD["min"][ji]
      max_rad = pi.JOINT_LIMITS_RAD["max"][ji]
      pos = min(max(pos, min_rad), max_rad)

      if self._joint_flip_map:
        pos = -pos if self._joint_flip_map[ji] else pos

      self._piper.command_joint_position_mit(ji, pos, kp, kd)

  def relax_joints(self, timeout: float) -> None:
    """Relaxes joints, using MIT mode, over a number of seconds.

    This can be useful to "rest" the arm just prior to shutting down.
    """

    num_steps = int(round(timeout * _CONTROL_RATE))
    kp_gains = np.geomspace(2.0, 0.01, num_steps)
    kd_gains = np.geomspace(1.0, 0.01, num_steps)

    # Maintain current position with ever decreasing gains to "relax" the arm.
    for i in range(num_steps):
      self.command_joints(
          self._piper.get_joint_positions(),
          kp_gains=[kp_gains[i]] * 6,
          kd_gains=[kd_gains[i]] * 6,
      )
      time.sleep(1.0 / _CONTROL_RATE)

  def command_torques(
      self,
      torques: Sequence[float],
  ) -> None:
    assert len(torques) == 6

    for ji, torque in enumerate(torques):
      if self._joint_flip_map:
        torque = -torque if self._joint_flip_map[ji] else torque
      torque = np.clip(torque, -_MIT_TORQUE_LIMITS[ji], _MIT_TORQUE_LIMITS[ji])

      self._piper.command_joint_torque_mit(ji, torque)

  def _smoothly_move_to_position(
      self,
      target: Sequence[float],
      threshold: Sequence[float] | float = 0.001,
      timeout: float = 1.0,
  ) -> bool:
    assert len(target) == 6

    ramp_steps = 400  # 2 seconds
    p_gains = np.geomspace(0.5, 5.0, ramp_steps)

    step_idx = 0
    start_time = time.time()
    while time.time() - start_time < timeout:
      if step_idx > len(p_gains):
        p_gain = p_gains[-1]
      else:
        p_gain = p_gains[step_idx]

      self.command_joints(target, kp_gains=[p_gain] * 6)
      cur_joints = self.piper.get_joint_positions()

      if _joints_within_target_threshold(cur_joints, target, threshold):
        return True

      time.sleep(1.0 / _CONTROL_RATE)
      step_idx += 1

    return False


class GripperController(abc.ABC):
  """Gripper controller."""

  def __init__(self, piper: pi.PiperInterface):
    self._piper = piper

  @property
  def piper(self) -> pi.PiperInterface:
    return self._piper

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exit_type, value, traceback):
    del exit_type, value, traceback
    self.stop()

  def command_open(self) -> None:
    self.piper.command_gripper(position=pi.GRIPPER_ANGLE_MAX)

  def command_close(self) -> None:
    self.piper.command_gripper(position=0.0)

  def command_position(
      self,
      target: float,
      effort: float = DEFAULT_GRIPPER_EFFORT,
  ) -> None:
    target = np.clip(target, 0.0, pi.GRIPPER_ANGLE_MAX)
    self.piper.command_gripper(position=target, effort=effort)

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass
