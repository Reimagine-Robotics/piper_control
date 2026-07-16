"""Gravity compensation model using MuJoCo simulation."""

# pylint: disable=logging-fstring-interpolation,inconsistent-quotes

import logging as log
import pathlib
from collections.abc import Sequence

import mujoco as mj
import numpy as np
from packaging import version as packaging_version

from piper_control import piper_interface

# These are the joint names in the default MuJoCo model for the piper arm.
DEFAULT_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)


def direct_scaling_factors(
    firmware_version: str | None,
    arm_type: piper_interface.PiperArmType = piper_interface.PiperArmType.PIPER,
) -> tuple[float, ...]:
  """Return per-joint command scaling factors for firmware and arm model.

  Firmware <= S-V1.8-2 multiplies MIT commands by the per-joint gear ratio b
  internally (4x on J1-3 for base piper), so commands are divided by b before
  sending. b is per-model (piper / piper_h / piper_l / piper_x).

  Firmware >= S-V1.8-3 normalizes to base-piper gearing, so only the gear ratio
  relative to base piper needs correcting (identity for base piper, b_base/b for
  other models). This matches known-good behaviour: base piper uses all 1s and a
  piper_h needs 1/1.7 on J5 (b_base=1, b_h=1.7).

  When firmware_version is None (unknown), the old-firmware scaling is applied
  as the safe default to avoid sending stronger torques.
  """
  # b = per-joint gear ratios (see piper_interface.joint_torque_coefficients).
  _, b = piper_interface.joint_torque_coefficients(arm_type)
  _, b_base = piper_interface.joint_torque_coefficients(
      piper_interface.PiperArmType.PIPER
  )
  parsed = (
      packaging_version.parse(firmware_version) if firmware_version else None
  )
  if parsed is not None and parsed > packaging_version.Version("1.8.post2"):
    return tuple(
        base_ratio / arm_ratio for base_ratio, arm_ratio in zip(b_base, b)
    )
  # firmware <= S-V1.8-2 amplifies commands by the gear ratio b; divide it out.
  # https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/Q%26A.MD#32-sdk-to-obtain-motor-feedback-torque
  return tuple(1.0 / arm_ratio for arm_ratio in b)


class GravityCompensationModel:
  """Predicts gravity compensation torques using MuJoCo + learned residual."""

  def __init__(
      self,
      model_path: str | pathlib.Path | None = None,
      joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
      firmware_version: str | None = None,
      arm_type: piper_interface.PiperArmType = piper_interface.PiperArmType.PIPER,
  ):
    model_path = model_path or get_default_model_path()
    self._model = mj.MjModel.from_xml_path(str(model_path))
    self._data = mj.MjData(self._model)
    self._joint_names = tuple(joint_names)
    self._firmware_version = firmware_version
    self._arm_type = arm_type
    self.gravity_models: dict = {}

    joint_indices = [self._model.joint(name).id for name in self._joint_names]
    self.qpos_indices = self._model.jnt_qposadr[joint_indices]
    self.qvel_indices = self._model.jnt_dofadr[joint_indices]

    self._setup_direct_model()

  def _setup_direct_model(self) -> None:
    scaling = direct_scaling_factors(self._firmware_version, self._arm_type)
    for joint_idx, joint_name in enumerate(self._joint_names):
      scale = scaling[joint_idx]
      self.gravity_models[joint_name] = lambda x, s=scale: x * s
      log.info(f"{joint_name}: direct model with scale={scale}")

  def _calculate_sim_tau(self, qpos):
    self._data.qpos[self.qpos_indices] = qpos
    mj.mj_forward(self._model, self._data)
    return self._data.qfrc_bias[self.qvel_indices]

  def predict(self, qpos) -> np.ndarray:
    mj_tau = self._calculate_sim_tau(qpos)
    return np.asarray(
        [
            self.gravity_models[name](mj_tau[i])
            for i, name in enumerate(self._joint_names)
        ]
    )


def get_default_model_path() -> pathlib.Path:
  """Return path to the bundled MuJoCo model."""
  return pathlib.Path(__file__).parent / "models" / "piper_grav_comp.xml"
