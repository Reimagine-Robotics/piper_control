"""Gravity compensation model using MuJoCo simulation."""

# pylint: disable=logging-fstring-interpolation,inconsistent-quotes

import logging as log
import pathlib
from collections.abc import Sequence

import mujoco as mj
import numpy as np
from packaging import version as packaging_version

# These are the joint names in the default MuJoCo model for the piper arm.
DEFAULT_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)


def _direct_scaling_factors(
    firmware_version: packaging_version.Version | None,
) -> tuple[float, ...]:
  """Return per-joint command scaling factors for the given firmware.

  Firmware versions older than 1.8 amplify J1-3 commands by 4x internally,
  so we divide by 4 before sending. Newer firmware does not amplify.

  When firmware_version is None (unknown), the old-firmware scaling is applied
  as the safe default to avoid sending 4x stronger torques.
  """
  if (
      firmware_version is not None
      and firmware_version >= packaging_version.Version("1.8")
  ):
    return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
  return (0.25, 0.25, 0.25, 1.0, 1.0, 1.0)


class GravityCompensationModel:
  """Predicts gravity compensation torques using MuJoCo + learned residual."""

  def __init__(
      self,
      model_path: str | pathlib.Path | None = None,
      joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
      firmware_version: str | None = None,
  ):
    model_path = model_path or get_default_model_path()
    self._model = mj.MjModel.from_xml_path(str(model_path))
    self._data = mj.MjData(self._model)
    self._joint_names = tuple(joint_names)
    self._firmware_version = (
        packaging_version.parse(firmware_version) if firmware_version else None
    )
    self.gravity_models: dict = {}

    joint_indices = [self._model.joint(name).id for name in self._joint_names]
    self.qpos_indices = self._model.jnt_qposadr[joint_indices]
    self.qvel_indices = self._model.jnt_dofadr[joint_indices]

    self._setup_direct_model()

  def _setup_direct_model(self) -> None:
    scaling = _direct_scaling_factors(self._firmware_version)
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
