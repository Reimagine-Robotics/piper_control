"""Gravity compensation model using MuJoCo simulation + learned residuals."""

# pylint: disable=logging-fstring-interpolation,inconsistent-quotes

import enum
import logging as log
import pathlib
from collections.abc import Sequence

import mujoco as mj
import numpy as np
from scipy import optimize


class ModelType(enum.Enum):
  """Gravity compensation model types."""

  LINEAR = "linear"
  AFFINE = "affine"
  QUADRATIC = "quadratic"
  CUBIC = "cubic"
  FEATURES = "features"
  DIRECT = "direct"


# These are the joint names in the default MuJoCo model for the piper arm.
DEFAULT_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

# Firmware scaling for old firmware (1.8-2 and earlier)
# J1-3: commanded torque is executed at 4x, so divide by 4
# J4-6: no scaling needed
DIRECT_SCALING_FACTORS = (0.25, 0.25, 0.25, 1.0, 1.0, 1.0)


def _linear_gravity_tau(tau, a):
  return a * tau


def _affine_gravity_tau(tau, a, b):
  return a * tau + b


def _quadratic_gravity_tau(tau, a, b, c):
  return a * tau * tau + b * tau + c


def _cubic_gravity_tau(tau, a, b, c, d):
  return a * tau * tau * tau + b * tau * tau + c * tau + d


def _build_features(sim_torques, joint_angles):
  features = [1.0]
  for sim_torque, joint_angle in zip(sim_torques, joint_angles):
    features.extend(
        [
            sim_torque,
            sim_torque**2,
            sim_torque**3,
            np.sin(joint_angle),
            np.cos(joint_angle),
        ]
    )
  return np.array(features)


def _make_feature_gravity_tau(n_joints):
  def _feature_gravity_tau(data, *params):
    if data.ndim == 1:
      data = data.reshape(1, -1)
    n_samples = data.shape[0]
    features_list = []
    for i in range(n_samples):
      sim_torques = data[i, :n_joints]
      joint_angles = data[i, n_joints:]
      features_list.append(_build_features(sim_torques, joint_angles))
    features_matrix = np.array(features_list)
    params_array = np.array(params)
    results = features_matrix @ params_array
    return results if results.shape[0] > 1 else results[0]

  return _feature_gravity_tau


class GravityCompensationModel:
  """Predicts gravity compensation torques using MuJoCo + learned residual."""

  def __init__(
      self,
      samples_path: str | pathlib.Path | None = None,
      model_path: str | pathlib.Path | None = None,
      model_type: ModelType = ModelType.DIRECT,
      joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
  ):
    model_path = model_path or get_default_model_path()
    self._model = mj.MjModel.from_xml_path(str(model_path))
    self._data = mj.MjData(self._model)
    self._model_type = model_type
    self._joint_names = tuple(joint_names)

    joint_indices = [self._model.joint(name).id for name in self._joint_names]
    self.qpos_indices = self._model.jnt_qposadr[joint_indices]
    self.qvel_indices = self._model.jnt_dofadr[joint_indices]

    if model_type is not ModelType.DIRECT:
      if not samples_path:
        raise ValueError(
            "samples_path must be provided for non-direct model types."
        )
      self._fit_model(samples_path)
    else:
      self._setup_direct_model()

  def _fit_model(self, samples_path: str | pathlib.Path) -> None:
    log.info(f"Loading samples from {samples_path}")
    npz_data = np.load(samples_path)
    if "qpos" not in npz_data or "efforts" not in npz_data:
      raise ValueError(
          f"Samples file must contain 'qpos' and 'efforts' arrays."
          f" Existing keys: {list(npz_data.keys())}"
      )
    qpos = npz_data["qpos"]
    tau = npz_data["efforts"]

    log.info(f"Calculating MuJoCo torques for {qpos.shape[0]} samples")
    mj_tau = np.array([self._calculate_sim_tau(q) for q in qpos])

    log.info(
        f"Fitting gravity compensation model using {self._model_type.value}..."
    )
    self.gravity_models: dict = {}

    if self._model_type == ModelType.LINEAR:
      self._fit_polynomial_model(_linear_gravity_tau, mj_tau, tau)
    elif self._model_type == ModelType.AFFINE:
      self._fit_polynomial_model(_affine_gravity_tau, mj_tau, tau)
    elif self._model_type == ModelType.QUADRATIC:
      self._fit_polynomial_model(_quadratic_gravity_tau, mj_tau, tau)
    elif self._model_type == ModelType.CUBIC:
      self._fit_polynomial_model(_cubic_gravity_tau, mj_tau, tau)
    elif self._model_type == ModelType.FEATURES:
      self._fit_feature_model(mj_tau, tau, qpos)
    else:
      raise ValueError(f"Unknown model type: {self._model_type}")

  def _fit_polynomial_model(self, model_fn, mj_tau, tau) -> None:
    n_params = {
        ModelType.LINEAR: 1,
        ModelType.AFFINE: 2,
        ModelType.QUADRATIC: 3,
        ModelType.CUBIC: 4,
    }[self._model_type]

    bounds = ([-100.0] * n_params, [100.0] * n_params)

    for joint_idx, joint_name in enumerate(self._joint_names):
      fit = optimize.curve_fit(
          model_fn,
          mj_tau[:, joint_idx],
          tau[:, joint_idx],
          bounds=bounds,
          full_output=True,
      )
      opt_params = fit[0]
      infodict = fit[2]
      mesg = fit[3]
      ier = fit[4]

      log.info(f"{joint_name}: {self._model_type.value}, params: {opt_params}")
      log.info(f"  convergence: {mesg} (ier={ier})")
      log.info(f"  residuals (sum): {np.abs(infodict['fvec']).sum():.6f}")

      self.gravity_models[joint_name] = lambda x, params=opt_params: model_fn(
          x, *params
      )

  def _fit_feature_model(self, mj_tau, tau, qpos) -> None:
    n_joints = len(self._joint_names)
    feature_fn = _make_feature_gravity_tau(n_joints)

    for joint_idx, joint_name in enumerate(self._joint_names):
      x_data = np.column_stack([mj_tau, qpos])
      n_features = 1 + n_joints * 5
      p0 = np.zeros(n_features)
      p0[1 + joint_idx * 5] = 1.0

      fit = optimize.curve_fit(
          feature_fn,
          x_data,
          tau[:, joint_idx],
          p0=p0,
          maxfev=5000,
          full_output=True,
      )
      opt_params = fit[0]
      infodict = fit[2]
      mesg = fit[3]
      ier = fit[4]

      log.info(f"{joint_name}: feature model, {len(opt_params)} params")
      log.info(f"  convergence: {mesg} (ier={ier})")
      log.info(f"  residuals (sum): {np.abs(infodict['fvec']).sum():.6f}")

      self.gravity_models[joint_name] = (
          lambda data, params=opt_params, fn=feature_fn: fn(data, *params)
      )

  def _setup_direct_model(self) -> None:
    for joint_idx, joint_name in enumerate(self._joint_names):
      scale = (
          DIRECT_SCALING_FACTORS[joint_idx]
          if joint_idx < len(DIRECT_SCALING_FACTORS)
          else 1.0
      )
      self.gravity_models[joint_name] = lambda x, s=scale: x * s
      log.info(f"{joint_name}: direct model with scale={scale}")

  def _calculate_sim_tau(self, qpos):
    self._data.qpos[self.qpos_indices] = qpos
    mj.mj_forward(self._model, self._data)
    return self._data.qfrc_bias[self.qvel_indices]

  def predict(self, qpos) -> np.ndarray:
    mj_tau = self._calculate_sim_tau(qpos)

    if self._model_type in (
        ModelType.LINEAR,
        ModelType.AFFINE,
        ModelType.QUADRATIC,
        ModelType.CUBIC,
        ModelType.DIRECT,
    ):
      return np.asarray(
          [
              self.gravity_models[name](mj_tau[i])
              for i, name in enumerate(self._joint_names)
          ]
      )
    elif self._model_type == ModelType.FEATURES:
      all_data = np.concatenate([mj_tau, qpos])
      return np.asarray(
          [self.gravity_models[name](all_data) for name in self._joint_names]
      )
    else:
      raise ValueError(f"Unknown model type: {self._model_type}")


def get_default_model_path() -> pathlib.Path:
  """Return path to the bundled MuJoCo model."""
  return pathlib.Path(__file__).parent / "models" / "piper_grav_comp.xml"
