"""Gravity compensation model using MuJoCo simulation + learned residuals."""

import argparse
import enum
import pathlib
import signal
import threading
import time
from collections.abc import Sequence

import mujoco as mj
import numpy as np
from scipy import optimize

from piper_control import piper_control, piper_init, piper_interface


class ModelType(enum.Enum):
  """Gravity compensation model types."""

  LINEAR = "linear"
  AFFINE = "affine"
  QUADRATIC = "quadratic"
  CUBIC = "cubic"
  FEATURES = "features"
  DIRECT = "direct"


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
      samples_path: str | pathlib.Path,
      model_path: str | pathlib.Path,
      model_type: ModelType = ModelType.QUADRATIC,
      joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
  ):
    self.model = mj.MjModel.from_xml_path(str(model_path))
    self.data = mj.MjData(self.model)
    self.model_type = model_type
    self.joint_names = tuple(joint_names)

    joint_indices = [self.model.joint(name).id for name in self.joint_names]
    self.qpos_indices = self.model.jnt_qposadr[joint_indices]
    self.qvel_indices = self.model.jnt_dofadr[joint_indices]

    self._fit_model(samples_path)

  def _fit_model(self, samples_path: str | pathlib.Path) -> None:
    print(f"Loading samples from {samples_path}")
    npz_data = np.load(samples_path)
    qpos = npz_data["qpos"]
    tau = npz_data["efforts"]

    print("Calculating MuJoCo torques...")
    mj_tau = np.array([self._calculate_sim_tau(q) for q in qpos])

    print(
        f"Fitting gravity compensation model using {self.model_type.value}..."
    )
    self.gravity_models: dict = {}

    if self.model_type == ModelType.LINEAR:
      self._fit_polynomial_model(_linear_gravity_tau, mj_tau, tau)
    elif self.model_type == ModelType.AFFINE:
      self._fit_polynomial_model(_affine_gravity_tau, mj_tau, tau)
    elif self.model_type == ModelType.QUADRATIC:
      self._fit_polynomial_model(_quadratic_gravity_tau, mj_tau, tau)
    elif self.model_type == ModelType.CUBIC:
      self._fit_polynomial_model(_cubic_gravity_tau, mj_tau, tau)
    elif self.model_type == ModelType.FEATURES:
      self._fit_feature_model(mj_tau, tau, qpos)
    elif self.model_type == ModelType.DIRECT:
      self._setup_direct_model()
    else:
      raise ValueError(f"Unknown model type: {self.model_type}")

  def _fit_polynomial_model(self, model_fn, mj_tau, tau) -> None:
    n_params = {
        ModelType.LINEAR: 1,
        ModelType.AFFINE: 2,
        ModelType.QUADRATIC: 3,
        ModelType.CUBIC: 4,
    }[self.model_type]

    bounds = ([-100.0] * n_params, [100.0] * n_params)

    for joint_idx, joint_name in enumerate(self.joint_names):
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

      print(f"{joint_name}: {self.model_type.value}, params: {opt_params}")
      print(f"  convergence: {mesg} (ier={ier})")
      print(f"  residuals (sum): {np.abs(infodict['fvec']).sum():.6f}")

      self.gravity_models[joint_name] = lambda x, params=opt_params: model_fn(
          x, *params
      )

  def _fit_feature_model(self, mj_tau, tau, qpos) -> None:
    n_joints = len(self.joint_names)
    feature_fn = _make_feature_gravity_tau(n_joints)

    for joint_idx, joint_name in enumerate(self.joint_names):
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

      print(f"{joint_name}: feature model, {len(opt_params)} params")
      print(f"  convergence: {mesg} (ier={ier})")
      print(f"  residuals (sum): {np.abs(infodict['fvec']).sum():.6f}")

      self.gravity_models[joint_name] = (
          lambda data, params=opt_params, fn=feature_fn: fn(data, *params)
      )

  def _setup_direct_model(self) -> None:
    for joint_idx, joint_name in enumerate(self.joint_names):
      scale = (
          DIRECT_SCALING_FACTORS[joint_idx]
          if joint_idx < len(DIRECT_SCALING_FACTORS)
          else 1.0
      )
      self.gravity_models[joint_name] = lambda x, s=scale: x * s
      print(f"{joint_name}: direct model with scale={scale}")

  def _calculate_sim_tau(self, qpos):
    self.data.qpos[self.qpos_indices] = qpos
    mj.mj_forward(self.model, self.data)
    return self.data.qfrc_bias[self.qvel_indices]

  def predict(self, qpos) -> np.ndarray:
    mj_tau = self._calculate_sim_tau(qpos)

    if self.model_type in (
        ModelType.LINEAR,
        ModelType.AFFINE,
        ModelType.QUADRATIC,
        ModelType.CUBIC,
        ModelType.DIRECT,
    ):
      return np.asarray(
          [
              self.gravity_models[name](mj_tau[i])
              for i, name in enumerate(self.joint_names)
          ]
      )
    elif self.model_type == ModelType.FEATURES:
      all_data = np.concatenate([mj_tau, qpos])
      return np.asarray(
          [self.gravity_models[name](all_data) for name in self.joint_names]
      )
    else:
      raise ValueError(f"Unknown model type: {self.model_type}")


def get_default_model_path() -> pathlib.Path:
  """Return path to the bundled MuJoCo model."""
  return pathlib.Path(__file__).parent / "models" / "piper_grav_comp.xml"


def main():
  """CLI entry point for running gravity compensation."""
  parser = argparse.ArgumentParser(
      description="Run gravity compensation on the Piper arm"
  )
  parser.add_argument("samples_path", help="Path to .npz samples file")
  parser.add_argument(
      "--model-path",
      default=str(get_default_model_path()),
      help="Path to MuJoCo XML model",
  )
  parser.add_argument(
      "--joint-names",
      nargs="+",
      default=list(DEFAULT_JOINT_NAMES),
      help="Joint names in the model",
  )
  parser.add_argument("--can-port", default="can0")
  parser.add_argument(
      "--model-type", default="cubic", choices=[t.value for t in ModelType]
  )
  args = parser.parse_args()

  model_type = ModelType(args.model_type)

  print("Loading gravity compensation model...")
  grav_model = GravityCompensationModel(
      samples_path=args.samples_path,
      model_path=args.model_path,
      model_type=model_type,
      joint_names=args.joint_names,
  )

  print("Connecting to Piper robot...")
  piper = piper_interface.PiperInterface(args.can_port)
  piper.show_status()

  piper.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)
  piper_init.reset_arm(
      piper,
      arm_controller=piper_interface.ArmController.MIT,
      move_mode=piper_interface.MoveMode.MIT,
  )

  controller = piper_control.MitJointPositionController(
      piper,
      kp_gains=[5.0, 5.0, 5.0, 5.6, 20.0, 6.0],
      kd_gains=0.8,
      rest_position=piper_control.ArmOrientations.upright.rest_position,
  )

  shutdown_event = threading.Event()

  def signal_handler(signum, frame):
    del signum, frame
    print("\nShutdown signal received...")
    shutdown_event.set()

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  print("Starting gravity compensation mode...")
  input("Press Enter to start...")

  try:
    while not shutdown_event.is_set():
      qpos = piper.get_joint_positions()
      qvel = np.array(piper.get_joint_velocities())

      hover_torque = grav_model.predict(qpos)
      stability_torque = -qvel * 1.0
      applied_torque = hover_torque + stability_torque

      controller.command_torques(applied_torque)
      time.sleep(0.005)
  finally:
    print("Cleaning up...")
    controller.stop()
    piper_init.disable_arm(piper)
    print("Done.")


if __name__ == "__main__":
  main()
