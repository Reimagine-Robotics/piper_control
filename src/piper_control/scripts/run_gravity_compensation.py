"""CLI script for running gravity compensation on the Piper arm."""

import argparse
import logging as log
import signal
import threading
import time

import numpy as np

from piper_control import piper_control, piper_init, piper_interface
from piper_control.gravity_compensation import (
    DEFAULT_JOINT_NAMES,
    GravityCompensationModel,
    ModelType,
    get_default_model_path,
)


def main():
  """CLI entry point for running gravity compensation."""
  parser = argparse.ArgumentParser(
      description="Run gravity compensation on the Piper arm"
  )
  parser.add_argument("--samples-path", help="Path to .npz samples file")
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
  parser.add_argument(
      "--damping",
      type=float,
      default=1.0,
      help="Velocity damping gain for stability",
  )
  args = parser.parse_args()

  model_type = ModelType(args.model_type)

  log.info("Loading gravity compensation model...")
  grav_model = GravityCompensationModel(
      samples_path=args.samples_path,
      model_path=args.model_path,
      model_type=model_type,
      joint_names=args.joint_names,
  )

  log.info("Connecting to Piper robot...")
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
    log.info("\nShutdown signal received...")
    shutdown_event.set()

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  log.info("Starting gravity compensation mode...")
  input("Press Enter to start...")

  try:
    while not shutdown_event.is_set():
      qpos = piper.get_joint_positions()
      qvel = np.array(piper.get_joint_velocities())

      hover_torque = grav_model.predict(qpos)
      stability_torque = -qvel * args.damping
      applied_torque = hover_torque + stability_torque

      controller.command_torques(applied_torque)
      time.sleep(0.005)
  finally:
    log.info("Cleaning up...")
    controller.stop()
    piper_init.disable_arm(piper)
    log.info("Done.")


if __name__ == "__main__":
  main()
