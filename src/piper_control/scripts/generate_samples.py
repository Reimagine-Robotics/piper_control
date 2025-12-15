"""Generate collision-free samples for gravity compensation calibration.

These samples are then used by a gravity-compensation model to predict the
feed-forward torques required to counteract the effect of gravity.

The model learns residuals from the efforts predicted by
the MuJoCo simulation model to the actual efforts measured by the arm.

NOTE: This will move the robot around the full range of motion of the arm.
Make sure the area around the robot is clear of obstacles and people.

To run with a custom model (recommended if you have attachments,
like custom fingers or cameras):

piper-generate-samples \
  --can-port can0 \
  --output samples.npz \
  --num-samples 25 \
  --model-path model.xml \
  --joint_names=piper_j{1..6}
"""

import argparse
import pathlib
import time

import mujoco as mj
import numpy as np

from piper_control import (
    collision_checking,
    piper_control,
    piper_init,
    piper_interface,
)
from piper_control.gravity_compensation import (
    DEFAULT_JOINT_NAMES,
    get_default_model_path,
)

CONTROL_FREQUENCY = 200.0
MOVE_DURATION = 2.5  # seconds to move between configurations
_KP_GAINS = np.array([5.0, 5.0, 5.0, 5.6, 20.0, 6.0])


class HaltonSampler:
  """Halton sequence sampler for joint configurations."""

  PRIMES = (2, 3, 5, 7, 11, 13)

  def __init__(self, limits_min, limits_max):
    self.center = 0.5 * (np.array(limits_max) + np.array(limits_min))
    self.radius = 0.5 * (np.array(limits_max) - np.array(limits_min))
    self.index = 0

  def sample(self):
    result = np.array(
        [
            self.center[i]
            + self.radius[i] * (2 * mj.mju_Halton(self.index, p) - 1)
            for i, p in enumerate(self.PRIMES)
        ]
    )
    self.index += 1
    return result


def main():
  """CLI entry point for generating gravity compensation samples."""
  parser = argparse.ArgumentParser(
      description="Generate collision-free samples for gravity compensation"
  )
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
  parser.add_argument("--num-samples", type=int, default=50)
  parser.add_argument("--can-port", default="can0")
  parser.add_argument(
      "-o", "--output", required=True, help="Output .npz file path"
  )
  args = parser.parse_args()

  # Make sure the user is ready for the robot to move.
  input(
      "WARNING: This script will move the robot around! "
      "Make sure the area around the robot is clear of obstacles and people. "
      "Press Enter to continue..."
  )

  joint_names = args.joint_names

  model = mj.MjModel.from_xml_path(args.model_path)
  data = mj.MjData(model)
  joint_indices = [model.joint(name).id for name in joint_names]
  qpos_indices = model.jnt_qposadr[joint_indices]

  joint_limits_min = [model.jnt_range[j][0] for j in joint_indices]
  joint_limits_max = [model.jnt_range[j][1] for j in joint_indices]

  print("Connecting to Piper robot...")
  robot = piper_interface.PiperInterface(args.can_port)
  robot.show_status()
  robot.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)

  controller = piper_control.MitJointPositionController(
      robot,
      kp_gains=_KP_GAINS,
      kd_gains=0.8,
      rest_position=piper_control.ArmOrientations.upright.rest_position,
  )
  piper_init.reset_arm(
      robot,
      arm_controller=piper_interface.ArmController.MIT,
      move_mode=piper_interface.MoveMode.MIT,
  )
  piper_init.reset_gripper(robot)
  print("Robot initialized.")

  dt = 1.0 / CONTROL_FREQUENCY
  num_steps = int(MOVE_DURATION * CONTROL_FREQUENCY)

  halton = HaltonSampler(joint_limits_min, joint_limits_max)

  samples_qpos = []
  samples_efforts = []
  samples_target_qpos = []

  print(f"Generating {args.num_samples} collision-free samples...")

  sample_count = 0
  while sample_count < args.num_samples:
    qpos_sample = halton.sample()
    data.qpos[qpos_indices] = qpos_sample

    if collision_checking.has_collision(model, data, verbose=True):
      continue

    sample_count += 1
    print(
        f"Sample {sample_count}/{args.num_samples}: Moving to configuration..."
    )

    # Linear interpolation from current position to target
    start_pos = np.array(robot.get_joint_positions())
    target_pos = np.array(qpos_sample)

    for step in range(num_steps):
      alpha = (step + 1) / num_steps
      interp_pos = start_pos + alpha * (target_pos - start_pos)
      controller.command_joints(interp_pos)

      # Record intermediate samples
      samples_qpos.append(robot.get_joint_positions())
      samples_efforts.append(robot.get_joint_efforts())
      samples_target_qpos.append(interp_pos)

      time.sleep(dt)

  output_path = pathlib.Path(args.output)
  np.savez(
      output_path,
      qpos=np.array(samples_qpos),
      efforts=np.array(samples_efforts),
      target_qpos=np.array(samples_target_qpos),
  )
  print(f"Saved {len(samples_qpos)} samples to {output_path}")

  controller.stop()
  piper_init.disable_arm(robot)


if __name__ == "__main__":
  main()
