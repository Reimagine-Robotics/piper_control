"""Generate collision-free samples for gravity compensation calibration."""

import argparse
import pathlib
import time

import mujoco as mj
import numpy as np
import ruckig

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

DISABLE_COLLISIONS = {
    ("link1", "link2"),
    ("right_finger", "left_finger"),
    ("link7", "link8"),
    ("world", "link1"),
}

ACCELERATION_LIMITS = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
JERK_LIMITS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
CONTROL_FREQUENCY = 200.0
FF_SCALING = np.array([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])


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
  parser.add_argument("--num-samples", type=int, default=250)
  parser.add_argument("--robot-name", default="can0")
  parser.add_argument(
      "-o", "--output", required=True, help="Output .npz file path"
  )
  args = parser.parse_args()

  joint_names = args.joint_names
  num_joints = len(joint_names)

  model = mj.MjModel.from_xml_path(args.model_path)
  data = mj.MjData(model)
  joint_indices = [model.joint(name).id for name in joint_names]
  qpos_indices = model.jnt_qposadr[joint_indices]

  joint_limits_min = [model.jnt_range[j][0] for j in joint_indices]
  joint_limits_max = [model.jnt_range[j][1] for j in joint_indices]
  vel_limits = [model.jnt_range[j][1] * 2 for j in joint_indices]

  print("Connecting to Piper robot...")
  robot = piper_interface.PiperInterface(args.robot_name)
  robot.show_status()
  robot.hard_reset()
  robot.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)

  kp_gains = np.array([5.0, 5.0, 5.0, 5.6, 20.0, 6.0])
  controller = piper_control.MitJointPositionController(
      robot,
      kp_gains=kp_gains * 5.0,
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
  ruckig_otg = ruckig.Ruckig(num_joints, dt)
  ruckig_input = ruckig.InputParameter(num_joints)
  ruckig_output = ruckig.OutputParameter(num_joints)

  ruckig_input.min_position = joint_limits_min
  ruckig_input.max_position = joint_limits_max
  ruckig_input.min_velocity = [-v for v in vel_limits]
  ruckig_input.max_velocity = vel_limits
  ruckig_input.max_acceleration = ACCELERATION_LIMITS[:num_joints]
  ruckig_input.max_jerk = JERK_LIMITS[:num_joints]

  ruckig_input.current_position = robot.get_joint_positions()
  ruckig_input.current_velocity = [0.0] * num_joints
  ruckig_input.current_acceleration = [0.0] * num_joints
  ruckig_input.target_position = list(ruckig_input.current_position)
  ruckig_input.target_velocity = [0.0] * num_joints
  ruckig_input.target_acceleration = [0.0] * num_joints

  halton = HaltonSampler(joint_limits_min, joint_limits_max)

  samples_qpos = []
  samples_efforts = []

  print(f"Generating {args.num_samples} collision-free samples...")

  sample_count = 0
  while sample_count < args.num_samples:
    qpos_sample = halton.sample()
    data.qpos[qpos_indices] = qpos_sample

    if collision_checking.has_collision(
        model, data, DISABLE_COLLISIONS, verbose=True
    ):
      continue

    sample_count += 1
    print(
        f"Sample {sample_count}/{args.num_samples}: Moving to configuration..."
    )

    ruckig_input.target_position = list(qpos_sample)
    ruckig_input.target_velocity = [0.0] * num_joints
    ruckig_input.target_acceleration = [0.0] * num_joints
    ruckig_input.current_position = robot.get_joint_positions()
    ruckig_input.current_velocity = robot.get_joint_velocities()
    ruckig_input.current_acceleration = [0.0] * num_joints

    result = ruckig.Result.Working
    while result == ruckig.Result.Working:
      result = ruckig_otg.update(ruckig_input, ruckig_output)
      ruckig_output.pass_to_input(ruckig_input)

      if result in (ruckig.Result.Working, ruckig.Result.Finished):
        target_pos = ruckig_output.new_position
        data.qpos[qpos_indices] = target_pos
        mj.mj_forward(model, data)
        controller.command_joints(target_pos)
      else:
        print(f"Ruckig failed: {result}")
        break

      time.sleep(dt)

    time.sleep(0.5)

    samples_qpos.append(robot.get_joint_positions())
    samples_efforts.append(robot.get_joint_efforts())

  output_path = pathlib.Path(args.output)
  np.savez(
      output_path,
      qpos=np.array(samples_qpos),
      efforts=np.array(samples_efforts),
  )
  print(f"Saved {len(samples_qpos)} samples to {output_path}")

  controller.stop()
  piper_init.disable_arm(robot)


if __name__ == "__main__":
  main()
