"""Record and replay joint trajectories for Piper arm(s).

Single interactive mode with keyboard controls for recording and replaying.
Prompts for filenames when saving/loading. Supports 1 or 2 arms.

Usage:
  python record_trajectories.py --robots can0 --samples-path samples.npz
  python record_trajectories.py \\
    --robots can0 can1 --samples-path samples.npz --model-path path/to/model.xml
"""

import argparse
import json
import logging as log
import pathlib
import select
import sys
import termios
import time
import tty

import numpy as np

from piper_control import piper_control, piper_init, piper_interface
from piper_control.gravity_compensation import GravityCompensationModel, ModelType

# pylint: disable=logging-fstring-interpolation,inconsistent-quotes

RECORD_HZ = 100
REPLAY_HZ = 100
MOVE_DURATION = 1.0  # seconds
GRIPPER_OPEN = piper_interface.GRIPPER_ANGLE_MAX
GRIPPER_CLOSED = 0.0
DEFAULT_KP_GAINS = np.array([5.0, 5.0, 5.0, 5.6, 20.0, 6.0])
DEFAULT_KD_GAINS = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

# Gravity compensation defaults
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "src" / "piper_control" / "models" / "piper_grav_comp.xml"
)
MODEL_TYPE_CHOICES = tuple(mt.value for mt in ModelType)


class RawTerminal:
  """Context manager for raw terminal input."""

  def __init__(self):
    self.fd = sys.stdin.fileno()
    self.old_settings = None

  def __enter__(self):
    self.old_settings = termios.tcgetattr(self.fd)
    tty.setcbreak(self.fd)
    return self

  def __exit__(self, *args):
    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

  def get_key(self) -> str | None:
    """Read a keypress if available, otherwise return None."""
    if select.select([sys.stdin], [], [], 0)[0]:
      return sys.stdin.read(1)
    return None


def prompt_filename(prompt: str, default: str = "") -> str:
  """Prompt user for a filename (restores terminal settings temporarily)."""
  # Restore terminal to normal mode for input
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

  try:
    if default:
      result = input(f"{prompt} [{default}]: ").strip()
      return result if result else default
    else:
      return input(f"{prompt}: ").strip()
  finally:
    # Return to raw mode
    tty.setcbreak(fd)


def move_to_position(
    controller: piper_control.MitJointPositionController,
    current_q: np.ndarray,
    target_q: np.ndarray,
    gravity_model: GravityCompensationModel | None = None,
    duration: float = MOVE_DURATION,
) -> None:
  """Smoothly interpolate from current to target position."""
  n_steps = int(duration * REPLAY_HZ)
  for alpha in np.linspace(0, 1, n_steps):
    q = (1 - alpha) * current_q + alpha * target_q
    if gravity_model:
      gravity_torque = gravity_model.predict(q)
      controller.command_joints(q.tolist(), torques_ff=gravity_torque.tolist())
    else:
      controller.command_joints(q.tolist())
    time.sleep(1.0 / REPLAY_HZ)


def main():
  parser = argparse.ArgumentParser(
      description="Record/Replay Piper trajectories"
  )
  parser.add_argument(
      "--robots",
      nargs="+",
      default=["can0"],
      help="Robot name(s), e.g. can0 or can0 can1",
  )
  parser.add_argument(
      "--gravity",
      action="store_true",
      help="Enable gravity compensation during replay",
  )
  parser.add_argument(
      "--samples-path",
      type=pathlib.Path,
      required=True,
      help="Path to the gravity compensation samples (.npz).",
  )
  parser.add_argument(
      "--model-path",
      type=pathlib.Path,
      default=DEFAULT_MODEL_PATH,
      help=(
          "Path to the MuJoCo model XML. Defaults to the bundled "
          "piper_grav_comp.xml in src/piper_control/models."
      ),
  )
  parser.add_argument(
      "--model-type",
      type=str,
      choices=MODEL_TYPE_CHOICES,
      default=ModelType.CUBIC.value,
      help="Gravity model type. Defaults to 'cubic'.",
  )
  parser.add_argument(
      "--kp-gains",
      type=float,
      nargs=6,
      metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
      default=DEFAULT_KP_GAINS.tolist(),
      help="Joint KP gains (6 values). Defaults to tuned values for Piper.",
  )
  parser.add_argument(
      "--kd-gains",
      type=float,
      nargs=6,
      metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
      default=DEFAULT_KD_GAINS.tolist(),
      help="Joint KD gains (6 values). Defaults to tuned values for Piper.",
  )
  args = parser.parse_args()

  assert len(args.robots) <= 2, "Maximum 2 robots supported"

  # Connect to robots
  robots = {}
  for robot_name in args.robots:
    log.info(f"Connecting to {robot_name}...")
    robot = piper_interface.PiperInterface(robot_name)
    robot.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)
    piper_init.reset_arm(
        robot,
        arm_controller=piper_interface.ArmController.MIT,
        move_mode=piper_interface.MoveMode.MIT,
    )
    piper_init.reset_gripper(robot)
    robot.show_status()
    robots[robot_name] = robot

  # Load gravity compensation model
  gravity_model = None
  samples_path = pathlib.Path(args.samples_path).expanduser()
  model_path = pathlib.Path(args.model_path).expanduser()
  model_type = ModelType(args.model_type)
  if args.gravity:
    try:
      log.info("Loading gravity compensation model...")
      gravity_model = GravityCompensationModel(
          samples_path=samples_path,
          model_path=model_path,
          model_type=model_type,
      )
      log.info("Gravity compensation enabled.")
    except Exception as e:  # pylint: disable=broad-except
      log.warning(f"Could not load gravity model: {e}")
      log.warning("Continuing without gravity compensation.")

  trajectory = []
  gripper_positions = {name: GRIPPER_OPEN for name in robots}
  recording = False
  start_time = 0.0
  dt = 1.0 / RECORD_HZ

  kp_gains = np.array(args.kp_gains, dtype=float)
  kd_gains = np.array(args.kd_gains, dtype=float)
  robot_names = list(robots.keys())

  print(
      f"\nRecord/Replay Mode ({len(robots)} robot(s): {', '.join(robot_names)})"
  )
  print("  r: toggle recording (arms disabled during recording)")
  print("  p: replay trajectory")
  print("  o: open grippers")
  print("  c: close grippers")
  print("  s: save trajectory (prompts for filename)")
  print("  l: load trajectory (prompts for filename)")
  print("  q: quit\n")

  with RawTerminal() as term:
    controllers = {name: None for name in robots}
    arm_enabled = False

    while True:
      key = term.get_key()

      if key == "r":
        if not recording:
          # Start recording - disable arms
          for name, ctrl in controllers.items():
            if ctrl:
              ctrl.stop()
              controllers[name] = None
          for robot in robots.values():
            robot.disable_arm()
            robot.enable_gripper()
          arm_enabled = False
          recording = True
          start_time = time.time()
          trajectory = []
          print("Recording started (arms disabled)...")
        else:
          # Stop recording
          recording = False
          print(f"Recording stopped. {len(trajectory)} samples.")

      elif key == "p":
        if recording:
          print("Stop recording first!")
        elif not trajectory:
          print("No trajectory to replay!")
        else:
          print("Replaying...")
          # Re-enable arms if they were disabled
          if not arm_enabled:
            for robot in robots.values():
              piper_init.reset_arm(
                  robot,
                  arm_controller=piper_interface.ArmController.MIT,
                  move_mode=piper_interface.MoveMode.MIT,
              )
            arm_enabled = True

          # Create controllers if needed
          for name, robot in robots.items():
            if not controllers[name]:
              current_q = np.array(robot.get_joint_positions())
              controllers[name] = piper_control.MitJointPositionController(
                  robot,
                  kp_gains=kp_gains,
                  kd_gains=kd_gains,
                  rest_position=current_q.tolist(),
              )
              controllers[name].start()

          # Move to start positions
          print("Moving to start...")
          for name, robot in robots.items():
            current_q = np.array(robot.get_joint_positions())
            start_q = np.array(trajectory[0]["q"][name])
            move_to_position(
                controllers[name], current_q, start_q, gravity_model
            )

          # Replay
          for robot in robots.values():
            robot.enable_gripper()
          replay_start = time.time()
          for sample in trajectory:
            target_t = sample["t"]

            while time.time() - replay_start < target_t:
              time.sleep(0.001)

            for name, robot in robots.items():
              q = np.array(sample["q"][name])
              grip = sample["gripper"][name]
              if gravity_model:
                gravity_torque = gravity_model.predict(q)
                controllers[name].command_joints(
                    q.tolist(), torques_ff=gravity_torque.tolist()
                )
              else:
                controllers[name].command_joints(q.tolist())
              robot.command_gripper(grip, piper_interface.GRIPPER_EFFORT_MAX)

          # Flush any buffered keypresses from during replay
          termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
          print("Replay complete.")

      elif key == "o":
        for name, robot in robots.items():
          gripper_positions[name] = GRIPPER_OPEN
          robot.command_gripper(
              GRIPPER_OPEN, piper_interface.GRIPPER_EFFORT_MAX
          )
        print("Grippers: OPEN")

      elif key == "c":
        for name, robot in robots.items():
          gripper_positions[name] = GRIPPER_CLOSED
          robot.command_gripper(
              GRIPPER_CLOSED, piper_interface.GRIPPER_EFFORT_MAX
          )
        print("Grippers: CLOSED")

      elif key == "s":
        if recording:
          print("Stop recording first!")
        elif trajectory:
          filename = prompt_filename(
              "Enter filename to save", "trajectory.json"
          )
          if filename:
            with open(filename, "w") as f:
              json.dump(trajectory, f, indent=2)
            print(f"Saved {len(trajectory)} samples to {filename}")
          else:
            print("Save cancelled.")
        else:
          print("No trajectory to save!")

      elif key == "l":
        if recording:
          print("Stop recording first!")
        else:
          filename = prompt_filename(
              "Enter filename to load", "trajectory.json"
          )
          if filename:
            try:
              with open(filename) as f:
                loaded = json.load(f)
              # Validate robot names match
              if loaded:
                loaded_robots = set(loaded[0]["q"].keys())
                current_robots = set(robots.keys())
                if loaded_robots != current_robots:
                  print(
                      f"Robot mismatch! File has {loaded_robots}, current session has {current_robots}"
                  )
                else:
                  trajectory = loaded
                  print(f"Loaded {len(trajectory)} samples from {filename}")
              else:
                print("Empty trajectory file!")
            except FileNotFoundError:
              print(f"File not found: {filename}")
            except json.JSONDecodeError as e:
              print(f"Invalid JSON file: {e}")
          else:
            print("Load cancelled.")

      elif key == "q":
        print("\nExiting.")
        break

      # Record if active
      if recording:
        t = time.time() - start_time
        sample = {
            "t": t,
            "q": {
                name: list(robot.get_joint_positions())
                for name, robot in robots.items()
            },
            "gripper": dict(gripper_positions),
        }
        trajectory.append(sample)

      time.sleep(dt)

    # Cleanup
    for ctrl in controllers.values():
      if ctrl:
        ctrl.stop()


if __name__ == "__main__":
  main()
