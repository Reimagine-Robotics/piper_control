"""CLI script for setting collision protection level on the Piper arm."""

import argparse

from piper_control import piper_connect, piper_interface

DEFAULT_LEVEL = 1
MAX_RETRIES = 25


def main():
  """CLI entry point for setting collision protection level."""
  parser = argparse.ArgumentParser(
      description="Set collision protection level for the Piper arm"
  )
  parser.add_argument(
      "level",
      type=int,
      nargs="?",
      default=DEFAULT_LEVEL,
      help=f"Protection level to set (default: {DEFAULT_LEVEL})",
  )
  parser.add_argument("--can-port", default="can0", help="CAN port to use")
  args = parser.parse_args()

  piper_connect.activate()
  robot = piper_interface.PiperInterface(can_port=args.can_port)

  expected_levels = [args.level] * 6
  initial_levels = [DEFAULT_LEVEL] * 6

  print(f"Setting collision protection level to {args.level} for all joints...")

  # First reset to default level (arm reports 0 when not initialized)
  for _ in range(MAX_RETRIES):
    robot.set_collision_protection(initial_levels)
    if robot.get_collision_protection() == initial_levels:
      break
  else:
    current = robot.get_collision_protection()
    print(
        f"Failed to reset protection level. Expected {initial_levels}, "
        f"got {current}"
    )
    return 1

  # Set the desired level
  for _ in range(MAX_RETRIES):
    robot.set_collision_protection(expected_levels)
    if robot.get_collision_protection() == expected_levels:
      print(f"Successfully set collision protection level to {args.level}.")
      return 0

  current = robot.get_collision_protection()
  print(
      f"Failed to set protection level. Expected {expected_levels}, "
      f"got {current}"
  )
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
