"""Moves the 2nd-to-last joint of the robot arm a small amount."""

from piper_control import piper_connect
from piper_control import piper_init
from piper_control import piper_interface
from piper_control import piper_control


def main():
  print(
      "This script will move the 2nd-to-last joint of the robot arm a small "
      "amount, using position control."
  )
  input("Warning: the robot will move. Press Enter to continue...")

  ports = piper_connect.find_ports()
  print(ports)

  piper_connect.activate(ports)
  ports = piper_connect.active_ports()
  assert ports

  robot = piper_interface.PiperInterface(can_port=ports[0])
  robot.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)

  print("resetting arm")
  piper_init.reset_arm(
      robot,
      arm_controller=piper_interface.ArmController.MIT,
      move_mode=piper_interface.MoveMode.MIT,
  )
  piper_init.reset_gripper(robot)

  robot.show_status()

  with piper_control.MitPositionController(
      robot,
      kp_gains=5.0,
      kd_gains=0.8,
      rest_position=piper_control.REST_POSITION,
  ) as controller:
    print("moving to position ...")
    success = controller.move_to_position(
        [0.0, 1.4, -0.4, 0.0, 0.0, 0.0],
        threshold=0.01,
        timeout=5.0,
    )
    print(f"reached target: {success}")

  print("finished, disabling arm")
  piper_init.disable_arm(robot)


if __name__ == "__main__":
  main()
