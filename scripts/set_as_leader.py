"""Sets the robot arm as leader in the leader/follower or master/slave setup.

Leader/follower setups have 2 separate arms wired to the same CAN bus, and you
can set one as the leader (a.k.a. master) and the other as the follower (a.k.a.
slave).

While we tested these setups, we noticed that afterwards the follower arm would
not be functional anymore in a standalone setup. This script allows you to
revert the follower arm back to a working config, by setting it as a
leader/master.
"""

import argparse

from piper_control import piper_control


def main():
  parser = argparse.ArgumentParser(description="Set the robot arm as leader.")
  parser.add_argument(
      "--can_port",
      type=str,
      default="can0",
      help="The CAN port to use (default: can0).",
  )
  args = parser.parse_args()
  robot = piper_control.PiperControl(args.can_port)

  linkage_config = 0xFA  # Teaching arm
  feedback_offset = 0x00  # Default feedback (2Ax)
  ctrl_offset = 0x00  # Default control (15x)
  linkage_offset = 0x00  # Not used
  robot.piper.EnableArm()
  robot.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # type: ignore
  robot.piper.MasterSlaveConfig(
      linkage_config, feedback_offset, ctrl_offset, linkage_offset
  )

  # The following may have an effect on setting it to a "regular" mode, but
  # this has not yet been verified by our team.
  # robot.piper.MasterSlaveConfig(0x00, 0x00, 0x00, 0x00)
