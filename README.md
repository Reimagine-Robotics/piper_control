# piper_control - Library for controlling AgileX Pipers

## Overview

This repo provides low-level python modules for connecting and controlling
AgileX Piper robots.

*   `piper_connect`: a python implementation of the CAN setup scripts bundled
    with `piper_sdk`. This are not accessible on pip installs, and we found it
    useful to be able to query and activate CAN ports programmatically.

*   `piper_control`:  our lightweight wrapper of `piper_sdk` for controlling
    AgileX Piper robots.

  The `piper_sdk` API is powerful and quickly maturing, but it's a bit complex
  and under-documented, and we found it helpful to define a simple abstraction
  for basic I/O.

  There are also several sharp bits in `piper_sdk` which can make the robots
  seem tempermental, e.g. becoming unresponsive despite repeated calls to
  `MotionCtrl_2`, `EnableArm`, `GripperCtrl`, etc. We've bundled our solutions
  into `PiperControl` so `reset` and the various move commands perform as one
  would expect.

## Quick start

Install the dependencies and package:

```shell
sudo apt install can-utils
pip install piper_control
```

Set up the CAN connection to the arm(s):

```python
# Set up the connection to the Piper arm.
# These steps require sudo access.
from piper_control import piper_connect

# Print out the CAN ports that are available to connect.
print(piper_connect.find_ports())

# Activate all the ports so that you can connect to any arms connected to your
# machine.
piper_connect.activate()

# Check to see that all the ports are active.
print(piper_connect.active_ports())
```

Then control the robot:

```python
from piper_control import piper_interface
from piper_control import piper_init

robot = piper_interface.PiperInterface(can_port="can0")

# Resets the robot and enables the motors and motion controller for the arm.
# This call is necessary to be able to both query state and send commands to the
# robot.
piper_init.reset_arm(
    robot,
    arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
    move_mode=piper_interface.MoveMode.JOINT,
)
piper_init.reset_gripper(robot)

# See where the robot is now.
joint_angles = robot.get_joint_positions()
print(joint_angles)

# Move one joint of the arm.
joint_angles = robot.get_joint_positions()
joint_angles[-2] -= 0.1
print(f"Setting joint angles to {joint_angles}")
robot.command_joint_positions(joint_angles)
```

See the [tutorial.ipynb][tutorial] for a longer walkthrough.

## Manual installation

To manually install the latest beta version:

```shell
git clone https://github.com/Reimagine-Robotics/piper_control.git
cd piper_control
pip install .
```

NOTE: The changes on the `main` branch are in beta and may be less stable than
the released versions of the project.

## Generating udev rules for CAN adapters

To avoid needing to run `sudo` to set up the CAN interface, you can create a udev rule to the bitrate and desired name for your CAN adapter.

### Usage

1. Plug in your CAN adapter
2. Run the script:
   ```bash
   sudo ./scripts/generate_udev_rule.bash -i can0 -b 1000000
   ```

   Or name your robot (e.g. myrobot):

   ```bash
   sudo ./scripts/generate_udev_rule.bash -i can0 -n myrobot -b 1000000
   ```
3. Unplug and replug the adapter to test

### Test

```bash
ip link show can0
```

Or if you named it something else:

```bash
ip link show myrobot
```

That's it!

## Linting

To lint the codebase, run:

```bash
uv run pre-commit
```

## Troubleshooting / FAQ

### Is my PiperInterface working?

This snippet is a good check for whether things are working:

```python
from piper_control import piper_interface
from piper_control import piper_init

robot = piper_interface.PiperInterface(can_port="can0")

piper_init.reset_arm(robot)
print(robot.get_joint_positions())
print(robot.get_joint_velocities())
print(robot.get_joint_efforts())
print(robot.get_gripper_state())
```

If you get the following text, then the CAN connection is not working. See
[this section](#canexceptionscanerror-errors) on how to debug the CAN
connection.

```text
<class 'can.exceptions.CanError'> Message NOT sent
```

If you get output that looks like this (as in the CAN seems like it is working,
but you get all 0's for the rest of the output), see
[this section](#get-all-zeros-when-calling-get_joint_positions).

```text
can0  is exist
can0  is UP
can0  bitrate is  1000000
can0 bus opened successfully.
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
(0.0, 0.0)
```

### Get all zeros when calling `get_joint_positions()`

Run through these steps:

```python
# Assume you already have the PipeInterface object.
robot = piper_interface.PiperInterface(can_port="can0")

# Reset again.
piper_init.reset_arm(robot)
```

And after that, calling `robot.get_joint_positions()` should return non-zero
values. If it doesn't, then double-check the CAN connection. See
[this section](#canexceptionscanerror-errors).

### `can.exceptions.CanError` errors

Quite often, you may see the following error while connecting or resetting your
arm:

```text
<class 'can.exceptions.CanError'> Message NOT sent
```

There are several possible issues here, and several things you can try to fix
it:

1.  Unplug the CAN cable, power off the robot, power on the robot, then plug the
    CAN cable back in.

    If this works, but the error happens often enough, the issue is the USB
    cable. The CAN adapter that ships with the Piper is finnicky and sensitive
    to the USB cable used.

    Be sure to call `piper_init.reset()` on the robot after starting it again.

2.  Still doesn't work?

    You can check whether the cable itself is working by making sure the CAN
    adapter is visible:

    ```shell
    lsusb | grep CAN
    ```

    If nothing shows up, the cable is not working.

    As mentioned above, the CAN adapter is sensitive to this piece.

3.  Still not working? Make sure the CAN interface is set up correctly.

    Some things to try:

    1.  Run:

        ```shell
        ifconfig
        ```

        And verify the output has your CAN interface (e.g. `can0`).

    2.  Check the bitrate of the interface:

        ```shell
        ip -details link show can0 | grep bitrate
        ```

        Verify this is set to something like 1000000.

    3.  Check the state of the CAN interface:

        ```shell
        ip -details link show can0 | grep "can state"
        ```

        If this is `ERROR-ACTIVE` or `ERROR-PASSIVE`, something is wrong here.

    If any of these are not working, try resetting the CAN connection. You can
    re-run `piper_connect.activate()` or run the steps manually here:

    ```shell
    sudo ip link set can0 down
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set can0 up
    ```

    Check the state afterwards:

    ```shell
    ip -details link show can0 | grep "can state"
    ```

    Try resetting again for good measure:

    ```shell
    sudo ip link set can0 down
    sudo ip link set can0 up
    ```

4.  Still not working? Likely one of the other components are not working.
    Make sure the high-low cables connected to your CAN adapter are inserted
    properly. This is the most common failure mode we've seen. In particular,
    ensure that the wire ends are wedged __at the top__ of the hole, not the
    bottom.

    If needed, swap out your main cable (the aviation connector in the back of
    the robot) and try again. Try swapping out the CAN adapter too if needed.

[tutorial]: <https://github.com/Reimagine-Robotics/piper_control/blob/main/tutorial.ipynb> "Tutorial"
