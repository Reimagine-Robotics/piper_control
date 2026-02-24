# Changelog

All notable changes to this project will be documented in this file.

## [1.3.7]

-   Default to old-firmware J1-3 scaling when firmware version is unknown for
    backward compatibility.

## [1.3.6]

-   Fix sign error in joint 6 max limit for Piper H and L.
-   Only apply J1-3 command scaling (0.25) for firmware older than 1.8.

## [1.3.5]

-   Fix swapped limit/comms error bits in status calculation error code decoding.
-   Don't move the gripper when the arm enables.

## [1.3.2]

-   Default gravity model to direct mode, which doesn't require samples.
-   Minor bugfix in gravity compensation script.

## [1.3.0]

-   Add per-arm/gripper limit helpers plus `PiperArmType`/`PiperGripperType` and
    expose limits via `PiperInterface`.
-   Deprecate module-level joint/gripper limit constants in favor of the new
    helper accessors and instance properties.

## [0.1.0]

Initial version of `piper_control`. Wrapper around `piper_sdk` with some basic
APIs for resetting and controlling the arm. The code is still a WIP under active
development with known issues around resetting the arm.

Features:

-   `piper_control` - module for controlling the arm.
-   `piper_connect` - module for activating and configuring the CAN interfaces.
