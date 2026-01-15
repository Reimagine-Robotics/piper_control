# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

-   Add utility method for getting the serial number of a USB can adapter.
-   Allow moving specific joints using MIT mode.

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
