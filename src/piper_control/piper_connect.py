"""Python implementation of AgileX CAN connection shell scripts.

This module provides functions to find, activate, and manage CAN interfaces.
There's nothing wrong with the scripts in piper_sdk, but they aren't available
in obvious way when pip installing piper_sdk, and you can't easily invoke them
from higher-level Python code.

NOTE: This module is intended for Linux systems with CAN interfaces. It uses
`ip` and `ethtool` commands to manage CAN interfaces. Ensure you have the
necessary permissions to run these commands (e.g., using `sudo`).
"""

import subprocess
import time
from typing import List, Optional, Tuple

# ------------------------
# Public API
# ------------------------


def find_ports() -> List[Tuple[str, str]]:
  """Return a list of (interface, usb_address) pairs."""
  _check_dependencies()
  return _get_can_interfaces()


def active_ports() -> List[str]:
  """Return list of CAN interfaces that are currently up."""
  _check_dependencies()
  result = []
  for iface, _ in _get_can_interfaces():
    if _interface_is_up(iface):
      result.append(iface)
  return result


def activate(
    ports: Optional[List[Tuple[str, str]]] = None,
    default_name_prefix: str = "can",
    default_bitrate: int = 1000000,
    timeout: Optional[int] = None,
):
  """Activate all provided ports, or auto-discover and activate all known CAN
  interfaces.

  Args:
    ports: Optional list of (interface, usb_address) pairs to activate. If None,
      all available ports are used.
    default_name_prefix: Prefix for naming interfaces (e.g., can0, can1, ...).
    default_bitrate: Bitrate to set for each CAN interface.
    timeout: Optional timeout in seconds to wait for CAN devices to appear (if
      none are found initially).
  """
  _check_dependencies()
  ports = ports or _get_can_interfaces()

  if not ports and timeout:
    start = time.time()
    while time.time() - start < timeout:
      ports = _get_can_interfaces()
      if ports:
        break
      time.sleep(5)
    if not ports:
      raise TimeoutError(
          f"Timed out after {timeout}s waiting for CAN devices to appear"
      )

  ports = sorted(ports, key=lambda p: p[1])  # Sort by usb_addr

  for idx, (iface, _) in enumerate(ports):
    target_name = f"{default_name_prefix}{idx}"
    current_bitrate = _get_interface_bitrate(iface)
    if current_bitrate == default_bitrate and iface == target_name:
      continue  # Already configured
    _rename_and_configure(iface, target_name, default_bitrate)


def get_can_adapter_serial(can_port: str) -> str | None:
  """Convenience method that returns the serial number of a USB CAN adapter."""
  ethtool_out = subprocess.check_output(["ethtool", "-i", can_port], text=True)

  usb_port = None
  for l in ethtool_out.splitlines():
    if "bus-info" in l:
      usb_port = l.split()[-1].split(":")[0]

  if usb_port:
    serial_file = f"/sys/bus/usb/devices/{usb_port}/serial"
    try:
      with open(serial_file, "r", encoding="utf-8") as file:
        return file.read().strip()
    except FileNotFoundError:
      return None

  return None


# ------------------------
# Internal Utility Methods
# ------------------------


def _check_dependencies() -> None:
  for pkg in ["ethtool", "can-utils"]:
    try:
      subprocess.run(["dpkg", "-s", pkg], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
      raise RuntimeError(
          f"Missing dependency: {pkg}. Please install with `sudo apt install "
          f"{pkg}`."
      ) from exc


def _get_can_interfaces() -> List[Tuple[str, str]]:
  """Return a list of (interface, usb_address) pairs."""
  result = []
  try:
    links = subprocess.check_output(
        ["ip", "-br", "link", "show", "type", "can"], text=True
    )
    for line in links.splitlines():
      iface = line.split()[0]
      try:
        ethtool = subprocess.check_output(["ethtool", "-i", iface], text=True)
        for l in ethtool.splitlines():
          if "bus-info" in l:
            usb_addr = l.split()[-1]
            result.append((iface, usb_addr))
      except subprocess.CalledProcessError:
        continue
  except subprocess.CalledProcessError:
    pass
  return result


def _get_interface_bitrate(interface: str) -> Optional[int]:
  try:
    details = subprocess.check_output(
        ["ip", "-details", "link", "show", interface], text=True
    )
    for line in details.splitlines():
      if "bitrate" in line:
        return int(line.split("bitrate")[-1].strip().split()[0])
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return None


def _interface_exists(name: str) -> bool:
  try:
    subprocess.check_output(
        ["ip", "link", "show", name], stderr=subprocess.DEVNULL
    )
    return True
  except subprocess.CalledProcessError:
    return False


def _interface_is_up(name: str) -> bool:
  try:
    output = subprocess.check_output(["ip", "link", "show", name], text=True)
    return "state UP" in output
  except subprocess.CalledProcessError:
    return False


def _rename_and_configure(interface: str, target_name: str, bitrate: int):
  subprocess.run(["sudo", "ip", "link", "set", interface, "down"], check=True)
  subprocess.run(
      [
          "sudo",
          "ip",
          "link",
          "set",
          interface,
          "type",
          "can",
          "bitrate",
          str(bitrate),
      ],
      check=True,
  )

  if target_name != interface:
    if _interface_exists(target_name):
      print(
          f"[WARN] Target name '{target_name}' already exists. Skipping rename "
          f"of '{interface}'."
      )
    else:
      subprocess.run(
          ["sudo", "ip", "link", "set", interface, "name", target_name],
          check=True,
      )
      interface = target_name

  subprocess.run(["sudo", "ip", "link", "set", interface, "up"], check=True)
