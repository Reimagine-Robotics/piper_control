"""Python implementation of AgileX CAN connection shell scripts.

This module provides functions to find, activate, and manage CAN interfaces.
There's nothing wrong with the scripts in piper_sdk, but they aren't available
in obvious way when pip installing piper_sdk, and you can't easily invoke them
from higher-level Python code.
"""

import subprocess
import time

# ------------------------
# Public API
# ------------------------


def find_ports() -> list[tuple[str, str]]:
  """Return a list of (interface, usb_address) pairs."""
  _check_dependencies()
  return _get_can_interfaces()


def active_ports() -> list[str]:
  """Return list of CAN interfaces that are currently up."""
  _check_dependencies()
  result = []
  for iface, _ in _get_can_interfaces():
    if _interface_is_up(iface):
      result.append(iface)
  return result


def activate(
    ports: list[tuple[str, str]] | None = None,
    default_bitrate: int = 1000000,
    timeout: int | None = None,
):
  """Activate all provided ports, or auto-discover and activate all known CAN
  interfaces.

  NOTE: This function is intended for Linux systems with CAN interfaces. It uses
  `ip` and `ethtool` commands to manage CAN interfaces. Ensure you have the
  necessary permissions to run these commands (e.g., using `sudo`).

  Args:
    ports: Optional list of (interface, usb_address) pairs to activate. If None,
      all available ports are used.
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

  for iface, _ in ports:
    current_bitrate = _get_interface_bitrate(iface)
    if current_bitrate == default_bitrate:
      continue  # Already configured
    _configure(iface, default_bitrate)


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
      with open(serial_file, encoding="utf-8") as file:
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


def _get_can_interfaces() -> list[tuple[str, str]]:
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


def _get_interface_bitrate(interface: str) -> int | None:
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


def _configure(interface: str, bitrate: int):
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
  subprocess.run(["sudo", "ip", "link", "set", interface, "up"], check=True)
