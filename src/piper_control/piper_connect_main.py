"""Script to connect to all Piper arms connected to the machine."""

from piper_control import piper_connect


def main():
  """Activates connection to all arms."""
  print(
      "Will find and activate all available ports. Make sure you have sudo "
      "access."
  )
  print(f"All available ports: {piper_connect.find_ports()}")
  print("Activating all ports...")
  piper_connect.activate()
  print(f"All activate ports: {piper_connect.active_ports()}")
  print("Done.")


if __name__ == "__main__":
  main()
