"""piper_control module-level properties."""

from ._version import __version__

try:
  # This part is dynamically generated during the module's build process or
  # obtained at runtime if the .git directory is available (unlikely for
  # installed packages)
  from ._git_hash import __git_hash__  # type: ignore
except ImportError:
  __git_hash__ = "unknown"  # Fallback if the git hash cannot be determined.
