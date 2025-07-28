"""Custom build backend for piper_control.

Does an additional step to generate the _git_version.py file containing the git
hash of the current commit. This is used to provide versioning information at
runtime. It is expected that this file is generated during the build process or
at runtime if the .git directory is available.
"""

import pathlib
import subprocess

import setuptools.build_meta as original_backend


def _write_git_hash():
  try:
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
  except Exception:  # pylint: disable=broad-except
    git_hash = "unknown"

  version_file = (
      pathlib.Path(__file__).parent.parent
      / "src"
      / "piper_control"
      / "_git_hash.py"
  )
  version_file.write_text(f'__git_hash__ = "{git_hash}"\n')


# Wrapper functions
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
  _write_git_hash()
  return original_backend.build_wheel(
      wheel_directory, config_settings, metadata_directory
  )


def build_sdist(sdist_directory, config_settings=None):
  _write_git_hash()
  return original_backend.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
  _write_git_hash()
  return original_backend.prepare_metadata_for_build_wheel(
      metadata_directory, config_settings
  )


def build_editable(
    wheel_directory, config_settings=None, metadata_directory=None
):
  _write_git_hash()
  return original_backend.build_editable(
      wheel_directory, config_settings, metadata_directory
  )
