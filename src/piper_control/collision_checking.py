"""Collision checking utilities using MuJoCo."""

from collections import Counter

import mujoco


def get_body_contact_counts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> Counter[tuple[str, str]]:
  """Get contact counts between body pairs in the simulation.

  Args:
      model: MuJoCo model.
      data: MuJoCo data.

  Returns:
      Counter mapping (body1_name, body2_name) tuples to contact counts.
  """
  contacts: Counter[tuple[str, str]] = Counter()

  for contact in data.contact:
    body1_id = model.geom_bodyid[contact.geom1]
    body1_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body1_id,
    )
    body2_id = model.geom_bodyid[contact.geom2]
    body2_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body2_id,
    )
    contacts[(body1_name, body2_name)] += 1

  return contacts


def has_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    disable_collisions: set[tuple[str, str]] | None = None,
    *,
    verbose: bool = False,
) -> bool:
  """Check if the robot has any active collisions.

  Args:
      model: MuJoCo model.
      data: MuJoCo data.
      disable_collisions: Set of body pairs to ignore for collision checking.
      verbose: If True, print contact information.

  Returns:
      True if there are any collisions (excluding disabled pairs).
  """
  if disable_collisions is None:
    disable_collisions = set()

  mujoco.mj_forward(model, data)

  contacts = get_body_contact_counts(model, data)

  for body1_name, body2_name in disable_collisions:
    contacts.pop((body1_name, body2_name), None)
    contacts.pop((body2_name, body1_name), None)

  if verbose and contacts:
    print(f"Contacts: {contacts}")

  return len(contacts) > 0
