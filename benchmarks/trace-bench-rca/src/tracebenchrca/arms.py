"""The arm registry: an arm name → its class, its declared inputs, the axes it is scored on
and the code it runs (`constants.ARMS`, registered in `plans/rca-arms.md` and
`plans/controls.md`).

A result cell is keyed `<arm>/<axis>` inside one corpus's results directory
(`results/<rung>/latent/seed=<k>/<arm slug>/`); the arm slug is the file-name form of the
arm name (`rca/baro-bocpd` → `rca-baro-bocpd`).
"""

from __future__ import annotations

from .constants import ARMS, AXES, CLASSES, INPUTS


def arm_spec(name):
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; registered: {sorted(ARMS)}")
    return ARMS[name]


def arm_slug(name):
    """File-name form of an arm name (`rca/baro-bocpd` → `rca-baro-bocpd`)."""
    return arm_spec(name) and name.replace("/", "-")


def arm_class(name):
    cls = arm_spec(name)["class"]
    if cls not in CLASSES:
        raise ValueError(f"arm {name!r} declares class {cls!r}, not one of {CLASSES}")
    return cls


def inputs_of(name):
    """The declared inputs of an arm; a stage that opens anything else is a registry violation."""
    inputs = arm_spec(name)["inputs"]
    bad = [i for i in inputs if i not in INPUTS]
    if bad:
        raise ValueError(f"arm {name!r} declares unknown inputs {bad}")
    return inputs


def check_input(arm, what):
    if what not in inputs_of(arm):
        raise PermissionError(f"arm {arm!r} did not declare input {what!r}; its inputs are {inputs_of(arm)}")
    return what


def axes_of(name):
    axes = arm_spec(name)["axes"]
    bad = [a for a in axes if a not in AXES]
    if bad:
        raise ValueError(f"arm {name!r} declares unknown axes {bad}")
    return axes


def check_axis(arm, axis):
    if axis not in axes_of(arm):
        raise ValueError(f"arm {arm!r} is not scored on axis {axis!r}; its axes are {axes_of(arm)}")
    return axis


def cell_key(arm, axis):
    check_axis(arm, axis)
    return f"{arm}/{axis}"


def parse_cell_key(key):
    arm, axis = key.rsplit("/", 1)
    cell_key(arm, axis)
    return arm, axis


def code_of(name):
    """The pinned code an arm runs (`None` for the analytic and rule controls)."""
    return arm_spec(name)["code"]


__all__ = [
    "arm_spec",
    "arm_slug",
    "arm_class",
    "inputs_of",
    "check_input",
    "axes_of",
    "check_axis",
    "cell_key",
    "parse_cell_key",
    "code_of",
]
