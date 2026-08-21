"""Declarative style state for 3D neuron plots.

The style is stored independently of any built vedo actors so it can be set on an
empty plot shell and applied later, when actors are generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

DEFAULT_COLOUR = "k"
DEFAULT_LW = 1.0
DEFAULT_ALPHA = 1.0
DEFAULT_ROOT_SIZE = 12.0

# Raw vedo constructor kwargs that map onto stored style fields.
LINE_KWARG_ALIASES = {
    "c": "colour",
    "color": "colour",
    "colour": "colour",
    "lw": "lw",
    "linewidth": "lw",
    "line_width": "lw",
    "Line_width": "lw",
    "alpha": "alpha",
    "a": "alpha",
    "opacity": "alpha",
}

ROOT_KWARG_ALIASES = {
    "c": "root_colour",
    "color": "root_colour",
    "colour": "root_colour",
    "root_c": "root_colour",
    "rc": "root_colour",
    "r": "root_size",
    "radius": "root_size",
    "ps": "root_size",
    "root_s": "root_size",
    "rs": "root_size",
    "alpha": "root_alpha",
    "a": "root_alpha",
    "opacity": "root_alpha",
    "root_a": "root_alpha",
    "ra": "root_alpha",
}


def split_style_kwargs(kwargs: dict | None, aliases: dict[str, str]) -> tuple[dict, dict]:
    """Split raw vedo kwargs into recognised style fields and passthrough extras.

    Parameters
    ----------
    kwargs : dict | None
        Raw keyword arguments intended for a vedo constructor.
    aliases : dict[str, str]
        Mapping of accepted kwarg names to style field names, e.g.
        :data:`LINE_KWARG_ALIASES`.

    Returns
    -------
    tuple[dict, dict]
        ``(style_fields, extras)`` where *extras* are kwargs with no style
        equivalent and must be forwarded to the constructor verbatim.
    """
    style_fields: dict = {}
    extras: dict = {}
    for key, value in (kwargs or {}).items():
        field_name = aliases.get(key)
        if field_name is None:
            extras[key] = value
        else:
            style_fields[field_name] = value
    return style_fields, extras


@dataclass
class Plot3DStyle:
    """Visual style of a :class:`~neurosetta.ops.plotting.utils.TreePlot3D`.

    Attributes
    ----------
    colour : Any
        Line colour, as any vedo colour specifier.
    lw : float
        Line width.
    alpha : float
        Line opacity in ``[0, 1]``.
    root_colour : Any | None
        Root marker colour. None follows :attr:`colour`.
    root_alpha : float | None
        Root marker opacity. None follows :attr:`alpha`.
    root_size : float
        Root marker point size (or sphere radius on the k3d backend).
    show_root : bool
        Whether the root marker is included in the plot's actor list.
    cmap : str | None
        Colour map name used when edges are coloured by scalar values.
    line_kwargs : dict
        Extra vedo ``Lines`` kwargs with no dedicated style field.
    root_kwargs : dict
        Extra root marker kwargs with no dedicated style field.
    """

    colour: Any = DEFAULT_COLOUR
    lw: float = DEFAULT_LW
    alpha: float = DEFAULT_ALPHA
    root_colour: Any | None = None
    root_alpha: float | None = None
    root_size: float = DEFAULT_ROOT_SIZE
    show_root: bool = True
    cmap: str | None = None
    line_kwargs: dict = field(default_factory=dict)
    root_kwargs: dict = field(default_factory=dict)

    @property
    def effective_root_colour(self) -> Any:
        """Root colour actually used, falling back to the line colour."""
        return self.colour if self.root_colour is None else self.root_colour

    @property
    def effective_root_alpha(self) -> float:
        """Root opacity actually used, falling back to the line opacity."""
        return self.alpha if self.root_alpha is None else self.root_alpha

    def line_build_kwargs(self) -> dict:
        """Return the full kwarg set for constructing the vedo ``Lines`` actor."""
        return {"c": self.colour, "lw": self.lw, "alpha": self.alpha} | self.line_kwargs

    def root_build_kwargs(self) -> dict:
        """Return the full kwarg set for constructing the root marker actor."""
        base = {
            "r": self.root_size,
            "c": self.effective_root_colour,
            "alpha": self.effective_root_alpha,
        }
        return base | self.root_kwargs

    def copy(self) -> Plot3DStyle:
        """Return an independent copy, including the extra kwarg dicts."""
        return replace(
            self,
            line_kwargs=dict(self.line_kwargs),
            root_kwargs=dict(self.root_kwargs),
        )

    def as_dict(self) -> dict:
        """Return the style as a plain dict."""
        return asdict(self)
