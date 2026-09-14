"""Graph reduction provenance maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReductionMap:
    """Maps original morphology edges onto reduced section edges.

    Each reduced edge ``i`` collapses an ordered chain of original edges
    traversed in the directed root→leaf orientation.

    Attributes
    ----------
    section_edge_indices
        ``section_edge_indices[i]`` is the list of original edge indices that
        compose reduced edge ``i``, in directed section order.
    section_edge_lengths
        Matching cable lengths for each original edge in the section.
    old_edge_to_section
        For every original edge index that appears in a section::

            (reduced_edge_index, offset_before, edge_length, section_length)

        where ``offset_before`` is the cable distance from the reduced-edge
        source to the start of that original edge.
    n_original_edges
        Number of edges in the pre-reduction graph.
    n_original_vertices
        Number of vertices in the pre-reduction graph.
    """

    section_edge_indices: tuple[tuple[int, ...], ...]
    section_edge_lengths: tuple[tuple[float, ...], ...]
    old_edge_to_section: dict[int, tuple[int, float, float, float]]
    n_original_edges: int
    n_original_vertices: int

    @classmethod
    def from_sections(
        cls,
        section_edge_indices: list[list[int]],
        section_edge_lengths: list[list[float]],
        *,
        n_original_edges: int,
        n_original_vertices: int,
    ) -> ReductionMap:
        """Build a :class:`ReductionMap` from per-section edge lists."""
        old_to_section: dict[int, tuple[int, float, float, float]] = {}
        sec_idx: list[tuple[int, ...]] = []
        sec_len: list[tuple[float, ...]] = []
        for new_e, (einds, elens) in enumerate(
            zip(section_edge_indices, section_edge_lengths, strict=True)
        ):
            lengths = [float(x) for x in elens]
            total = float(sum(lengths))
            offset = 0.0
            for old_e, L in zip(einds, lengths, strict=True):
                old_to_section[int(old_e)] = (int(new_e), offset, float(L), total)
                offset += float(L)
            sec_idx.append(tuple(int(i) for i in einds))
            sec_len.append(tuple(lengths))
        return cls(
            section_edge_indices=tuple(sec_idx),
            section_edge_lengths=tuple(sec_len),
            old_edge_to_section=old_to_section,
            n_original_edges=int(n_original_edges),
            n_original_vertices=int(n_original_vertices),
        )

    @property
    def n_reduced_edges(self) -> int:
        return len(self.section_edge_indices)

    def transfer_edge_fraction(
        self,
        old_edge_index: int,
        edge_fraction: float,
    ) -> tuple[int, float, float]:
        """Map ``(old_edge, fraction)`` → ``(new_edge, new_fraction, distance_along)``.

        Uses cable distance along the original section (not the Euclidean chord).
        """
        try:
            new_e, offset, length, section_length = self.old_edge_to_section[int(old_edge_index)]
        except KeyError as exc:
            raise KeyError(
                f"Original edge {old_edge_index} was not retained in any reduced section"
            ) from exc
        t = float(np.clip(edge_fraction, 0.0, 1.0))
        distance_along = offset + t * length
        if section_length <= 0.0:
            new_frac = 0.0
        else:
            new_frac = float(np.clip(distance_along / section_length, 0.0, 1.0))
        return new_e, new_frac, distance_along
