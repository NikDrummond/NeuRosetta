"""Functions for 3D geometric analysis of tree graphs."""

from numpy import isin, ndarray, hstack
from graph_tool.all import DFSVisitor

from ...core import _Tree
from ...utils.geometry_utils import (
    align_with,
    angle,
    angular_mean,
    angular_var,
    bisector,
    cross,
    normalize,
    planar_angle,
    signed_angle,
)
from ...utils.graph_utils import (
    bifurcation_indices,
    g_has_property,
    get_property,
    set_property,
    root_index,
    branch_indices,
    leaf_indices,
    AngleVisitor,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_coordinates import get_edge_coordinates, get_root_coordinate
from .tree_counting import count_edges
from .tree_traversals import depth_first_search

UnitVector = tuple[ndarray, ndarray, ndarray]
BifurcationUnitVectors = tuple[UnitVector, UnitVector, UnitVector]


def get_edge_angles(
    tree: _Tree,
    between_vector: tuple | ndarray,
    perspective_vector: None | tuple | ndarray = None,
    alignment_vector: None | tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    bind: bool = False,
) -> None | ndarray:
    """Compute the angle of each edge relative to a reference direction.

    For each edge, the angle is measured between the normalized edge direction
    and *between_vector* in the plane perpendicular to *perspective_vector*.
    When *perspective_vector* is None, it defaults to the cross product of
    each edge direction with *between_vector*.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray | None, optional
        Look vector defining the plane normal for angle measurement. When
        None, computed per edge as ``cross(edge, between_vector)``.
        By default None.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction
        before measuring angles. By default None.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.
    signed : bool, optional
        If True, return signed planar angles via :func:`signed_angle`; if
        False, return unsigned angles via :func:`planar_angle`.
        By default True.
    bind : bool, optional
        If True, store results as edge property ``Edge_angle`` and return
        None. By default False.

    Returns
    -------
    ndarray | None
        Per-edge angles with length equal to the number of edges when
        bind=False; None when bind=True.
    """

    # subtract source from target and normalize
    starts, stops = get_edge_coordinates(tree, SoA=False)
    vectors = stops - starts
    xv, yv, zv = vectors.T
    xv, yv, zv = normalize(xv, yv, zv)

    # if no perspective vectoer given, use cross of between and edges
    if perspective_vector is None:
        perspective_vector = cross(xv, yv, zv, *between_vector)

    # Align perspective normal globally when an alignment vector is given.
    if alignment_vector is not None:
        normal = align_with(*perspective_vector, *alignment_vector)
    else:
        normal = perspective_vector

    # get angles
    if signed:
        edge_angles = signed_angle(xv, yv, zv, *between_vector, *normal, degrees)
    else:
        edge_angles = planar_angle(xv, yv, zv, *between_vector, *normal, degrees)

    if bind:
        c = not g_has_property(tree.graph, "Edge_angle", "e")
        set_property(
            tree.graph, "Edge_angle", edge_angles, "e", dtype="double", create=c
        )
        return
    return edge_angles


def get_mean_edge_angle(
    tree: _Tree,
    between_vector: tuple | ndarray,
    perspective_vector: None | tuple | ndarray,
    alignment_vector: None | tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    weights: None | ndarray = None,
) -> float:
    """Compute the circular mean of edge angles relative to a reference direction.

    Reuses a cached ``Edge_angle`` edge property when present; otherwise
    calls :func:`get_edge_angles`. The mean is computed with
    :func:`angular_mean`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray | None
        Look vector defining the plane normal for signed angles. Passed to
        :func:`get_edge_angles` when ``Edge_angle`` is not cached.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction
        before measuring angles. By default None.
    degrees : bool, optional
        Return the mean in degrees instead of radians. By default False.
    signed : bool, optional
        Use signed planar angles when computing uncached edge angles.
        By default True.
    weights : ndarray | None, optional
        Per-edge weights; length must equal the number of edges. By default
        None.

    Returns
    -------
    float
        Circular mean edge angle in radians (or degrees when *degrees* is
        True).
    """
    if weights is not None:
        n_edges = count_edges(tree)
        assert n_edges == len(weights), (
            "If weights are given, len(weights) must match the number of edges "
            f"({n_edges}), not {len(weights)}"
        )

    if g_has_property(tree.graph, "Edge_angle", "e"):
        edge_angles = get_property(tree.graph, "Edge_angle", "e")
    else:
        edge_angles = get_edge_angles(
            tree, between_vector, perspective_vector, alignment_vector, degrees, signed
        )

    return angular_mean(edge_angles, weights)


def get_edge_angle_variance(
    tree: _Tree,
    between_vector: tuple | ndarray,
    perspective_vector: None | tuple | ndarray = None,
    alignment_vector: None | tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    weights: None | ndarray = None,
) -> float:
    """Compute the circular variance of edge angles relative to a reference direction.

    Reuses a cached ``Edge_angle`` edge property when present; otherwise
    calls :func:`get_edge_angles`. The variance is computed with
    :func:`angular_var`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray | None, optional
        Look vector defining the plane normal for signed angles. Passed to
        :func:`get_edge_angles` when ``Edge_angle`` is not cached.
        By default None.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction
        before measuring angles. By default None.
    degrees : bool, optional
        Interpret cached or computed angles in degrees. By default False.
    signed : bool, optional
        Use signed planar angles when computing uncached edge angles.
        By default True.
    weights : ndarray | None, optional
        Per-edge weights; length must equal the number of edges. By default
        None.

    Returns
    -------
    float
        Circular variance of edge angles in radians (or degrees when
        *degrees* is True).
    """
    if weights is not None:
        n_edges = count_edges(tree)
        assert n_edges == len(weights), (
            "If weights are given, len(weights) must match the number of edges "
            f"({n_edges}), not {len(weights)}"
        )

    if g_has_property(tree.graph, "Edge_angle", "e"):
        edge_angles = get_property(tree.graph, "Edge_angle", "e")
    else:
        edge_angles = get_edge_angles(
            tree, between_vector, perspective_vector, alignment_vector, degrees, signed
        )

    return angular_var(edge_angles, weights)


def get_radial_angle(
    tree: _Tree,
    alignment_vector: None | tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    bind: bool = True,
) -> None | ndarray:
    """Compute the radial angle of each edge relative to the tree root.

    For each edge, measures the angle between the direction from the edge
    midpoint toward the target node and the direction from the root to the
    edge midpoint.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    alignment_vector : tuple | ndarray | None, optional
        Reference direction used to orient the sign when *signed* is True.
        Required when *signed* is True. By default None.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.
    signed : bool, optional
        If True, return signed angles via :func:`signed_angle`; if False,
        return unsigned angles via :func:`angle`. By default True.
    bind : bool, optional
        If True, store results as edge property ``Radial_angle`` and return
        None. By default True.

    Returns
    -------
    ndarray | None
        Per-edge radial angles with length equal to the number of edges when
        bind=False; None when bind=True.

    Raises
    ------
    ValueError
        If *signed* is True and *alignment_vector* is None.
    """
    # source and target coords
    source, target = get_edge_coordinates(tree, SoA=True)
    # angle related data:
    vectors = target - source
    vectors = normalize(*vectors)
    section_mid_points = (source + target) / 2
    rc = get_root_coordinate(tree)
    center_vecs = section_mid_points - rc[:, None]
    center_vecs = normalize(*center_vecs)

    section_ends = target - section_mid_points
    section_ends = normalize(*section_ends)

    crosses = cross(*center_vecs, *section_ends)

    if signed:
        if alignment_vector is None:
            raise ValueError("alignment_vector must be given if signed is True")

        normals = normalize(*align_with(*crosses, *alignment_vector))
        a = signed_angle(*section_ends, *center_vecs, *normals, degrees=degrees)
    else:
        a = angle(*section_ends, *center_vecs, degrees=degrees)

    if bind:
        tree.set_property("Radial_angle", a, "e", dtype="double", create=True)
        return

    return a


# get_bifurcation_geometry


### Get ALL the needed vectors (not normals)
def _get_bifurcation_unit_vectors(tree: _Tree) -> BifurcationUnitVectors:
    """Return unit vectors at each bifurcation for geometry calculations.

    For every bifurcation node, returns normalized vectors from the branch
    point to each child (``bc1``, ``bc2``) and to the parent (``sb``).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    tuple[tuple[ndarray, ndarray, ndarray], ...]
        ``(bc1, bc2, sb)`` where each element is an ``(x, y, z)`` component
        tuple of per-bifurcation unit vectors.
    """
    # get bifurcation inds
    b_inds = bifurcation_indices(tree.graph)

    # get coordinates + branch poit coordinates
    coords = tree.get_node_coordinates()
    branch_coords = coords[b_inds]
    # get edges
    edges = tree.get_edge_indices()

    ### Needed node indices around bifurcations

    # get child edges, as in the source is branch - we index to only get the target of the edge
    source_inds = isin(edges[:, 0], b_inds)
    child_edges_to_keep = edges[source_inds]
    child_edges = child_edges_to_keep[:, 1]

    # split into odd and even
    even_child = child_edges[::2]
    odd_child = child_edges[1::2]

    # now we want the parents of the branches - again we index, this time to only get the source
    source_inds = isin(edges[:, 1], b_inds)
    parent_edges = edges[source_inds][:, 0]

    # node coordinates
    v_bc1 = coords[even_child]
    v_bc2 = coords[odd_child]
    v_sb = coords[parent_edges]

    # subtract branch_coords
    v_bc1 -= branch_coords
    v_bc2 -= branch_coords
    v_sb -= branch_coords

    # transpose for numba kernels
    v_bc1 = v_bc1.T
    v_bc2 = v_bc2.T
    v_sb = v_sb.T

    # normalize to return unit vectors

    return normalize(*v_bc1), normalize(*v_bc2), normalize(*v_sb)


def get_bifurcation_angles(
    tree: _Tree, degrees: bool = False
) -> tuple[ndarray, ndarray, ndarray]:
    """Compute planar angles at each bifurcation node.

    For each bifurcation, returns the parent-to-first-child angle
    (``pc1_angles``), parent-to-second-child angle (``pc2_angles``), and
    inter-child angle (``c_angles``).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        ``(pc1_angles, pc2_angles, c_angles)``, one value per bifurcation.
    """
    bc1, bc2, sb = _get_bifurcation_unit_vectors(tree)
    # normals
    child_normals = cross(*bc1, *bc2)
    pc1_normals = cross(*bc1, *sb)
    pc2_normals = cross(*bc2, *sb)

    # angles
    c_angles = planar_angle(*bc1, *bc2, *child_normals, degrees=degrees)
    pc1_angles = planar_angle(*sb, *bc1, *pc1_normals, degrees=degrees)
    pc2_angles = planar_angle(*sb, *bc2, *pc2_normals, degrees=degrees)

    return pc1_angles, pc2_angles, c_angles


def get_bifurcation_angle_sums(tree: _Tree, degrees: bool = False) -> ndarray:
    """Compute the sum of parent-child and inter-child angles at bifurcations.

    Equivalent to ``pc1 + pc2 + c`` from :func:`get_bifurcation_angles`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.

    Returns
    -------
    ndarray
        Angle sum at each bifurcation node.
    """
    pc1_angles, pc2_angles, c_angles = get_bifurcation_angles(tree, degrees=degrees)
    return pc1_angles + pc2_angles + c_angles


# dehedral beta
def get_bifurcation_deihedral_beta(tree: _Tree, degrees: bool = False) -> ndarray:
    """Compute the dihedral beta angle at each bifurcation.

    Measures the planar angle between the parent branch vector and the
    bisector of the two child branch vectors, viewed along the plane normal
    defined by the child cross product.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.

    Returns
    -------
    ndarray
        Dihedral beta angle at each bifurcation node.
    """
    # get normal bifurcation vectors
    bc1, bc2, sb = _get_bifurcation_unit_vectors(tree)

    # bisection of child vectors
    cc_bisector = bisector(*bc1, *bc2)

    # cross of child vectors
    cc_cross = cross(*bc1, *bc2)

    # cross of the cross and the bisector
    normal = cross(*cc_cross, *cc_bisector)

    # Dihedral beta - angle between parent and bisector from perspective of normal
    return planar_angle(*sb, *cc_bisector, *normal, degrees=degrees)


# angular mean/var of sections for non-reduced trees
def get_section_angular_deviation(
    tree: _Tree, return_Visitor: bool = False
) -> tuple[ndarray, ndarray] | DFSVisitor:
    """Compute per-section angular mean and variance for a non-reduced tree.

    Sections run between branch nodes (and the root) and end at branch or
    leaf nodes. For each section, edge directions are compared to the section
    axis and summarised with :func:`angular_mean` and :func:`angular_var`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree. Must not be reduced.
    return_Visitor : bool, optional
        If True, return the underlying :class:`AngleVisitor` instead of the
        summary arrays. By default False.

    Returns
    -------
    tuple[list, list] | AngleVisitor
        ``(mean_angles, angle_variances)`` per section when
        return_Visitor=False; otherwise the visitor instance.

    Raises
    ------
    AttributeError
        If the tree is reduced.
    """
    if tree.is_reduced():
        raise AttributeError("Cannot calculate angular deviation for a reduced Neuron")

    root = root_index(tree.graph)

    # get start and stop inds of graph if reduced
    b_inds = branch_indices(tree.graph)
    l_inds = leaf_indices(tree.graph)

    # starts are branches and the root
    starts = hstack([b_inds, [root]]) if root not in b_inds else b_inds
    # stops are branches (without the root!) and leaves
    stops = hstack([b_inds[b_inds != root], l_inds])

    vis = depth_first_search(
        tree=tree,
        visitor=AngleVisitor,
        root=root,
        init_kwargs={
            "graph": tree.graph,
            "starts": starts,
            "stops": stops,
            "x_prop": tree.graph.vp["x"],
            "y_prop": tree.graph.vp["y"],
            "z_prop": tree.graph.vp["z"],
        },
    )

    if return_Visitor:
        return vis

    return vis.mean_angles, vis.angle_variances


enrich_tree_graph_docstrings(globals())
