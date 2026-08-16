"""Tests for Forest metadata get/set/del, summaries, and index lookup."""

import pytest

from neurosetta.api import Forest, Tree
from neurosetta.core.metadata import PROTECTED_META_KEYS


def _forest(simple_tree, n: int = 3) -> Forest:
    base = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    trees = [base]
    for i in range(2, n + 1):
        tree = base.copy()
        tree.ID = i
        trees.append(tree)
    return Forest(trees)


def test_get_meta_per_tree(simple_tree):
    forest = _forest(simple_tree, n=4)
    forest.set_meta("score", [0.1, 0.2, 0.3, 0.4])

    assert forest.get_meta("score") == [0.1, 0.2, 0.3, 0.4]
    assert forest.get_meta("missing", default=-1) == [-1, -1, -1, -1]


def test_set_meta_broadcasts_length3_numeric_list(simple_tree):
    """Length-3 numeric sequences are geometry-like and broadcast, not split."""
    forest = _forest(simple_tree, n=3)
    forest.set_meta("vector_tag", [0.1, 0.2, 0.3])

    assert forest.get_meta("vector_tag") == [[0.1, 0.2, 0.3]] * 3


def test_set_meta_broadcast(simple_tree):
    forest = _forest(simple_tree)
    forest.set_meta("Neuron_type", "T4")

    assert forest.get_meta("Neuron_type") == ["T4", "T4", "T4"]


def test_set_meta_per_tree_iterable(simple_tree):
    forest = _forest(simple_tree, n=2)
    forest.set_meta("Neuron_type", ["T4", "T5"])

    assert forest.get_meta("Neuron_type") == ["T4", "T5"]


def test_has_meta(simple_tree):
    forest = _forest(simple_tree, n=2)
    forest.set_meta("a", "x")

    assert forest.has_meta("a") == [True, True]
    assert forest.has_meta("b") == [False, False]


def test_del_meta(simple_tree):
    forest = _forest(simple_tree, n=2)
    forest.set_meta("a", "x")
    forest.del_meta("a")

    assert forest.has_meta("a") == [False, False]


def test_del_meta_skips_missing(simple_tree):
    forest = _forest(simple_tree, n=2)
    forest.set_meta("a", "x")
    forest._trees[1].del_meta("a")

    forest.del_meta("a")
    assert forest.has_meta("a") == [False, False]


@pytest.mark.parametrize("key", sorted(PROTECTED_META_KEYS))
def test_set_meta_protected_raises(simple_tree, key):
    forest = _forest(simple_tree, n=1)
    with pytest.raises(KeyError, match="protected"):
        forest.set_meta(key, "x")


def test_list_meta_union(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("a", "x")
    forest._trees[1].set_meta("b", 1)

    assert forest.list_meta() == ["a", "b"]


def test_meta_summary(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("a", "x")
    forest._trees[2].set_meta("b", 1)

    assert forest.meta_summary() == {"a": 3, "b": 1}


def test_meta_indices_scalar(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("Neuron_type", ["T4", "T5", "T4"])

    assert forest.meta_indices("Neuron_type", "T4") == [0, 2]
    assert forest.meta_indices("Neuron_type", "T5") == [1]
    assert forest.meta_indices("Neuron_type", "T6") == []


def test_meta_indices_membership(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("Neuron_type", ["T4", "T5", "T6"])

    assert forest.meta_indices("Neuron_type", ["T4", "T6"]) == [0, 2]


def test_meta_indices_missing_key(simple_tree):
    forest = _forest(simple_tree, n=2)
    assert forest.meta_indices("Neuron_type", "T4") == []


def test_filter_still_matches_keyword_conditions(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("Neuron_type", ["T4", "T5", "T4"])

    filtered = forest.filter(Neuron_type="T4")
    assert [t.ID for t in filtered] == [1, 3]


def test_filter_tree_predicate_named_function(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("Neuron_type", ["T4", "T5", "T4"])

    def is_t4(tree):
        return tree.get_meta("Neuron_type") == "T4"

    filtered = forest.filter(is_t4)
    assert [t.ID for t in filtered] == [1, 3]


def test_filter_tree_predicate_structural(simple_tree):
    forest = _forest(simple_tree, n=3)
    n_nodes = forest[0].count_nodes()

    def large_enough(tree):
        return tree.count_nodes() >= n_nodes

    assert len(forest.filter(large_enough)) == 3

    def too_large(tree):
        return tree.count_nodes() > n_nodes

    assert len(forest.filter(too_large)) == 0


def test_filter_tree_predicate_default_args(simple_tree):
    forest = _forest(simple_tree, n=3)
    n_nodes = forest[0].count_nodes()

    def large_enough(tree, min_nodes=n_nodes):
        return tree.count_nodes() >= min_nodes

    assert len(forest.filter(large_enough)) == 3


def test_filter_predicate_and_keywords_raises(simple_tree):
    forest = _forest(simple_tree, n=2)

    def always_true(tree):
        return True

    with pytest.raises(ValueError, match="not both"):
        forest.filter(always_true, Neuron_type="T4")


def test_filter_predicate_non_bool_raises(simple_tree):
    forest = _forest(simple_tree, n=1)

    def not_bool(tree):
        return 1

    with pytest.raises(TypeError, match="must return bool"):
        forest.filter(not_bool)


def test_filter_predicate_wrong_arity_raises(simple_tree):
    forest = _forest(simple_tree, n=1)

    def two_args(tree, extra):
        return True

    with pytest.raises(TypeError, match="exactly one required argument"):
        forest.filter(two_args)


def test_filter_no_criteria_raises(simple_tree):
    forest = _forest(simple_tree, n=2)

    with pytest.raises(ValueError, match="predicate or at least one keyword"):
        forest.filter()


def test_filter_missing_metadata_key_does_not_match(simple_tree):
    forest = _forest(simple_tree, n=2)
    forest._trees[0].set_meta("score", 1)

    assert [t.ID for t in forest.filter(score=1)] == [1]
    assert forest.filter(score=1).ids() == [1]

    forest._trees[1].set_meta("score", None)
    assert [t.ID for t in forest.filter(score=None)] == [2]


def test_filter_keyword_matches_meta_indices(simple_tree):
    forest = _forest(simple_tree, n=3)
    forest.set_meta("Neuron_type", ["T4", "T5", "T4"])

    assert [t.ID for t in forest.filter(Neuron_type="T4")] == [1, 3]
    assert forest.meta_indices("Neuron_type", "T4") == [0, 2]


def test_apply_empty_forest_returns_empty_list():
    forest = Forest([])

    assert forest.apply(lambda t: t.count_nodes()) == []


def test_empty_forest_metadata(simple_tree):
    forest = Forest([])
    assert forest.get_meta("a") == []
    assert forest.list_meta() == []
    assert forest.meta_summary() == {}
    forest.set_meta("a", 1)
    forest.del_meta("a")
