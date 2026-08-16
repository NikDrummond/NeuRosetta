"""Tests for _Forest container mutation and lookup helpers."""

from __future__ import annotations

import pytest

from neurosetta.api import Forest, Tree
from neurosetta.io.example_data import example_ids, load_example_data


@pytest.fixture
def example_forest() -> Forest:
    forest = load_example_data()
    assert isinstance(forest, Forest)
    assert len(forest) == len(example_ids)
    return forest


def test_ids_returns_trees_in_order(example_forest):
    assert example_forest.ids() == list(example_ids)


def test_by_id_single_and_sequence(example_forest):
    first_id, second_id, third_id, fourth_id = example_ids

    assert first_id == example_forest.by_id(first_id).ID
    assert example_forest.by_id([fourth_id, second_id]).ids() == [fourth_id, second_id]
    assert example_forest.by_id((third_id, third_id)).ids() == [third_id, third_id]

    with pytest.raises(KeyError, match=f"No tree with ID {fourth_id + 1}"):
        example_forest.by_id(fourth_id + 1)

    with pytest.raises(KeyError, match=f"No tree with ID {fourth_id + 1}"):
        example_forest.by_id([first_id, fourth_id + 1])


def test_append_adds_tree(example_forest):
    new_tree = example_forest[0].copy()
    new_tree.ID = max(example_ids) + 1

    example_forest.append(new_tree)

    assert example_forest.ids()[-1] == new_tree.ID
    assert example_forest.by_id(new_tree.ID) is new_tree


def test_append_duplicate_id_raises(example_forest):
    duplicate = example_forest[0].copy()

    with pytest.raises(ValueError, match="Duplicate Tree ID"):
        example_forest.append(duplicate)


def test_extend_adds_trees_in_order(example_forest):
    base_id = max(example_ids)
    new_trees = []
    for offset in (1, 2, 3):
        tree = example_forest[0].copy()
        tree.ID = base_id + offset
        new_trees.append(tree)

    original_ids = example_forest.ids()
    example_forest.extend(new_trees)

    assert example_forest.ids() == original_ids + [tree.ID for tree in new_trees]
    assert example_forest.by_id(new_trees[1].ID) is new_trees[1]


def test_extend_accepts_generator(example_forest):
    base_id = max(example_ids)

    def tree_generator():
        for offset in (10, 11):
            tree = example_forest[0].copy()
            tree.ID = base_id + offset
            yield tree

    original_len = len(example_forest)
    example_forest.extend(tree_generator())

    assert len(example_forest) == original_len + 2
    assert example_forest.ids()[-2:] == [base_id + 10, base_id + 11]


def test_extend_empty_iterable_is_noop(example_forest):
    original_ids = example_forest.ids()

    example_forest.extend([])

    assert example_forest.ids() == original_ids


def test_extend_duplicate_id_raises(example_forest):
    duplicate = example_forest[0].copy()
    new_tree = example_forest[1].copy()
    new_tree.ID = max(example_ids) + 1

    with pytest.raises(ValueError, match="Duplicate Tree ID"):
        example_forest.extend([new_tree, duplicate])


def test_insert_updates_order_and_index(example_forest):
    inserted = example_forest[1].copy()
    inserted.ID = max(example_ids) + 1
    original_ids = example_forest.ids()

    example_forest.insert(0, inserted)
    assert example_forest.ids()[0] == inserted.ID
    assert example_forest.ids()[1:] == original_ids

    tail = example_forest[-1].copy()
    tail.ID = max(example_ids) + 2
    example_forest.insert(-1, tail)
    assert example_forest.ids()[-2] == tail.ID
    assert example_forest.by_id(tail.ID) is tail


def test_insert_duplicate_id_raises(example_forest):
    duplicate = example_forest[0].copy()

    with pytest.raises(ValueError, match="Duplicate Tree ID"):
        example_forest.insert(0, duplicate)


def test_remove_by_tree_object(example_forest):
    target = example_forest[1]
    target_id = target.ID
    remaining_ids = [tree_id for tree_id in example_forest.ids() if tree_id != target_id]

    example_forest.remove(target)

    assert target_id not in example_forest.ids()
    assert example_forest.ids() == remaining_ids


def test_remove_missing_tree_raises(example_forest):
    outsider = example_forest[0].copy()
    outsider.ID = max(example_ids) + 99

    with pytest.raises(ValueError, match="is not in this forest"):
        example_forest.remove(outsider)


def test_remove_id(example_forest):
    target_id = example_forest.ids()[2]
    remaining_ids = [tree_id for tree_id in example_forest.ids() if tree_id != target_id]

    example_forest.remove_id(target_id)

    assert target_id not in example_forest.ids()
    assert example_forest.ids() == remaining_ids


def test_remove_id_iterable(example_forest):
    remove_ids = example_forest.ids()[1:3]
    remaining_ids = [tree_id for tree_id in example_forest.ids() if tree_id not in remove_ids]

    example_forest.remove_id(remove_ids)

    assert example_forest.ids() == remaining_ids


def test_remove_id_iterable_dedupes(example_forest):
    target_id = example_forest.ids()[0]
    remaining_ids = example_forest.ids()[1:]

    example_forest.remove_id([target_id, target_id])

    assert example_forest.ids() == remaining_ids


def test_remove_id_missing_raises(example_forest):
    missing_id = max(example_ids) + 99

    with pytest.raises(KeyError, match=f"No tree with ID {missing_id}"):
        example_forest.remove_id(missing_id)


def test_remove_id_iterable_missing_raises(example_forest):
    missing_id = max(example_ids) + 99
    ids_before = example_forest.ids()

    with pytest.raises(KeyError, match=f"No tree with ID {missing_id}"):
        example_forest.remove_id([example_forest.ids()[0], missing_id])

    assert example_forest.ids() == ids_before


def test_pop_default_returns_last_tree(example_forest):
    expected_id = example_forest.ids()[-1]
    remaining_ids = example_forest.ids()[:-1]

    popped = example_forest.pop()

    assert isinstance(popped, Tree)
    assert expected_id == popped.ID
    assert example_forest.ids() == remaining_ids


def test_pop_at_index(example_forest):
    target_id = example_forest.ids()[1]
    remaining_ids = [tree_id for i, tree_id in enumerate(example_forest.ids()) if i != 1]

    popped = example_forest.pop(1)

    assert target_id == popped.ID
    assert example_forest.ids() == remaining_ids


def test_clear_empties_forest(example_forest):
    example_forest.clear()

    assert len(example_forest) == 0
    assert example_forest.ids() == []

    with pytest.raises(KeyError):
        example_forest.by_id(example_ids[0])


def test_copy_returns_shallow_container(example_forest):
    copied = example_forest.copy()

    assert copied is not example_forest
    assert copied.ids() == example_forest.ids()
    assert copied[0] is example_forest[0]


def test_contains_by_id_and_tree(example_forest):
    first_id = example_forest.ids()[0]
    first_tree = example_forest[0]
    outsider = first_tree.copy()
    outsider.ID = max(example_ids) + 99

    assert first_id in example_forest
    assert first_tree in example_forest
    assert outsider not in example_forest
    assert max(example_ids) + 99 not in example_forest
