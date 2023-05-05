import pytest

from pyop3.tree import *


def test_parent_works_with_nodes_and_ids():
    tree = Tree()
    a = Node("a")
    b = Node("b")
    tree.root = a
    tree.add_node(b, parent=a)

    assert tree.parent(b) == a
    assert tree.parent("b") == a


def test_children_works_with_nodes_and_ids():
    tree = Tree()
    a = Node("a")
    b = Node("b")
    tree.root = a
    tree.add_node(b, parent=a)

    assert tree.children(a) == (b,)
    assert tree.children("a") == (b,)


def test_tree_root_has_no_parent():
    tree = Tree()
    a = Node("a")
    tree.root = a
    assert tree.parent(a) is None


def test_tree_is_empty():
    tree = Tree()
    assert tree.is_empty
    tree.root = Node("a")
    assert not tree.is_empty


def test_can_set_root_multiple_times():
    tree = Tree()
    tree.root = Node("a")
    assert tree.root.id == "a"
    tree.root = Node("b")
    assert tree.root.id == "b"


def test_cannot_add_another_root():
    tree = Tree()
    tree.root = Node("a")
    with pytest.raises(ValueError):
        tree.add_node(Node("b"))


def test_add_node():
    tree = Tree()
    a = Node("a")
    b = Node("b")
    tree.root = a
    tree.add_node(b, parent=a)

    assert tree.children(a) == (b,)
    assert tree.parent(b) == a
    assert tree.children(b) == ()


@pytest.mark.parametrize("bulk", [True, False])
def test_add_multiple_children(bulk):
    tree = Tree()
    a = Node("a")
    b = Node("b")
    c = Node("c")

    tree.root = a
    if bulk:
        tree.add_node(b, parent=a)
        tree.add_node(c, parent=a)
    else:
        tree.add_nodes([b, c], parent=a)

    assert tree.children(a) == (b, c)
    assert tree.parent(b) == a
    assert tree.parent(c) == a
    assert tree.children(b) == ()
    assert tree.children(c) == ()
