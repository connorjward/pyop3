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


@pytest.fixture
def treeA():
    tree = Tree()
    a = Node("a")
    b = Node("b")
    c = Node("c")
    d = Node("d")
    e = Node("e")
    f = Node("f")

    tree.root = a
    tree.add_nodes([b, c], parent=a)
    tree.add_nodes([d, e], parent=b)
    tree.add_node(f, parent=c)
    return tree


def test_tree_str(treeA):
    assert str(treeA) == """\
Node(id='a')
├──➤ Node(id='b')
│    ├──➤ Node(id='d')
│    └──➤ Node(id='e')
└──➤ Node(id='c')
     └──➤ Node(id='f')"""


def test_tree_depth():
    tree = Tree()
    assert tree.depth == 0
    tree.root = Node("a")
    assert tree.depth == 1
    tree.add_node(Node("b"), "a")
    assert tree.depth == 2
    tree.add_node(Node("c"), "a")
    assert tree.depth == 2


def test_tree_copy(treeA):
    treeB = treeA.copy()
    assert treeA.depth == treeB.depth == 3
    assert str(treeA) == str(treeB)

    treeA.add_node(Node("g"), "e")
    assert treeA.depth == 4
    assert treeB.depth == 3
