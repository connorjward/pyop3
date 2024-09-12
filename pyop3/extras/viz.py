import collections
import numbers

import graphviz
import numpy as np
from matplotlib import cm
from pyrsistent import pmap

from pyop3.distarray import IndexedMultiArray, MultiArray
from pyop3.dtypes import IntType
from pyop3.utils import PrettyTuple, strict_int
from pyop3.axis import _axis_size

# def plot_dag(expr: tlang.Expression, *, name="expression", view=False, **kwargs):
#     """Render loop expression as a DAG and write to a file.
#
#     Parameters
#     ----------
#     expr : pyop3.Expression
#         The loop expression.
#     name : str, optional
#         The name of DAG (and the save file).
#     view : bool, optional
#         Should the rendered result be opened with the default application?
#     **kwargs : dict, optional
#         Extra keyword arguments passed to the `graphviz.Digraph` constructor.
#     """
#     dag = graphviz.Digraph(name, **kwargs)
#
#     if not isinstance(expr, collections.abc.Collection):
#         expr = (expr,)
#
#     for e in expr:
#         _plot_dag(e, dag)
#
#     dag.render(quiet_view=view)
#
#
# def _plot_dag(expr: tlang.Expression, dag: graphviz.Digraph):
#     label = str(expr)
#     dag.node(label)
#     for o in expr.children:
#         dag.edge(label, _plot_dag(o, dag))
#     return label


mybadcolormap = {
    1: ["white"],
    2: ["#0066ff", "#ff6666"],
    3: ["#0066ff", "#ff1a1a", "lightgreen"],
}


# TODO This is just the same tree visitor as we have for computing layouts
def _view_axis_tree(dag, axes, axis, indices=PrettyTuple(), show_permutation=False):
    npoints = 0
    for component in axis.components:
        if isinstance(component.count, MultiArray):
            npoints += strict_int(component.count.get_value(indices))
        else:
            assert isinstance(component.count, numbers.Integral)
            npoints += component.count

    permutation = (
        axis.permutation
        if axis.permutation is not None
        else np.arange(npoints, dtype=IntType)
    )

    point_to_component_id = np.empty(npoints, dtype=np.int8)
    point_to_component_num = np.empty(npoints, dtype=IntType)
    pos = 0
    for cidx, component in enumerate(axis.components):
        if isinstance(component.count, IndexedMultiArray):
            csize = strict_int(component.count.data.get_value(indices))
        elif isinstance(component.count, MultiArray):
            csize = strict_int(component.count.get_value(indices))
        else:
            assert isinstance(component.count, numbers.Integral)
            csize = component.count

        for i in range(csize):
            point = permutation[pos + i]
            point_to_component_id[point] = cidx
            point_to_component_num[point] = i
        pos += csize

    inverse_perm = np.empty_like(permutation)
    for i, p in enumerate(permutation):
        inverse_perm[p] = i

    cells = []
    for pt in range(npoints):
        component_id = point_to_component_id[pt]

        if show_permutation:
            component_num = inverse_perm[pt]
        else:
            component_num = point_to_component_num[pt]

        color = mybadcolormap[axis.degree][component_id]
        cells.append(f"<TD PORT='x{pt}' BGCOLOR='{color}'>{component_num}</TD>")

    label = (
        "<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0'><TR>"
        + "".join(cells)
        + "</TR></TABLE>>"
    )
    node_id = str(indices)
    dag.node(node_id, label, shape="plaintext")

    if indices:
        parent = f"{indices[:-1]}:x{indices[-1]}"
        dag.edge(parent, node_id, tailport=f"x{indices[-1]}:s", headport="n")

    for pt in range(npoints):
        component_id = point_to_component_id[pt]
        component = axis.components[component_id]
        if subaxis := axes.child(axis, component):
            _view_axis_tree(dag, axes, subaxis, indices | pt)


def view_axes(axes, name="axes", display=True, format="pdf"):
    dag = graphviz.Digraph(name, format=format)
    _view_axis_tree(dag, axes, axes.root)
    dag.render(quiet_view=display)
