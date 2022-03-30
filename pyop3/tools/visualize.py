import functools

import graphviz
import pyop3


def visualize(expr, *, name="expression", view=False, **kwargs):
    """Render loop expression as a DAG and write to a file.

    Parameters
    ----------
    expr : pyop3.Expression
        The loop expression.
    name : str, optional
        The name of DAG (and the save file).
    view : bool, optional
        Should the rendered result be opened with the default application?
    **kwargs : dict, optional
        Extra keyword arguments passed to the `graphviz.Digraph` constructor.
    """
    dag = graphviz.Digraph(name, **kwargs)
    _visualize(expr, dag)
    dag.render(quiet_view=view)


@functools.singledispatch
def _visualize(expr: pyop3.Expression, dag: graphviz.Digraph):
    raise AssertionError


@_visualize.register
def _(expr: pyop3.Loop, dag: graphviz.Digraph):
    label = str(expr)
    dag.node(label)
    for stmt in expr.statements:
        child_label = _visualize(stmt, dag)
        dag.edge(label, child_label)
    return label


@_visualize.register(pyop3.Restrict)
@_visualize.register(pyop3.FunctionCall)
def _(expr, dag: graphviz.Digraph):
    label = str(expr)
    dag.node(label)
    return label
