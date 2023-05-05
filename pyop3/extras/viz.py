import collections

import graphviz

from pyop3 import tlang


def plot_dag(expr: tlang.Expression, *, name="expression", view=False, **kwargs):
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

    if not isinstance(expr, collections.abc.Collection):
        expr = (expr,)

    for e in expr:
        _plot_dag(e, dag)

    dag.render(quiet_view=view)


def _plot_dag(expr: tlang.Expression, dag: graphviz.Digraph):
    label = str(expr)
    dag.node(label)
    for o in expr.children:
        dag.edge(label, _plot_dag(o, dag))
    return label
