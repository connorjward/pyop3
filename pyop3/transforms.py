from __future__ import annotations


def split_loop(loop: Loop, path, tile_size: int) -> Loop:
    orig_loop_index = loop.index

    # I think I need to transform the index expressions of the iterset?
    # or get a new iterset? let's try that
    # It will not work because then the target path would change and the
    # data structures would not know what to do.

    orig_index_exprs = orig_loop_index.index_exprs
    breakpoint()
    # new_index_exprs

    new_loop_index = orig_loop_index.copy(index_exprs=new_index_exprs)
    return loop.copy(index=new_loop_index)
