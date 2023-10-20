def preprocess_t_unit_for_gpu(t_unit):
    # {{{ inline all kernels in t_unit

    kernels_to_inline = {
        name
        for name, clbl in t_unit.callables_table.items()
        if isinstance(clbl, lp.CallableKernel)
    }

    for knl_name in kernels_to_inline:
        t_unit = lp.inline_callable_kernel(t_unit, knl_name)

    # }}}

    kernel = t_unit.default_entrypoint

    # changing the address space of temps
    def _change_aspace_tvs(tv):
        if tv.read_only:
            assert tv.initializer is not None
            return tv.copy(address_space=lp.AddressSpace.GLOBAL)
        else:
            return tv.copy(address_space=lp.AddressSpace.PRIVATE)

    new_tvs = {
        tv_name: _change_aspace_tvs(tv)
        for tv_name, tv in kernel.temporary_variables.items()
    }
    kernel = kernel.copy(temporary_variables=new_tvs)

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        import pymbolic

        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, pymbolic.primitives.Subscript):
                assignee_name = insn.assignee.aggregate.name
            else:
                assert isinstance(insn.assignee, pymbolic.primitives.Variable)
                assignee_name = insn.assignee.name

            if assignee_name in kernel.arg_dict:
                return assignee_name in insn.read_dependency_names()
        return False

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if insn_needs_atomic(insn):
            atomicity = (lp.AtomicUpdate(insn.assignee.aggregate.name),)
            insn = insn.copy(atomicity=atomicity)
            args_marked_for_atomic |= set([insn.assignee.aggregate.name])

        new_insns.append(insn)

    # label args as atomic
    new_args = []
    for arg in kernel.args:
        if arg.name in args_marked_for_atomic:
            new_args.append(arg.copy(for_atomic=True))
        else:
            new_args.append(arg)

    kernel = kernel.copy(instructions=new_insns, args=new_args)

    return t_unit.with_kernel(kernel)


def _make_tv_array_arg(tv):
    assert tv.address_space != lp.AddressSpace.PRIVATE
    arg = lp.ArrayArg(
        name=tv.name,
        dtype=tv.dtype,
        shape=tv.shape,
        dim_tags=tv.dim_tags,
        offset=tv.offset,
        dim_names=tv.dim_names,
        order=tv.order,
        alignment=tv.alignment,
        address_space=tv.address_space,
        is_output=not tv.read_only,
        is_input=tv.read_only,
    )
    return arg


def split_n_across_workgroups(kernel, workgroup_size):
    """
    Returns a transformed version of *kernel* with the workload in the loop
    with induction variable 'n' distributed across work-groups of size
    *workgroup_size* and each work-item in the work-group performing the work
    of a single iteration of 'n'.
    """

    kernel = lp.assume(kernel, "start < end")
    kernel = lp.split_iname(
        kernel, "n", workgroup_size, outer_tag="g.0", inner_tag="l.0"
    )

    # {{{ making consts as globals: necessary to make the strategy emit valid
    # kernels for all forms

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [
        tv.initializer.flatten()
        for tv in old_temps.values()
        if tv.initializer is not None
    ]

    new_temps = {tv.name: tv for tv in old_temps.values() if tv.initializer is None}
    kernel = kernel.copy(
        args=kernel.args
        + [
            _make_tv_array_arg(tv)
            for tv in old_temps.values()
            if tv.initializer is not None
        ],
        temporary_variables=new_temps,
    )

    # }}}

    return kernel, args_to_make_global
