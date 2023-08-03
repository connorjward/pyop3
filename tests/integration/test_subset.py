def test_scalar_copy_of_subset(scalar_copy_kernel):
    m, n = 6, 4
    sdata = np.asarray([2, 3, 5, 0], dtype=IntType)
    untouched = [1, 4]

    axes = AxisTree(Axis([AxisComponent(m, "cpt0")], "ax0"))
    dat0 = MultiArray(axes, name="dat0", data=np.arange(axes.size, dtype=ScalarType))
    dat1 = MultiArray(axes, name="dat1", dtype=dat0.dtype)

    # a subset is really a map from a small set into a larger one
    saxes = AxisTree(
        Axis([AxisComponent(n, "scpt0")], "sax0", id="root"), {"root": Axis(1)}
    )
    subset = MultiArray(saxes, name="subset0", data=sdata)

    p = (Index(TabulatedMap("sax0", "scpt0", "ax0", "cpt0", arity=1, data=subset)),)
    do_loop(p, scalar_copy_kernel(dat0[p], dat1[p]))

    assert np.allclose(dat1.data[sdata], dat0.data[sdata])
    assert np.allclose(dat1.data[untouched], 0)
