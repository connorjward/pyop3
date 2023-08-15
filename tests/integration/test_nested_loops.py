def test_transpose():
    # array0 = MultiArray(???)
    # array0 = MultiArray(???)

    do_loop(
        p := axis0.index(),
        loop(q := axis1.index(), scalar_copy_kernel(array0[p, q], array1[q, p])),
    )
