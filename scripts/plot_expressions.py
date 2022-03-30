from pyop3 import arguments, domains, loops


kernel = loops.Function("kernel")
dat = arguments.Dat("mydat")
expr = loops.Loop(p := domains.FreePointSet("P").point_index, kernel(dat[p]))

loops.visualize(expr, view=True)
