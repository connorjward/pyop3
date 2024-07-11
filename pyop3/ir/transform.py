import loopy as lp


def add_likwid_markers(knl):
    """
    See https://github.com/RRZE-HPC/likwid/wiki/TutorialMarkerC
    """
    import pylikwid

    preambles = knl.preambles + (("99_likwid", "#include <likwid-marker.h>"),)

    marker_name = knl.name
    pylikwid.markerregisterregion(marker_name)

    start_insn = lp.CInstruction((), f"LIKWID_MARKER_START(\"{marker_name}\");", id="likwid_start")
    insns = (
        [start_insn]
        + [insn.copy(depends_on=insn.depends_on | {"likwid_start"}) for insn in knl.instructions]
    )
    stop_insn = lp.CInstruction((), f"LIKWID_MARKER_STOP(\"{marker_name}\");", id="likwid_stop", depends_on=frozenset(insn.id for insn in insns))
    insns.append(stop_insn)

    return knl.copy(preambles=preambles, instructions=insns)
