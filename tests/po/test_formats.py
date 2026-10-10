"""Pins format identity: adding axes must not change ids recorded by earlier sweeps."""

from datadec.po.formats import AXES, CANONICAL, EXTENDED_AXES, format_id, normalize, validate


def test_canonical_id_is_stable():
    assert format_id(CANONICAL) == "daa93775"


def test_extended_axis_absent_or_canonical_gives_same_id():
    old = {axis: CANONICAL[axis] for axis in AXES if axis not in EXTENDED_AXES}
    assert format_id(old) == format_id(CANONICAL) == "daa93775"
    assert normalize(old) == CANONICAL
    validate(old)


def test_extended_axis_non_canonical_changes_id():
    fmt = dict(CANONICAL, choice_text_separator="tab")
    assert format_id(fmt) != "daa93775"
