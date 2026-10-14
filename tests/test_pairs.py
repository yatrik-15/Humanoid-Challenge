"""Match-pair selection for structure from motion."""

from src.pairs import windowed_pairs


def test_windowed_pairs_cover_neighbours_and_both_recordings():
    names = [f"A_{i:04d}.jpg" for i in range(1, 31)] + [f"B_{i:04d}.jpg" for i in range(1, 31)]
    pairs = windowed_pairs(names, window=5, cross_stride=3)
    s = {frozenset(p) for p in pairs}
    assert len(s) == len(pairs)  # no duplicates
    assert frozenset(("A_0001.jpg", "A_0006.jpg")) in s and frozenset(("A_0001.jpg", "A_0007.jpg")) not in s
    cross = [p for p in pairs if p[0][0] != p[1][0]]
    assert len(cross) == 30 * 10  # every A frame with a third of the B frames
    assert {b for _, b in cross} == {n for n in names if n.startswith("B_")}  # every B frame reached
