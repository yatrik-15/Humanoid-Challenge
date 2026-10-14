"""Which image pairs structure-from-motion matches (kept free of heavy imports)."""


def windowed_pairs(names: list[str], window: int, cross_stride: int) -> list[tuple[str, str]]:
    """Image pairs for matching without trying all of them.

    Within each recording, every frame is paired with the next `window` frames (frames
    are taken in walking order, so neighbours overlap most). Across recordings, every A
    frame is paired with every `cross_stride`-th B frame, staggered by the A index so each
    B frame is paired with about 1 / cross_stride of the A frames. Names are A_*/B_*.
    """
    by_epoch = {e: sorted(n for n in names if n.startswith(f"{e}_")) for e in ("A", "B")}
    pairs = []
    for seq in by_epoch.values():
        for i, a in enumerate(seq):
            pairs += [(a, b) for b in seq[i + 1 : i + 1 + window]]
    for i, a in enumerate(by_epoch["A"]):
        pairs += [(a, b) for j, b in enumerate(by_epoch["B"]) if (i + j) % cross_stride == 0]
    return pairs
