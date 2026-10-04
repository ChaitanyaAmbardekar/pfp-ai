"""Rank analysed photos."""


def rank(analyses):
    ok = sorted((a for a in analyses if a.ok),
                key=lambda a: (-a.overall, -a.sub.get("robustness", 0.0)))
    return ok + [a for a in analyses if not a.ok]


def top_n(ranked, n=3):
    return [a for a in ranked if a.ok][:n]


def close_call(ranked, threshold=3.0):
    """True if the top two photos are within `threshold` points."""
    ok = [a for a in ranked if a.ok]
    return len(ok) >= 2 and (ok[0].overall - ok[1].overall) < threshold