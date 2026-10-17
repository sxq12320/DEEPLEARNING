"""GT-free fixed-budget crop selection. Image content stays RGB; no colour/shape veto."""

import random

import numpy as np


def select_tiles(candidates, windows, budget=2, policy="coverage", seed=0, score_min=0.25, small_area=1024):
    """Select exactly min(budget, available) original-pixel crops without GT.

    Greedy weighted coverage favours small *predicted* masks on the global640
    raster. Empty gains are filled with deterministic dispersed exploration.
    This fallback addresses the selector's blindness to absent global candidates,
    but does not guarantee missed-fruit recovery. AP is not the coverage objective.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if policy not in ("coverage", "random", "all", "global"):
        raise ValueError("policy must be coverage/random/all/global")
    n = len(windows)
    if policy == "global" or not n:
        return []
    if policy == "all":
        return list(range(n))
    count = min(budget, n)
    if policy == "random":
        return random.Random(seed).sample(range(n), count)
    retained = [c for c in candidates if c["score"] >= score_min and c["patch"].size
                and 0 < c["patch"].sum() < small_area]
    points = np.asarray([[(c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2]
                         for c in retained], dtype=float).reshape(-1, 2)
    weights = np.asarray([c["score"] for c in retained])
    covered = np.zeros(len(points), dtype=bool)
    chosen = []
    centers = np.asarray([[(w[0] + w[2]) / 2, (w[1] + w[3]) / 2] for w in windows], dtype=float)
    if len(points):
        membership = np.stack([(points[:, 0] >= w[0]) & (points[:, 0] < w[2])
                               & (points[:, 1] >= w[1]) & (points[:, 1] < w[3]) for w in windows])
        while len(chosen) < count:
            gains = (membership & ~covered) @ weights
            gains[chosen] = -1
            k = int(gains.argmax())
            if gains[k] <= 0:
                break
            chosen.append(k)
            covered |= membership[k]
    while len(chosen) < count:
        available = [i for i in range(n) if i not in chosen]
        if not chosen:
            mid = centers.mean(0)
            k = min(available, key=lambda i: (float(((centers[i] - mid) ** 2).sum()), i))
        else:
            distances = ((centers[:, None] - centers[chosen][None]) ** 2).sum(-1).min(1)
            k = max(available, key=lambda i: (float(distances[i]), -i))
        chosen.append(k)
    return chosen
