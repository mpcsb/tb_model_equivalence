# z3_rf_viz.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ---------- path & explanation helpers ----------

def tree_leaf_and_path(dt, x: np.ndarray):
    """
    Returns (leaf_class, path_node_indices) for a DecisionTreeClassifier and 1×d x.
    """
    t = dt.tree_
    node = 0
    path = [0]
    while t.children_left[node] != t.children_right[node]:
        f = t.feature[node]
        thr = t.threshold[node]
        if x[0, f] <= thr:
            node = t.children_left[node]
        else:
            node = t.children_right[node]
        path.append(node)

    cls = int(np.argmax(t.value[node][0]))
    return cls, path


def pretty_path(dt, path, feature_names: Optional[Sequence[str]] = None, x: Optional[np.ndarray] = None) -> str:
    """Human-readable path constraints."""
    t = dt.tree_
    out = []
    for i in range(len(path) - 1):
        n = path[i]
        f = t.feature[n]
        thr = t.threshold[n]
        left = t.children_left[n]
        right = t.children_right[n]

        name = feature_names[f] if feature_names is not None else f"x[{f}]"
        went_left = (path[i + 1] == left)
        cond = f"{name} <= {thr:.6g}" if went_left else f"{name} > {thr:.6g}"
        if x is not None:
            cond += f"  (x={x[0, f]:.6g})"
        out.append(cond)
    return " ∧ ".join(out)


def trace_forest_disagreement(
    x_cex: np.ndarray,
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Diagnose how big vs pruned differ on a single x_cex.

    Prints:
      - forest-level votes
      - per-tree differences
      - key removed trees and their paths
      - greedy "coalition" of trees that flip BIG's decision
    """
    xb = x_cex.reshape(1, -1)

    # Forest-level stats using per-tree hard predictions
    big_prob = np.mean([t.predict(xb)[0] for t in big.estimators_])
    prun_prob = np.mean([t.predict(xb)[0] for t in pruned.estimators_])
    print(f"[forest] big:   prob1={big_prob:.6f}, pred={int(big_prob >= 0.5)}")
    print(f"[forest] pruned: prob1={prun_prob:.6f}, pred={int(prun_prob >= 0.5)}")

    big_votes = np.array([t.predict(xb)[0] for t in big.estimators_], dtype=int)
    prun_votes = np.array([t.predict(xb)[0] for t in pruned.estimators_], dtype=int)

    kept_ids = {id(t) for t in pruned.estimators_}
    kept_idx = [i for i, t in enumerate(big.estimators_) if id(t) in kept_ids]

    # Theoretically identical trees; this only becomes interesting if you later
    # distill/simplify pruned's trees.
    diff_idx = [
        i
        for i in kept_idx
        if big.estimators_[i].predict(xb)[0]
        != pruned.estimators_[kept_idx.index(i)].predict(xb)[0]
    ]
    print(f"[kept trees] disagreeing predictions: {len(diff_idx)} / {len(kept_idx)}")

    # Trees that were removed
    removed_idx = [i for i in range(len(big.estimators_)) if i not in kept_idx]
    removed_vote = big_votes[removed_idx].mean() if removed_idx else np.nan
    print(f"[removed trees] count={len(removed_idx)}, mean vote={removed_vote}")

    # Show a few influential removed trees
    if removed_idx:
        base = big_votes.mean()
        effects = []
        for i in removed_idx:
            new_mean = (big_votes.sum() - big_votes[i]) / (len(big_votes) - 1)
            effects.append((abs(base - new_mean), i))
        effects.sort(reverse=True)
        show = [idx for _, idx in effects[: min(top_k, len(effects))]]
        print(f"[removed trees] top contributors (by single-tree effect): {show}")
        for i in show:
            leaf, path = tree_leaf_and_path(big.estimators_[i], xb)
            print(
                f"  - tree#{i} vote={leaf} | path: "
                f"{pretty_path(big.estimators_[i], path, feature_names, xb)}"
            )

    # Minimal-ish coalition that flips BIG at x_cex (greedy over trees)
    target_label = int(big_prob >= 0.5)
    votes = big_votes.astype(float)
    idxs = list(range(len(votes)))
    current_mean = votes.mean()
    coalition: List[int] = []

    while int(current_mean >= 0.5) == target_label and idxs:
        best_drop: Optional[int] = None
        best_mean: Optional[float] = None
        for i in idxs:
            if len(idxs) > 1:
                new_mean = (votes.sum() - votes[i]) / (len(idxs) - 1)
            else:
                # Degenerate: dropping the last tree flips to 1 - vote
                new_mean = 1.0 - votes[i]
            if best_mean is None or abs(new_mean - 0.5) < abs(best_mean - 0.5):
                best_drop, best_mean = i, new_mean
        coalition.append(best_drop)
        idxs.remove(best_drop)
        current_mean = (votes.sum() - votes[coalition].sum()) / (len(votes) - len(coalition))

    print(f"[minimal-ish coalition] #trees to flip BIG at x_cex (greedy): {len(coalition)}")

    return {
        "big_prob": float(big_prob),
        "pruned_prob": float(prun_prob),
        "removed_idx": removed_idx,
        "diff_idx": diff_idx,
        "coalition": coalition,
    }


# ---------- smoothing (scipy optional) ----------

def _smooth(Z: np.ndarray, sigma: Optional[float]):
    if sigma is None or sigma <= 0:
        return Z
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        return gaussian_filter(Z, sigma=float(sigma))
    except Exception:
        # tiny separable box blur fallback
        k = max(1, int(round(sigma * 2)))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        Zp = np.pad(Z, pad, mode="edge")
        W = np.ones((k, k), dtype=float) / (k * k)
        out = np.zeros_like(Z)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = (Zp[i : i + k, j : j + k] * W).sum()
        return out


# ---------- core helpers for 2D slices ----------

def rf_vote_prob(estimators, X: np.ndarray) -> np.ndarray:
    """Mean per-tree P(class=1)."""
    return np.mean([t.predict_proba(X)[:, 1] for t in estimators], axis=0)


def _grid(lo, hi, center, i: int, j: int, n: int):
    xs = np.linspace(lo[i], hi[i], n)
    ys = np.linspace(lo[j], hi[j], n)
    XX, YY = np.meshgrid(xs, ys)
    base = np.tile(center, (n * n, 1))
    for r in range(n):
        base[r * n : (r + 1) * n, i] = XX[r]
        base[r * n : (r + 1) * n, j] = YY[r]
    return xs, ys, base


def best_dims_for_disagreement(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    lo: np.ndarray,
    hi: np.ndarray,
    samples: int = 4000,
    rng: int = 0,
):
    """Pick (i,j) where |Δ vote| has most structure (variance on a coarse grid)."""
    r = np.random.default_rng(rng)
    X = r.uniform(lo, hi, size=(samples, len(lo)))
    dv = np.abs(
        rf_vote_prob(big.estimators_, X) - rf_vote_prob(pruned.estimators_, X)
    )
    d = len(lo)
    bins = 16
    scores = {}
    for i in range(d):
        for j in range(i + 1, d):
            bi = np.clip(((X[:, i] - lo[i]) / (hi[i] - lo[i]) * bins).astype(int), 0, bins - 1)
            bj = np.clip(((X[:, j] - lo[j]) / (hi[j] - lo[j]) * bins).astype(int), 0, bins - 1)
            grid_sum = np.zeros((bins, bins))
            grid_cnt = np.zeros((bins, bins))
            for k in range(len(X)):
                grid_sum[bi[k], bj[k]] += dv[k]
                grid_cnt[bi[k], bj[k]] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                grid = grid_sum / grid_cnt
            scores[(i, j)] = np.nanvar(grid)
    return max(scores, key=scores.get)


def _imshow_copper(
    ZZ: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    cbar_label: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    if vmin is None or vmax is None:
        finite = ZZ[np.isfinite(ZZ)]
        if finite.size:
            vmin = np.nanpercentile(finite, 5)
            vmax = np.nanpercentile(finite, 95)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        ZZ,
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        aspect="equal",
        cmap="copper",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# ---------- visuals ----------

def plot_vote_diff_slice(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    center: Optional[np.ndarray] = None,
    dims: Optional[tuple[int, int]] = None,
    n: int = 220,
    pts: Optional[List[np.ndarray]] = None,
    eps: Optional[float] = None,
    smooth_sigma: float = 1.0,
    title: str = "|Δ vote probability| on 2D slice",
):
    """
    Heatmap of |P_big(1) - P_pruned(1)| on a 2D slice.
    """
    d = len(lo)
    if center is None:
        center = (lo + hi) / 2.0
    if dims is None:
        dims = best_dims_for_disagreement(big, pruned, lo, hi)
    i, j = dims
    xs, ys, base = _grid(lo, hi, center, i, j, n)

    vb = rf_vote_prob(big.estimators_, base)
    vp = rf_vote_prob(pruned.estimators_, base)
    ZZ = np.abs(vb - vp).reshape(n, n)

    ZZ = _smooth(ZZ, smooth_sigma)
    if eps is not None:
        ZZ = np.where(ZZ >= eps, ZZ, np.nan)

    _imshow_copper(
        ZZ,
        xs,
        ys,
        title,
        cbar_label="|Δ vote prob|",
        xlabel=f"x[{i}]",
        ylabel=f"x[{j}]",
    )

    # overlay markers: user can scatter themselves if needed


def plot_label_disagreement_region(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    center: Optional[np.ndarray] = None,
    dims: Optional[tuple[int, int]] = None,
    n: int = 220,
    smooth_sigma: float = 0.0,
    title: str = "Label disagreement region",
):
    d = len(lo)
    if center is None:
        center = (lo + hi) / 2.0
    if dims is None:
        dims = best_dims_for_disagreement(big, pruned, lo, hi)
    i, j = dims
    xs, ys, base = _grid(lo, hi, center, i, j, n)

    yb = (rf_vote_prob(big.estimators_, base) >= 0.5).astype(int)
    yp = (rf_vote_prob(pruned.estimators_, base) >= 0.5).astype(int)
    D = (yb != yp).reshape(n, n).astype(float)
    D = _smooth(D, smooth_sigma)

    _imshow_copper(
        D,
        xs,
        ys,
        title,
        cbar_label="disagree=1 / agree=0",
        xlabel=f"x[{i}]",
        ylabel=f"x[{j}]",
    )


def plot_removed_tree_effects_at_x(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    x_cex: np.ndarray,
    top_k: int = 15,
    title: str = "Top removed trees at CE",
):
    xb = x_cex.reshape(1, -1)
    big_votes = np.array([t.predict_proba(xb)[:, 1][0] for t in big.estimators_], dtype=float)
    kept_ids = {id(t) for t in pruned.estimators_}
    removed = [i for i, t in enumerate(big.estimators_) if id(t) not in kept_ids]
    if not removed:
        print("No removed trees.")
        return

    base = big_votes.mean()
    eff = []
    for i in removed:
        new_mean = (big_votes.sum() - big_votes[i]) / (len(big_votes) - 1)
        eff.append((abs(base - new_mean), i))
    eff.sort(reverse=True)
    eff = eff[: min(top_k, len(eff))]
    deltas = [e[0] for e in eff]
    idxs = [e[1] for e in eff]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(eff)), deltas)
    plt.xticks(range(len(eff)), idxs, rotation=45)
    plt.ylabel("Δ avg vote if removed")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_conflict_where_big_confident(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    center: Optional[np.ndarray] = None,
    dims: Optional[tuple[int, int]] = None,
    n: int = 220,
    m: float = 0.2,
    title: str = "Pruned disagrees where BIG margin is high",
):
    d = len(lo)
    if center is None:
        center = (lo + hi) / 2.0
    if dims is None:
        dims = best_dims_for_disagreement(big, pruned, lo, hi)
    i, j = dims
    xs, ys, base = _grid(lo, hi, center, i, j, n)

    vb = rf_vote_prob(big.estimators_, base)
    yb = (vb >= 0.5).astype(int)
    yp = (rf_vote_prob(pruned.estimators_, base) >= 0.5).astype(int)
    confident = (vb >= 0.5 + m) | (vb <= 0.5 - m)
    conflict = (yb != yp) & confident
    Z = conflict.reshape(n, n).astype(float)

    _imshow_copper(
        Z,
        xs,
        ys,
        title,
        cbar_label=f"disagree & margin≥{m}",
        xlabel=f"x[{i}]",
        ylabel=f"x[{j}]",
    )
