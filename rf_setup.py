# rf_setup.py
from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def make_synthetic_data(
    seed: int = 7,
    n_samples: int = 8000,
    n_features: int = 6,
    n_informative: int = 5,
    n_redundant: int = 1,
    class_sep: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary dataset and split into train/val/test.

    Returns:
        X_tr, y_tr, X_val, y_val, X_te, y_te, lo, hi
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        class_sep=class_sep,
        random_state=seed,
    )

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed)

    lo = X_tr.min(axis=0)
    hi = X_tr.max(axis=0)

    return X_tr, y_tr, X_val, y_val, X_te, y_te, lo, hi


def train_big_forest(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 7,
    n_estimators: int = 60,
    max_depth: int = 8,
    min_samples_leaf: int = 2,
) -> RandomForestClassifier:
    """Train the large reference forest."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    return rf


def prune_forest_by_val_greedy(
    rf: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_trees: int,
) -> Tuple[RandomForestClassifier, List[int]]:
    """
    Greedy pruning: repeatedly drop the tree whose removal least hurts
    validation accuracy until only `target_trees` remain.

    Returns:
        (pruned_forest, kept_indices)
    """
    n_trees = len(rf.estimators_)
    if target_trees >= n_trees:
        # Nothing to do; keep everything.
        return deepcopy(rf), list(range(n_trees))

    keep = list(range(n_trees))

    def forest_pred_on(indices: List[int]) -> np.ndarray:
        estims = [rf.estimators_[i] for i in indices]
        proba = sum(e.predict_proba(X_val) for e in estims) / len(estims)
        return (proba[:, 1] >= 0.5).astype(int)

    # baseline accuracy (not used in the loop but useful for debugging if needed)
    _ = accuracy_score(y_val, forest_pred_on(keep))

    while len(keep) > target_trees:
        best_drop: int | None = None
        best_acc: float = -1.0
        for i in keep:
            cand = [j for j in keep if j != i]
            acc = accuracy_score(y_val, forest_pred_on(cand))
            if acc >= best_acc:
                best_acc, best_drop = acc, i
        keep.remove(best_drop)

    pruned = deepcopy(rf)
    pruned.estimators_ = [rf.estimators_[i] for i in keep]
    pruned.n_estimators = len(pruned.estimators_)
    return pruned, keep
