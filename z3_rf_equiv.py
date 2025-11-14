# z3_rf_equiv.py
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from z3 import (
    And,
    If,
    Or,
    Real,
    RealVal,
    Solver,
    Sum,
    sat,
)


def encode_sklearn_tree_as_z3(tree, x_vars):
    """
    Encode a sklearn DecisionTreeClassifier into a Z3 expression
    returning a Real in {0, 1}, the predicted class at the leaf.
    """
    t = tree.tree_
    feat = t.feature
    thresh = t.threshold
    children_left = t.children_left
    children_right = t.children_right
    values = t.value  # shape [nodes, 1, n_classes]

    def node_expr(idx):
        if children_left[idx] == children_right[idx]:  # leaf
            counts = values[idx][0]
            pred_class = int(np.argmax(counts))
            return RealVal(pred_class)
        f = feat[idx]
        thr = thresh[idx]
        cond = (x_vars[f] <= thr)
        return If(cond, node_expr(children_left[idx]), node_expr(children_right[idx]))

    return node_expr(0)


def encode_forest_as_z3_avg_vote(
    rf: RandomForestClassifier,
    x_vars: Sequence,
):
    """
    Encode a RandomForestClassifier as the average vote (Real in [0,1]).
    """
    tree_exprs = [encode_sklearn_tree_as_z3(t, x_vars) for t in rf.estimators_]
    return Sum(*tree_exprs) / len(tree_exprs)


def make_box_constraints(x_vars, lo, hi):
    """Axis-aligned box: lo[i] <= x_i <= hi[i]."""
    cons = []
    for i, xi in enumerate(x_vars):
        cons.append(xi >= lo[i])
        cons.append(xi <= hi[i])
    return And(*cons)


def label_from_vote(expr):
    """Threshold at 0.5 → class in {0,1} as Real."""
    return If(expr >= RealVal(0.5), RealVal(1), RealVal(0))


def find_disagreement_cex(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    lo: np.ndarray,
    hi: np.ndarray,
    decimals: int = 10,
):
    """
    Find a counterexample x in [lo, hi] where the two forests disagree
    (0/1 labels differ). Returns np.ndarray or None.
    """
    d = len(lo)
    x = [Real(f"x{i}") for i in range(d)]

    big_out = encode_forest_as_z3_avg_vote(big, x)
    prun_out = encode_forest_as_z3_avg_vote(pruned, x)

    s = Solver()
    s.add(make_box_constraints(x, lo, hi))
    s.add(label_from_vote(big_out) != label_from_vote(prun_out))

    res = s.check()
    if res != sat:
        return None

    m = s.model()
    vals = []
    for xi in x:
        # as_decimal yields e.g. "0.123456?"; strip '?'
        v = float(m[xi].as_decimal(decimals).replace("?", ""))
        vals.append(v)
    return np.array(vals, dtype=float)


def z3_counterexample_margin(
    big: RandomForestClassifier,
    pruned: RandomForestClassifier,
    lo: np.ndarray,
    hi: np.ndarray,
    eps: float = 0.10,
    decimals: int = 20,
):
    """
    Find x where |P_big(1) - P_pruned(1)| >= eps on [lo, hi].

    Returns:
        np.ndarray or None
    """
    d = len(lo)
    x = [Real(f"x{i}") for i in range(d)]

    big_out = encode_forest_as_z3_avg_vote(big, x)
    prun_out = encode_forest_as_z3_avg_vote(pruned, x)

    s = Solver()
    s.add(make_box_constraints(x, lo, hi))
    s.add(
        Or(
            big_out - prun_out >= RealVal(eps),
            prun_out - big_out >= RealVal(eps),
        )
    )

    if s.check() != sat:
        return None

    m = s.model()
    vals = []
    for xi in x:
        v = float(m[xi].as_decimal(decimals).replace("?", ""))
        vals.append(v)
    return np.array(vals, dtype=float)
