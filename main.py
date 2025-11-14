import numpy as np

from rf_setup import make_synthetic_data, prune_forest_by_val_greedy, train_big_forest
from z3_rf_equiv import find_disagreement_cex, z3_counterexample_margin
from z3_rf_viz import (
    plot_conflict_where_big_confident,
    plot_label_disagreement_region,
    plot_removed_tree_effects_at_x,
    plot_vote_diff_slice,
    trace_forest_disagreement,
)




# 1) Data + model
X_tr, y_tr, X_val, y_val, X_te, y_te, lo, hi = make_synthetic_data(seed=7)
big = train_big_forest(X_tr, y_tr, seed=7)

print("Big RF — train/val acc:",
        (big.score(X_tr, y_tr), big.score(X_val, y_val)))

pruned, kept_idx = prune_forest_by_val_greedy(big, X_val, y_val, target_trees=18)
print("Pruned RF — trees kept:", len(kept_idx))
print("Big vs Pruned val acc:", big.score(X_val, y_val), pruned.score(X_val, y_val))

# 2) Z3: counterexamples
cex = find_disagreement_cex(big, pruned, lo, hi)
if cex is not None:
    x_cex = cex.reshape(1, -1)
    print("❌ Counterexample found (big vs pruned differ):", cex)
    print("big:   ", big.predict(x_cex), big.predict_proba(x_cex)[0])
    print("pruned:", pruned.predict(x_cex), pruned.predict_proba(x_cex)[0])
else:
    print("✅ Equivalent on the bounded box [lo, hi] (no Z3 disagreement).")

cex_margin = z3_counterexample_margin(big, pruned, lo, hi, eps=0.25)
print("margin-CE:", cex_margin)
if cex_margin is not None:
    xb = cex_margin.reshape(1, -1)
    print(
        "big/pruned probs at margin-CE:",
        np.mean([t.predict(xb)[0] for t in big.estimators_]),
        np.mean([t.predict(xb)[0] for t in pruned.estimators_]),
    )

# 3) Empirical disagreement rate on random samples
rng = np.random.default_rng(0)
Z = rng.uniform(lo, hi, size=(20_000, len(lo)))
emp = (big.predict(Z) != pruned.predict(Z)).mean()
print("sampled disagree rate:", emp)

# 4) Diagnostics + viz (only if a CE exists)
pts = []
if cex is not None:
    pts.append(cex)
if cex_margin is not None:
    pts.append(cex_margin)

if cex is not None:
    _ = trace_forest_disagreement(cex, big, pruned, feature_names=None, top_k=8)

if pts:
    # vote-diff slice
    plot_vote_diff_slice(
        big,
        pruned,
        lo,
        hi,
        center=(lo + hi) / 2,
        dims=None,
        n=300,
        pts=pts,
        eps=0.06,
    )

    # label disagreement
    plot_label_disagreement_region(
        big,
        pruned,
        lo,
        hi,
        center=(lo + hi) / 2,
        dims=None,
        n=300,
    )

    # removed-tree impact at CE
    if cex is not None:
        plot_removed_tree_effects_at_x(big, pruned, cex, top_k=12)

    # safety view: where big is confident
    plot_conflict_where_big_confident(
        big,
        pruned,
        lo,
        hi,
        center=(lo + hi) / 2,
        dims=None,
        n=300,
        m=0.05,
    )

    # Optional: slice through a particular CE and feature pair
    if cex is not None:
        plot_vote_diff_slice(
            big,
            pruned,
            lo,
            hi,
            center=cex,
            dims=(0, 3),
            n=350,
            pts=[cex],
            eps=0.05,
        )


