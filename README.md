# Embedding Geometry

This repository contains the exact code used to generate the figures and results for the blog post:

**https://www.testingbranch.com/Z3-and-model-equivalence**

It is a **snapshot**, not a maintained library.  
All results are reproducible with the included environment and scripts.


## Environment

This project uses uv for environment management.

To set up:

```bash
uv sync
uv run main.py
```

What this does:

- generates a synthetic binary classification dataset
- trains a reference Random Forest
- greedily prunes it down to a smaller forest with similar validation accuracy
- encodes both forests into Z3 and:
  - searches the training-domain box for exact label disagreements (counterexamples)
  - searches for large-margin disagreements in vote probabilities
- estimates empirical disagreement frequency via random sampling
- visualizes disagreement regions and the contribution of removed trees
 