"""Metrics and cluster-aware statistics for the Right Pause multi-turn study.

Turns are nested in cases (conversations), and cases are repeated over
multiple passes, so the CASE is the unit of inference for every between-config
comparison. See `metrics.py` (per-turn/per-case metric definitions) and
`stats.py` (case-cluster bootstrap, sign-flip permutation, Holm correction,
Wilson intervals, and the GLMM odds-ratio estimator).
"""
