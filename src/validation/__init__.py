"""Validation strategies for the WiDS wildfire pipeline.

Submodules:
- ``repeated_cv``: repeated stratified K-fold for variance-reduced CV.
- ``adversarial``: train-vs-test covariate-shift detection.
- ``nested_cv``: nested CV helper for honest tuning.
"""

from __future__ import annotations
