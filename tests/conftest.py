"""Pytest configuration."""

import os

# scikit-learn's array API dispatch requires scipy's array API support, which is
# read when scipy is first imported.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
