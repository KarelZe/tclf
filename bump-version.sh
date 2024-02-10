#!/bin/bash
# adapted from: https://github.com/pre-commit/pre-commit/issues/1825
set -e
if git diff --exit-code origin/main -- version; then
    bump-my-version bump --allow-dirty patch version --current-version `cat version`
fi
