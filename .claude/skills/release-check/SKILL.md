---
name: release-check
description: Pre-release validation — version, changelog, tests, tag readiness
disable-model-invocation: true
---

# Release Check

Validate the project is ready for a PyPI release. Do NOT skip any step.

## Steps

1. **Version check** — read version from `pyproject.toml` and confirm it has been bumped from the last git tag:
   ```bash
   grep '^version' pyproject.toml
   git tag --sort=-v:refname | head -5
   ```

2. **Changelog check** — confirm `CHANGELOG.md` has an entry for the new version (not just "Unreleased")

3. **Run full test suite**:
   ```bash
   .venv/bin/pytest tests/ -v --tb=short
   ```
   All tests must pass.

4. **Lint check**:
   ```bash
   .venv/bin/ruff check src/ tests/
   ```

5. **Type check**:
   ```bash
   .venv/bin/mypy src/omni_proof/ --ignore-missing-imports
   ```

6. **Build check** — confirm the package builds cleanly:
   ```bash
   .venv/bin/python -m build
   ```

7. **Branch check** — confirm we are on `main` (release workflow rejects tags on other branches):
   ```bash
   git branch --show-current
   ```

8. **Clean working tree** — no uncommitted changes:
   ```bash
   git status --porcelain
   ```

9. **Report results**:
   ```
   | Check              | Status | Detail |
   |--------------------|--------|--------|
   | Version bumped     | PASS/FAIL | current vs last tag |
   | Changelog updated  | PASS/FAIL | |
   | Tests pass         | PASS/FAIL | X/Y passed |
   | Lint clean         | PASS/FAIL | |
   | Types clean        | PASS/FAIL | |
   | Build succeeds     | PASS/FAIL | |
   | On main branch     | PASS/FAIL | |
   | Clean working tree | PASS/FAIL | |
   ```

   If all pass, output the release commands:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
