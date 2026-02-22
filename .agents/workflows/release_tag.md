---
description: Format code, bump version, create a tag, and push a new release
---

// turbo-all

1. Format all code to prevent CI check failures:
```bash
uv run ruff format
```

2. Run linter and type checker just to be safe:
```bash
uv run ruff check
uv run basedpyright
```

3. Ensure all tests still pass:
```bash
uv run pytest tests/ -q -x --ignore=tests/test_benchmark.py
```

4. Create the new tag (ensure you have bumped the version in `pyproject.toml` and committed the changes first):
```bash
# Replace vX.Y.Z with your new version
export NEW_VERSION="v0.1.28"
git tag $NEW_VERSION
```

5. Push the tag to kick off the release workflow:
```bash
git push origin $NEW_VERSION
```

6. Watch the release pipeline:
```bash
gh run watch $(gh run list --branch $NEW_VERSION --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
```
