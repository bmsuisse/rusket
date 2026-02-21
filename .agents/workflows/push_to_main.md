---
description: Push to main and watch the GitHub Actions deployment via CLI
---

// turbo-all

1. Ensure you are on the `main` branch (or merge/push to it):
```bash
git push origin main
```

2. Wait a few seconds for the run to be registered, then get the latest run ID:
```bash
gh run list --branch main --limit 1
```

3. Watch the latest run in real-time until it completes (exits with non-zero on failure):
```bash
gh run watch $(gh run list --branch main --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
```

If the run fails, inspect the logs:
```bash
gh run view $(gh run list --branch main --limit 1 --json databaseId -q '.[0].databaseId') --log-failed
```
