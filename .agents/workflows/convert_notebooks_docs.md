---
description: Convert Jupyter notebooks to markdown for MkDocs
---

1. Convert all Jupyter notebooks in the `docs/notebooks/` directory to `.md` format.
// turbo
```bash
uv run jupyter nbconvert --to markdown docs/notebooks/*.ipynb
```

2. Optional: Clean up the generated files if they are no longer needed
```bash
# Verify the markdown files exist before removing notebooks, if desired
# rm docs/notebooks/*.ipynb
```

3. Build the MkDocs documentation to verify everything renders correctly.
// turbo
```bash
uv run mkdocs build
```
