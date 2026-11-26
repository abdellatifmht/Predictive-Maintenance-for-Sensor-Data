# Documentation

From the repository root, install the docs extra and build with the project configuration:

```sh
uv sync --locked --extra docs
uv run mkdocs build --strict -f docs/mkdocs.yml
uv run mkdocs serve -f docs/mkdocs.yml
```

Edit the pages under `docs/docs/`. The completion audit is in `docs/project-audit.md`.
