# Working agreements

The user wants incremental Git history, with small local commits and branches
that are merged when their work is ready.

- Make one meaningful change per commit. Use clear messages such as `feat:`,
  `fix:`, `test:`, `docs:`, or `chore:`. Do not create a commit for every line edit.
- Develop related changes on a short-lived `feat/`, `fix/`, or `docs/` branch.
  Check the current branch and worktree before creating or switching branches.
- Run checks appropriate to each change before committing. Documentation-only
  changes do not require retraining models.
- Commit completed authorized work locally as it is finished. Do not bundle
  unrelated user edits or discard their work. Stage specific files or hunks.
- Merge a completed, verified branch into `main` with a merge commit, unless
  the user asks to leave it open or use a pull request. Preserve existing history;
  do not squash, amend, or rewrite commits without a reason and authorization.
- Pushing/publishing is a separate step. Do not push merely because local
  commits are ready; follow the user's instruction about when and where to push.
- Keep virtual environments, downloaded runtimes, raw downloaded datasets,
  generated artifacts, caches, local tool state, and secrets out of Git.
- Preserve dataset provenance, dependency versions, and reproducible commands
  in tracked code and documentation. The current project is `ml-projects/credit-default`.
- Use the workspace Python: `.venv/Scripts/python.exe`. On a new checkout,
  follow the setup steps in the project README.
