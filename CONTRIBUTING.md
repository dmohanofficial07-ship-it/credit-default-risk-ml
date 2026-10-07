# Small commits and branch merges

A **commit** records a local checkpoint. A **branch** groups related work.
A **merge** brings completed branch work into `main`. A **push** uploads local
history to a remote such as GitHub. You can make many local commits before
pushing; one push transfers their separate history.

Use one commit for each meaningful, reviewable step: for example, an additional
feature calculation, its focused checks, or a documentation improvement.
Each commit should leave the project understandable and usable at that stage.

## Workflow

Start by checking the working tree and branch:

```powershell
git status
git switch main
git switch -c feat/repayment-features
```

Make one focused change, inspect it, and run relevant checks. Stage the specific
files for that change, then inspect the staged patch before committing:

```powershell
git diff
git add ml-projects/credit-default/train.py
git diff --cached
git commit -m "feat: add repayment-history features"
```

That filename and message are examples; use the files and description matching
the actual change. Include directly relevant test updates in the same commit
when they are needed to verify the feature. Repeat for each completed step.

For the current credit-default workflow, after training, the existing checks run with:

```powershell
.\.venv\Scripts\python.exe .\ml-projects\credit-default\test_project.py
```

When the branch's work is complete and verified, merge it locally:

```powershell
git switch main
git merge --no-ff feat/repayment-features -m "merge: add repayment-history features"
git log --oneline --graph --decorate --all
```

Resolve conflicts by reviewing the intended result, then rerun relevant checks.
Do not discard other work to force a clean merge. Keep published history intact.

## What belongs in Git

Track source code, dependency pins, tests, documentation, and shared VS Code
settings/tasks. `.gitignore` excludes local Python/tool installations, downloaded
data, generated model artifacts, caches, local tool configuration, and `.env`
files. The code downloads data and produces artifacts again when needed.

## Publishing later

No remote is configured initially. Once a repository destination is chosen,
connect it using `git remote add origin <repository-url>`. Upload the committed
main history using `git push -u origin main` when publication is requested.
Neither command is needed to make local commits or merges.

Once collaborating through a remote, push a feature branch and open a pull
request before merging when that is the agreed review workflow.
