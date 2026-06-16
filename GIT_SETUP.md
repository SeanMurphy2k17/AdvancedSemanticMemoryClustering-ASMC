# Git setup fix (untracked folder)

If this `AdvancedSemanticMemoryClustering/` folder has no `.git` directory inside it,
it's not actually tracking this remote — it's just a plain folder of files that
happens to live inside the GhostEngine repo (which itself only holds a stale gitlink
pointer to an old commit here, with no `.gitmodules`). That means `git pull` won't
work and the folder will silently drift out of sync with this repo.

Check first:

```bash
cd AdvancedSemanticMemoryClustering
ls -la .git   # if this errors "No such file or directory", you need the fix below
```

## Fix

This repo's own `.gitignore` already excludes the live runtime data
(`V2/MemoryStructures/STM/stm.json`, `V2/MemoryStructures/LTM/data.mdb`,
`lock.mdb`, `asmc.faiss`, `counter.json`), so wiring up git in place will not
touch or reset any live memory data.

```bash
cd AdvancedSemanticMemoryClustering
git init
git remote add origin git@github.com:SeanMurphy2k17/AdvancedSemanticMemoryClustering-ASMC.git
git fetch origin

# Force checkout: the existing files are untracked (no .git existed before),
# so a normal checkout will refuse due to "would be overwritten" conflicts.
# This is expected and safe — diff your files against the fetched commit first
# if you want to confirm there's no uncommitted local-only work before forcing it:
#   git diff --no-index . <(git show origin/main:path/to/file)
git checkout -f -b main origin/main
```

After this, `git status` should report "up to date with origin/main" and only
the live runtime files (excluded by `.gitignore`) should show as untracked.

From then on, this folder behaves like any normal git repo — `git pull`,
`git push`, etc. all work directly inside it.
