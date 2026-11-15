#!/bin/bash
# Commit and push all changes of this repository to origin/main. Never force-pushes. Safe to run
# repeatedly (it does nothing when there is nothing to commit), e.g. from tools/autopush_loop.sh or cron.
#   tools/autopush.sh ["commit message"]
cd "$(dirname "$0")/.." || exit 1
git add -A
git diff --cached --quiet && exit 0

# safety: no large files, no credentials/personal paths in what is about to be committed
big=$(git diff --cached --name-only | while read -r f; do [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 20000000 ] && echo "$f"; done)
if [ -n "$big" ]; then echo "autopush: refusing, files over 20 MB staged: $big" >&2; git reset -q; exit 2; fi
if git diff --cached -U0 | grep -E '^\+' | grep -qiE 'gho_[A-Za-z0-9]|ghp_[A-Za-z0-9]|password *[=:]|/home/[a-z]+/|/media/[a-z]+/|@gmail\.com'; then
  echo "autopush: refusing, staged changes contain a credential or a personal path" >&2; git reset -q; exit 3
fi

git commit -q -m "${1:-autosave $(date '+%Y-%m-%d %H:%M')}

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || exit 4
git pull --rebase -q origin main || { echo "autopush: pull --rebase failed, not pushing" >&2; exit 5; }
git push -q origin main || { echo "autopush: push failed (offline?), will retry next time" >&2; exit 6; }
echo "autopush: pushed $(git rev-parse --short HEAD)"
