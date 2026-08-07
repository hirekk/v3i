# Wayfinder tracker

This repo's wayfinder efforts live on **GitHub Issues** (`hirekk/v3i`). The
local-markdown tracker that briefly lived in this directory was migrated there
on 2026-08-07; issue bodies, resolution comments, and blocking edges carry the
full content.

- **Map**: the issue labelled `wayfinder:map` — currently
  [Widened octonion architecture — from isometric chains to an implementable spec (#1)](https://github.com/hirekk/v3i/issues/1).
- **Tickets**: child issues of the map (native sub-issues), each labelled
  `wayfinder:research` / `wayfinder:grilling` / `wayfinder:prototype` /
  `wayfinder:task`.

## Wayfinding operations

- **Create**: `gh issue create` with a `wayfinder:<type>` label, then attach as
  a sub-issue of the map:
  `gh api repos/hirekk/v3i/issues/<map>/sub_issues -X POST -F sub_issue_id=<id>`
  (`<id>` is the issue's database id: `gh api repos/hirekk/v3i/issues/<n> -q .id`).
- **Blocking**: native issue dependencies —
  `gh api repos/hirekk/v3i/issues/<n>/dependencies/blocked_by -X POST -F issue_id=<id>`.
  A ticket is unblocked when every issue blocking it is closed; GitHub renders
  this in the issue UI.
- **Claim**: assign the issue (`gh issue edit <n> --add-assignee <user>`)
  before any work. Open + unassigned = unclaimed.
- **Frontier**: open, unblocked, unclaimed sub-issues of the map:
  `gh issue list -R hirekk/v3i --state open --no-assignee` filtered by the
  blocked-by relationships shown on each issue.
- **Resolve**: post the answer as an issue comment, close as completed, and add
  a one-line gist linking the ticket to the map issue's *Decisions so far*
  section (edit the map body).
- **Out of scope**: close the ticket and add a line to the map's *Out of scope*
  section instead of *Decisions so far*.
- **Assets**: research notes and verification scripts live under
  `docs/research/` in this repo; link them from issues via `blob/main` URLs.
