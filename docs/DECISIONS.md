# Decisions Log — SVARA

Record of architectural, data, modeling, and scoping decisions.
Format: Date | Decision | Rationale | Owner

| Date | ID | Decision | Rationale | Owner |
|---|---|---|---|---|
| 2026-10-08 | DEC-001 | Repository scaffolding initialized | Set up baseline folder structure per docs/03 §4 and P0-04 | Y |
| 2026-10-08 | DEC-002 | Direct main push policy for Yoga | Development by Yoga commits directly / merges to main; Haikal uses feature branches & review PRs | Y |
| 2026-10-08 | DEC-003 | Audio duration cutoff set to 5.0 seconds | Measured full dataset (30,043 clips) duration p99 = 4.181s; rounded up to 5s per docs/01 §3 and P1-06 | Y |
