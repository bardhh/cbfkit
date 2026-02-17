# `.jules` Artifact Policy

Status date: 2026-02-17

Purpose:
- Keep repository history/reviews focused on source and test changes.
- Avoid committing generated Relay planning artifacts that are not runtime dependencies.

Policy:
1. Allowed tracked files under `.jules/`:
   - `.jules/bolt.md`
   - `.jules/examples_tutorials_tracking.md`
2. Disallowed generated artifacts under `.jules/`:
   - `.jules/relay_planner.py`
   - `.jules/relay_tasks.json`
3. Enforcement:
   - `scripts/check_jules_policy.sh` fails if disallowed `.jules` files are present.
   - `scripts/run_premerge_gate.sh` runs this check before test gates.

Rationale:
- The disallowed files are generated planning snapshots/process tooling,
  not package/runtime assets.
- Keeping them out reduces merge noise and long-term maintenance debt.
