# Examples & Tutorials Tracking

This file tracks examples and tutorials that Relay could not automatically fix.
Future Relay runs will skip these files to avoid stalling on known issues.

**To re-enable a file**: Remove its row from the table below after manually fixing it.

---

## Unfixable Files

| File | Error Type | Reason | Date Added |
|------|------------|--------|------------|
| tutorials/single_integrator_dynamic_obstacles.py | Empty File | File content is missing/empty. | 2026-02-02 |
| tutorials/tutorials/mppi_cbf_si/ros2/controller.py | ROS2 Dependency | Requires ROS2 infrastructure | 2026-02-02 |

---

## Notes

- Files are added here when Relay attempts a fix but cannot resolve the issue within its constraints (< 50 LOC, no new dependencies, no architecture changes).
- A human should review these files and either fix them manually or mark them as intentionally deprecated.
- Once fixed, remove the entry from the table above so Relay will verify them again.
