Tests layout:
- unit/: pure logic, no IO
- integration/: multi-module pipelines
- smoke/: minimal entrypoint checks
Add real tests by replacing *_placeholder.py with assertions over snapshots / determinism.
