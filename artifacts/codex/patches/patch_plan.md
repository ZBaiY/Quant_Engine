# Patch Plan

1. Add raw path mirror resolver and ingestion raw writer helpers (parquet/jsonl).
2. Implement worker-owned backfill (fetch → persist raw → emit tick) without FileSource persistence.
3. Rewire handlers/apps to use backfill workers (no runtime Source calls) and update tests for raw persistence + runtime isolation.
