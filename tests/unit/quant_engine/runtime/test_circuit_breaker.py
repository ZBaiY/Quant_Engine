### TODO: Testing Circuit Breaker, to be implemented later
### This mimics the following coditions:
# 1. MarketInfo with different gap types
# 2. Data ticks arriving with gaps larger than expected intervals
# 3. Circuit breaker triggering and resetting
# 4. Ensuring correct gap type is reported in snapshots
# 5. Edge cases with missing data and out-of-order ticks
# 6. Integration with data handlers and snapshot generation
# 7. Performance under high-frequency tick ingestion
# This requires mocking data sources and simulating time progression.
#  