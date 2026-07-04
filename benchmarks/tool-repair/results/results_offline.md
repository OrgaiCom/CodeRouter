# L1 Offline Tool-Repair Benchmark

- Generated: 2026-07-04T23:46:03.699617+00:00
- Total cases: 55
- Overall recall (recovered / expect-repair): 43/43 = 100.0%
- Overall false-positive rate (fp / expect-pass): 0/12 = 0.0%
- Missed: 0  |  Correct-pass: 12

## Per-category

| category | cases | recovered | missed | recall | correct_pass | false_pos | fp_rate |
|---|---|---|---|---|---|---|---|
| fenced_json | 6 | 6 | 0 | 100.0% | 0 | 0 | n/a |
| bare_json | 5 | 5 | 0 | 100.0% | 0 | 0 | n/a |
| multiple_calls | 6 | 6 | 0 | 100.0% | 0 | 0 | n/a |
| malformed | 9 | 9 | 0 | 100.0% | 0 | 0 | n/a |
| nested_fence | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| thinking_leak | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| cjk_arguments | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| negative | 12 | 0 | 0 | n/a | 12 | 0 | 0.0% |
| xml_forms | 5 | 5 | 0 | 100.0% | 0 | 0 | n/a |

## Missed cases (0)

_none_

## False positives (0)

_none — repairer produced no over-eager extractions._

## deduplicate_tool_calls stability

All multiple_calls cases: re-applying deduplicate_tool_calls is idempotent (stable).
