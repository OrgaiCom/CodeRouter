# L1 Offline Tool-Repair Benchmark

- Generated: 2026-07-05T06:33:13.469742+00:00
- Total cases: 84
- Overall recall (recovered / expect-repair): 59/59 = 100.0%
- Overall false-positive rate (fp / expect-pass): 0/25 = 0.0%
- Missed: 0  |  Correct-pass: 25

## Per-category

| category | cases | recovered | missed | recall | correct_pass | false_pos | fp_rate |
|---|---|---|---|---|---|---|---|
| fenced_json | 6 | 6 | 0 | 100.0% | 0 | 0 | n/a |
| bare_json | 5 | 5 | 0 | 100.0% | 0 | 0 | n/a |
| multiple_calls | 6 | 6 | 0 | 100.0% | 0 | 0 | n/a |
| malformed | 11 | 11 | 0 | 100.0% | 0 | 0 | n/a |
| nested_fence | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| thinking_leak | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| cjk_arguments | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| negative | 24 | 0 | 0 | n/a | 24 | 0 | 0.0% |
| xml_forms | 5 | 5 | 0 | 100.0% | 0 | 0 | n/a |
| nested_xml | 5 | 5 | 0 | 100.0% | 0 | 0 | n/a |
| json_wrappers | 4 | 4 | 0 | 100.0% | 0 | 0 | n/a |
| python_call | 6 | 5 | 0 | 100.0% | 1 | 0 | 0.0% |

## Missed cases (0)

_none_

## False positives (0)

_none — repairer produced no over-eager extractions._

## deduplicate_tool_calls stability

All multiple_calls cases: re-applying deduplicate_tool_calls is idempotent (stable).
