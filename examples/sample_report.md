# Sample audit reports

Real captured output of `python examples/run_audit.py` (the shipped demo task, `data/demo_task.json`) and of `python -m context_segment_value demo --scenario redundant_key --seed 2`. Nothing below is edited.

## A contradictory tool output

```
Context segment value audit -- task-contradictory_tool-3
Scenario: contradictory_tool | seed 3 | 6 segments | 114 tokens
Query: For the server deployment, what is the confirmed target region?
Full-context measured success: 0.443

Segment prices (leave-one-out, paired replays):
  id    role         planted class  tokens    value per-token  verdict
  s02   retrieved    key                30   +0.335   +0.0112  asset
  s04   retrieved    supporting         28   +0.180   +0.0064  asset
  s05   history      neutral            10   +0.015   +0.0015  inert
  s06   history      neutral            12   +0.013   +0.0010  inert
  s01   system       neutral            26   +0.000   +0.0000  inert
  s03   tool_output  contradictory       8   -0.422   -0.0528  liability

Negative-value register:
  s03 [tool_output / contradictory] value -0.422 over 8 tokens -- removing it raises measured success

Duplicate probes:
  (no near-duplicate pairs)
```

## Redundancy masking

```
Context segment value audit -- task-redundant_key-2
Scenario: redundant_key | seed 2 | 6 segments | 145 tokens
Query: For the expense approval, what is the confirmed spending limit?
Full-context measured success: 0.675

Segment prices (leave-one-out, paired replays):
  id    role         planted class  tokens    value per-token  verdict
  s04   retrieved    supporting         28   +0.213   +0.0076  asset
  s02   retrieved    key                31   +0.073   +0.0023  asset
  s03   retrieved    duplicate_key      27   +0.073   +0.0027  asset
  s01   system       neutral            22   +0.000   +0.0000  inert
  s06   history      neutral            11   -0.010   -0.0009  inert
  s05   retrieved    distractor         26   -0.193   -0.0074  liability

Negative-value register:
  s05 [retrieved / distractor] value -0.193 over 26 tokens -- removing it raises measured success

Duplicate probes:
  s02 + s03 (similarity 0.62): individual values +0.073 / +0.073, joint value +0.518 -- REDUNDANCY MASKING
```
