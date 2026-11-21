"""Context Segment Value Audit.

Price every segment of an agent's context -- system text, tool
outputs, history turns, retrieved chunks -- by its causal
leave-one-out effect on task success, and register the segments
whose value is negative: the distractors, stale instructions and
contradictory tool outputs that measurably hurt the task they ride
along with.
"""

__version__ = "0.1.0"
