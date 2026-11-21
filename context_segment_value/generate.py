"""Seeded task generator with planted ground truth.

Each scenario family assembles a context the way real agent prompts
actually go wrong:

- ``standard_mix``      a healthy prompt with one distractor riding
                        along;
- ``stale_instruction`` an outdated instruction left in the system
                        text next to its replacement;
- ``contradictory_tool`` a tool output that contradicts the retrieved
                        key fact, formatted with full authority;
- ``redundant_key``     the key fact retrieved twice, near-verbatim
                        -- the redundancy-masking case;
- ``bloated_history``   a long tail of neutral history turns burying
                        a small useful core;
- ``distractor_heavy``  several topically-similar wrong chunks that
                        a relevance ranker will love and a value
                        ranker will drop.

Segment texts are composed from per-subject templates. Distractor
texts deliberately share the query's vocabulary (that is what makes
them distractors), while neutral history shares almost none.
"""

import random

from .models import BASE_CONTRIBUTIONS, Segment, Task

# (subject, field, correct value, wrong value)
SUBJECTS = (
    ("database migration", "cutover window", "Sunday 02:00", "Friday 18:00"),
    ("flight booking", "departure time", "09:40", "17:25"),
    ("expense approval", "spending limit", "750 dollars", "250 dollars"),
    ("server deployment", "target region", "eu-central", "us-west"),
    ("course registration", "enrolment deadline", "March 14", "January 30"),
    ("visa application", "appointment date", "June 9", "April 2"),
    ("lab equipment order", "delivery week", "week 31", "week 27"),
    ("conference travel", "hotel check-in", "October 5", "October 8"),
)

HISTORY_LINES = (
    "User said thanks and asked to keep the summary short.",
    "Assistant acknowledged and offered to continue when ready.",
    "User mentioned the weather had delayed their commute today.",
    "Assistant confirmed the previous unrelated request was closed.",
    "User asked for a reminder about an unrelated meeting tomorrow.",
    "Assistant noted the preference for concise bullet points.",
    "User said the earlier draft looked fine and moved on.",
    "Assistant asked whether anything else needed attention.",
)

SCENARIOS = (
    "standard_mix",
    "stale_instruction",
    "contradictory_tool",
    "redundant_key",
    "bloated_history",
    "distractor_heavy",
)


def _seg(seg_id, role, value_class, text, position, rng, dup_group=None):
    contribution = BASE_CONTRIBUTIONS[value_class]
    if value_class == "neutral":
        contribution = rng.uniform(-0.05, 0.05)
    return Segment(seg_id, role, value_class, text, position,
                   contribution, dup_group=dup_group)


def _system_current(subject, field, correct):
    return ("System: You are a careful operations assistant. For %s, "
            "always use the confirmed %s, which is %s. Prefer retrieved "
            "reference material over informal notes." % (subject, field, correct))


def _system_plain(subject):
    return ("System: You are a careful operations assistant helping "
            "with %s. Answer precisely and cite the reference "
            "material when it is available." % subject)


def _key_text(subject, field, correct):
    return ("Reference entry for %s: the confirmed %s is %s. This entry "
            "was verified against the master schedule and supersedes "
            "all earlier drafts and informal notes on the topic."
            % (subject, field, correct))


def _key_copy_text(subject, field, correct):
    return ("Reference entry for %s (retrieved copy): the confirmed %s "
            "is %s. Verified against the master schedule; this entry "
            "supersedes earlier drafts and informal notes."
            % (subject, field, correct))


def _supporting_text(subject, field):
    return ("Supporting note on %s: the %s was double-checked by the "
            "coordination team this week, and the reference entry is "
            "the one to trust if anything disagrees." % (subject, field))


def _distractor_text(subject, field, wrong, variant):
    templates = (
        "Forum post about %s: someone mentioned the %s might be %s, "
        "though nobody confirmed it and the thread went off topic.",
        "Old blog summary of %s claiming the %s is %s; readers in the "
        "comments disputed the claim but it still ranks in search.",
        "Unofficial cheat-sheet for %s lists the %s as %s, copied from "
        "an outdated edition that was never corrected.",
    )
    return templates[variant % len(templates)] % (subject, field, wrong)


def _stale_text(subject, field, wrong):
    return ("Instruction (legacy, kept for reference): for %s, use the "
            "%s %s. This was the policy before the schedule change."
            % (subject, field, wrong))


def _current_instruction_text(subject, field, correct):
    return ("Instruction (current): for %s, the %s is %s. This replaces "
            "the legacy instruction above wherever they disagree."
            % (subject, field, correct))


def _contradictory_tool_text(subject, field, wrong):
    return ("Tool output [schedule_lookup]: subject=%s field=%s "
            "value=%s status=ok source=cache" % (
                subject.replace(" ", "_"), field.replace(" ", "_"), wrong))


def _tool_ok_text(subject, field, correct):
    return ("Tool output [schedule_lookup]: subject=%s field=%s "
            "value=%s status=ok source=master" % (
                subject.replace(" ", "_"), field.replace(" ", "_"), correct))


def _build(scenario, seed):
    subject, field, correct, wrong = SUBJECTS[seed % len(SUBJECTS)]
    spec = []

    def add(role, value_class, text, dup_group=None):
        spec.append((role, value_class, text, dup_group))

    if scenario == "standard_mix":
        add("system", "neutral", _system_current(subject, field, correct))
        add("retrieved", "key", _key_text(subject, field, correct))
        add("retrieved", "supporting", _supporting_text(subject, field))
        add("retrieved", "distractor",
            _distractor_text(subject, field, wrong, seed))
        n_hist = 2
    elif scenario == "stale_instruction":
        add("system", "stale", _stale_text(subject, field, wrong))
        add("instruction", "supporting",
            _current_instruction_text(subject, field, correct))
        add("system", "neutral", _system_plain(subject))
        add("retrieved", "key", _key_text(subject, field, correct))
        add("retrieved", "distractor",
            _distractor_text(subject, field, wrong, seed + 1))
        n_hist = 1
    elif scenario == "contradictory_tool":
        add("system", "neutral", _system_current(subject, field, correct))
        add("retrieved", "key", _key_text(subject, field, correct))
        add("tool_output", "contradictory",
            _contradictory_tool_text(subject, field, wrong))
        add("retrieved", "supporting", _supporting_text(subject, field))
        n_hist = 2
    elif scenario == "redundant_key":
        add("system", "neutral", _system_plain(subject))
        add("retrieved", "key", _key_text(subject, field, correct),
            dup_group="keyfact")
        add("retrieved", "duplicate_key",
            _key_copy_text(subject, field, correct), dup_group="keyfact")
        add("retrieved", "supporting", _supporting_text(subject, field))
        add("retrieved", "distractor",
            _distractor_text(subject, field, wrong, seed + 2))
        n_hist = 1
    elif scenario == "bloated_history":
        add("system", "neutral", _system_current(subject, field, correct))
        add("retrieved", "key", _key_text(subject, field, correct))
        add("retrieved", "supporting", _supporting_text(subject, field))
        add("retrieved", "distractor",
            _distractor_text(subject, field, wrong, seed))
        n_hist = 6
    elif scenario == "distractor_heavy":
        add("system", "neutral", _system_plain(subject))
        add("retrieved", "key", _key_text(subject, field, correct))
        for k in range(3):
            add("retrieved", "distractor",
                _distractor_text(subject, field, wrong, seed + k))
        add("tool_output", "supporting", _tool_ok_text(subject, field, correct))
        n_hist = 1
    else:
        raise ValueError("unknown scenario: %s" % scenario)

    for k in range(n_hist):
        line = HISTORY_LINES[(seed + k) % len(HISTORY_LINES)]
        add("history", "neutral", "History turn: " + line)

    return spec, (subject, field)


def build_task(scenario, seed):
    """Build one seeded task from a scenario family."""
    spec, (subject, field) = _build(scenario, seed)
    rng = random.Random(seed * 7919 + 13)
    segments = []
    for idx, (role, value_class, text, dup_group) in enumerate(spec):
        seg_id = "s%02d" % (idx + 1)
        segments.append(_seg(seg_id, role, value_class, text, idx, rng,
                             dup_group=dup_group))
    query = "For the %s, what is the confirmed %s?" % (subject, field)
    return Task("task-%s-%d" % (scenario, seed), scenario, seed, query,
                segments)


def build_suite(seeds=None):
    """All scenario families x the given seeds (default 0..7)."""
    if seeds is None:
        seeds = range(8)
    tasks = []
    for scenario in SCENARIOS:
        for seed in seeds:
            tasks.append(build_task(scenario, seed))
    return tasks


def task_to_dict(task):
    """Serialise a task (draws are rebuilt from the seed on load)."""
    return {
        "task_id": task.task_id,
        "scenario": task.scenario,
        "seed": task.seed,
        "query": task.query,
        "segments": [
            {
                "segment_id": seg.segment_id,
                "role": seg.role,
                "value_class": seg.value_class,
                "text": seg.text,
                "position": seg.position,
                "contribution": seg.contribution,
                "dup_group": seg.dup_group,
            }
            for seg in task.segments
        ],
    }


def task_from_dict(data):
    segments = [
        Segment(
            item["segment_id"],
            item["role"],
            item["value_class"],
            item["text"],
            item["position"],
            item["contribution"],
            dup_group=item.get("dup_group"),
        )
        for item in data["segments"]
    ]
    return Task(data["task_id"], data["scenario"], data["seed"],
                data["query"], segments)
