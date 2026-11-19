"""Assign pilot participants to the recommendation group or the control group (run once, before
the first trading day).

    python scripts/field_eval/assign_groups.py participants.csv --seed 20261118 > assignment.csv

participants.csv has the columns participant_id, name, stratum (stratum: team or outside).
Within each stratum the participants are shuffled with the seed and dealt to the two groups in
turn, starting with a group drawn by a coin flip, so each group gets half of every stratum
(4 team members -> 2 + 2, 2 outside participants -> 1 + 1). The seed and this script are the
record of the assignment.
"""

import argparse
import csv
import random
import sys

GROUPS = ("recommendation", "control")


def assign(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for stratum in sorted({r["stratum"] for r in rows}):
        members = sorted(
            (r for r in rows if r["stratum"] == stratum), key=lambda r: r["participant_id"]
        )
        rng.shuffle(members)
        order = GROUPS if rng.random() < 0.5 else GROUPS[::-1]
        out += [{**r, "group": order[i % 2]} for i, r in enumerate(members)]
    return sorted(out, key=lambda r: r["participant_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("participants")
    ap.add_argument("--seed", type=int, default=20261118)
    args = ap.parse_args()
    with open(args.participants) as fh:
        rows = list(csv.DictReader(fh))
    w = csv.DictWriter(sys.stdout, fieldnames=["participant_id", "name", "stratum", "group"])
    w.writeheader()
    w.writerows(assign(rows, args.seed))


if __name__ == "__main__":
    main()
