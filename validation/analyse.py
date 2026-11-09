"""Compare the reviewers' answers with the pipeline labels.

For every question that has a model field, it reports how often the reviewer
agreed with the model (raw and weighted by the sampling weights), Cohen's kappa,
and agreement without the clips the reviewer could not judge. The confusion
tables go to validation/results/. Pass --crash-segments to compare against the
newest crash labels instead of the ones stored with the sample.

    python validation/analyse.py --crash-segments crash_segments.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANT_TELL = {"unknown", "cant_tell"}

# Free-text model values mapped onto the reviewers' options.
SYNONYMS = {
    "road_environment": {
        "intersection": "junction", "highway": "motorway",
        "residential street": "urban", "residential": "urban", "mountain road": "rural",
    },
    "weather": {
        "snowy": "snow", "foggy": "fog", "hazy": "fog", "overcast": "clear",
        "stormy": "rain", "storm": "rain", "windy": "clear",
    },
    "time_of_day": {"dusk": "dawn_dusk", "dawn": "dawn_dusk"},
}

csv.field_size_limit(sys.maxsize)


def shown(question: dict, values: dict) -> bool:
    """Whether the review page shows this question for these answers."""
    return all(values.get(qid) in allowed for qid, allowed in question.get("show_if", {}).items())


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    human = Counter(a for a, _ in pairs)
    model = Counter(b for _, b in pairs)
    expected = sum(human[k] * model[k] for k in human) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def weighted_share(rows: list[tuple[str, str, float]]) -> float | None:
    total = sum(weight for _, _, weight in rows)
    if not total:
        return None
    return sum(weight for a, b, weight in rows if a == b) / total


def fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=Path, default=HERE / "sample.json")
    parser.add_argument("--state", type=Path, default=HERE / "data" / "state.json")
    parser.add_argument("--crash-segments", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    items = sample["items"]

    fresh: dict[str, dict] = {}
    if args.crash_segments:
        with args.crash_segments.open(newline="", encoding="utf-8") as handle:
            fresh = {row["segment_id"]: row for row in csv.DictReader(handle) if row["segment_id"] in items}

    answers = []
    for reviewer in state["reviewers"]:
        for item_id, answer in (reviewer or {}).get("answers", {}).items():
            answers.append((reviewer["name"], items[item_id], answer["values"]))
    unavailable = sum(len((r or {}).get("unavailable", {})) for r in state["reviewers"])
    print(f"{len(answers)} answered clips, {unavailable} could not be played\n")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset, dataset_questions in questions.items():
        rows = [(name, item, values) for name, item, values in answers if item["dataset"] == dataset]
        if not rows:
            continue
        print(f"== {dataset} ({len(rows)} clips)")
        print(f"{'question':24} {'n':>4} {'agree':>6} {'wtd':>6} {'kappa':>6} {'judged':>7} {'agree*':>6}")
        for question in dataset_questions:
            qid = question["id"]
            # Answers given before a question became conditional are left out.
            answered = [(item, values[qid]) for _, item, values in rows
                        if qid in values and shown(question, values)]
            if not answered:
                continue
            field = question.get("model_field")
            if not field:
                counts = Counter(value for _, value in answered)
                shares = ", ".join(f"{k} {v / len(answered):.0%}" for k, v in counts.most_common())
                print(f"{qid:24} {len(answered):4d}  {shares}")
                summary.append({"dataset": dataset, "question": qid, "n": len(answered), "distribution": dict(counts)})
                continue
            triples = []
            for item, human in answered:
                label = (fresh.get(item["id"]) or item["model"]).get(field, "") or "unknown"
                label = SYNONYMS.get(field, {}).get(label, label)
                triples.append((human, label, item.get("weight", 1.0)))
            pairs = [(human, label) for human, label, _ in triples]
            judged = [t for t in triples if t[0] not in CANT_TELL]
            result = {
                "dataset": dataset, "question": qid, "n": len(triples),
                "agreement": sum(a == b for a, b in pairs) / len(pairs),
                "weighted_agreement": weighted_share(triples),
                "kappa": kappa(pairs),
                "judged": len(judged),
                "agreement_judged": weighted_share([(a, b, 1.0) for a, b, _ in judged]),
            }
            summary.append(result)
            print(f"{qid:24} {result['n']:4d} {fmt(result['agreement']):>6} {fmt(result['weighted_agreement']):>6} "
                  f"{fmt(result['kappa']):>6} {result['judged']:7d} {fmt(result['agreement_judged']):>6}")

            table: dict[str, Counter] = defaultdict(Counter)
            for human, label in pairs:
                table[human][label] += 1
            labels = sorted({label for _, label in pairs} | set(table))
            with (args.out / f"confusion_{dataset}_{qid}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["reviewer \\ model"] + labels)
                for human in labels:
                    writer.writerow([human] + [table[human][label] for label in labels])
        print()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("agree* = agreement on clips the reviewer could judge (no 'can't tell')")
    print(f"Tables: {args.out}")


if __name__ == "__main__":
    main()
