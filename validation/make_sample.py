"""Draw the validation sample of crash and walking segments.

Segments are stratified by continent (and, for crashes, by event kind), with at
most one segment per video. Each stratum gets a minimum number of clips and the
rest is split in proportion to its size; the sampling weight of every clip is
stored so the analysis can report dataset-level figures. The sample is split
into two disjoint halves, one per reviewer.

    python validation/make_sample.py \
        --crash-segments crash_segments.csv \
        --walking-mapping ../walking-yt/mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_PLAY_SECONDS = 4.0
TIME_OF_DAY_NAMES = {"0": "day", "1": "night", "2": "dawn_dusk", "-1": "unknown"}

csv.field_size_limit(sys.maxsize)


def parse_list(text: str):
    """Parse the unquoted nested lists of the walking mapping, e.g. [[a,b],[c]]."""
    stack: list[list] = [[]]
    token = ""
    for char in text.strip():
        if char == "[":
            stack.append([])
        elif char in ",]":
            if token.strip():
                stack[-1].append(token.strip())
            token = ""
            if char == "]":
                done = stack.pop()
                stack[-1].append(done)
        else:
            token += char
    return stack[0][0] if stack[0] else []


def location_text(locality: str, state: str, country: str) -> str:
    parts = [part for part in (locality, state, country) if part]
    return ", ".join(dict.fromkeys(parts))


def play_window(start: float, end: float) -> tuple[float, float]:
    if end - start >= MIN_PLAY_SECONDS:
        return start, end
    middle = (start + end) / 2
    play_start = max(0.0, middle - MIN_PLAY_SECONDS / 2)
    return play_start, play_start + MIN_PLAY_SECONDS


def crash_pool(path: Path) -> list[dict]:
    pool = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["event_kind"] or row["crash_taxonomy_status"] == "video_unavailable":
                continue
            start, end = float(row["start_time"]), float(row["end_time"])
            continent = row["continent"] or "Unresolved"
            pool.append({
                "id": row["segment_id"],
                "dataset": "crash",
                "video_id": row["video_id"],
                "start": start,
                "end": end,
                "stratum": f"{continent} | {row['event_kind']}",
                "location": location_text(row["locality"], row["state"], row["country"]),
                "model": {
                    field: row[field]
                    for field in (
                        "event_kind", "manner_of_collision", "crash_type",
                        "camera_view", "time_of_day", "weather",
                        "road_environment", "crash_taxonomy_version",
                    )
                },
            })
    return pool


def walking_pool(path: Path) -> list[dict]:
    pool = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            videos = parse_list(row["videos"])
            columns = {
                name: parse_list(row[name])
                for name in ("start_time", "end_time", "time_of_day", "walking_environment")
            }
            continent = row["continent"] or "Unresolved"
            location = location_text(row["locality"], row["state"], row["country"])
            for v, video_id in enumerate(videos):
                for s, start in enumerate(columns["start_time"][v]):
                    end = columns["end_time"][v][s]
                    pool.append({
                        "id": f"{video_id}_{start}_{end}",
                        "dataset": "walking",
                        "video_id": video_id,
                        "start": float(start),
                        "end": float(end),
                        "stratum": continent,
                        "location": location,
                        "model": {
                            "content": "walking",
                            "time_of_day": TIME_OF_DAY_NAMES.get(
                                columns["time_of_day"][v][s], "unknown"
                            ),
                            "walking_environment": columns["walking_environment"][v][s],
                        },
                    })
    return pool


def allocate(sizes: dict[str, int], total: int, minimum: int) -> dict[str, int]:
    """Give every stratum `minimum` clips (or all it has), the rest by size."""
    alloc = {name: min(size, minimum) for name, size in sizes.items()}
    left = total - sum(alloc.values())
    spare = {name: sizes[name] - alloc[name] for name in sizes}
    spare_total = sum(spare.values())
    if left > 0 and spare_total:
        shares = {name: left * spare[name] / spare_total for name in sizes}
        for name in sizes:
            alloc[name] += min(spare[name], int(shares[name]))
        # Hand out the remainder by largest fractional share.
        order = sorted(sizes, key=lambda name: shares[name] - int(shares[name]), reverse=True)
        for name in order:
            if sum(alloc.values()) >= total:
                break
            if alloc[name] < sizes[name]:
                alloc[name] += 1
    return alloc


def draw(pool: list[dict], total: int, minimum: int, rng: random.Random) -> list[dict]:
    strata: dict[str, list[dict]] = defaultdict(list)
    for item in pool:
        strata[item["stratum"]].append(item)
    # One segment per video: count videos, not segments, when allocating.
    video_counts = {
        name: len({item["video_id"] for item in items}) for name, items in strata.items()
    }
    alloc = allocate(video_counts, total, minimum)
    used_videos: set[str] = set()
    chosen = []
    for name in sorted(strata):
        items = strata[name][:]
        rng.shuffle(items)
        picked = []
        for item in items:
            if len(picked) == alloc[name]:
                break
            if item["video_id"] in used_videos:
                continue
            used_videos.add(item["video_id"])
            picked.append(item)
        for item in picked:
            item["weight"] = len(items) / len(picked)
        chosen.extend(picked)
    return chosen


def split_reviewers(items: list[dict], rng: random.Random) -> tuple[list[str], list[str]]:
    """Alternate within each dataset and stratum so both halves look alike."""
    halves: tuple[list[str], list[str]] = ([], [])
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        groups[(item["dataset"], item["stratum"])].append(item)
    turn = 0
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        for item in group:
            halves[turn].append(item["id"])
            turn = 1 - turn
    for half in halves:
        rng.shuffle(half)
    return halves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crash-segments", type=Path, required=True)
    parser.add_argument("--walking-mapping", type=Path, required=True)
    parser.add_argument("--n-crash", type=int, default=200)
    parser.add_argument("--n-walking", type=int, default=200)
    parser.add_argument("--min-per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out", type=Path, default=HERE / "sample.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = draw(crash_pool(args.crash_segments), args.n_crash, args.min_per_stratum, rng)
    items += draw(walking_pool(args.walking_mapping), args.n_walking, args.min_per_stratum, rng)
    for item in items:
        item["play_start"], item["play_end"] = play_window(item["start"], item["end"])
    first, second = split_reviewers(items, rng)

    sample = {
        "created": date.today().isoformat(),
        "seed": args.seed,
        "sources": {
            "crash": str(args.crash_segments),
            "walking": str(args.walking_mapping),
        },
        "queues": [first, second],
        "items": {item["id"]: item for item in items},
    }
    args.out.write_text(json.dumps(sample, indent=1), encoding="utf-8")

    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[f"{item['dataset']}: {item['stratum']}"] += 1
    for name in sorted(counts):
        print(f"{counts[name]:4d}  {name}")
    print(f"{len(items)} clips, {len(first)} + {len(second)} per reviewer -> {args.out}")


if __name__ == "__main__":
    main()
