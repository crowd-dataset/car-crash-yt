"""Compare where the dataset's videos come from with where road deaths happen.

For every country the share of videos is set against its share of the world's
road traffic deaths (WHO estimates, 2021). The ratio of the two shares is the
representation index: 1 means the country appears as often as its road deaths
would suggest, above 1 over-represented, below 1 under-represented. The same is
reported per continent, together with how much of the world's road deaths
happen in countries the dataset covers at all.

The reference table (reference/country_baseline.csv) is committed so the
numbers can be reproduced; --refresh-reference downloads it again from the WHO
GHO and World Bank APIs.

    python coverage_baseline.py
    python coverage_baseline.py --walking-mapping ../walking-yt/mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import plotly.graph_objects as go
import pycountry
from scipy.stats import spearmanr

from car_crash_pipeline.location import CONTINENT_CODES

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "country_baseline.csv"
YEAR = 2021
WHO_DEATH_RATE = "https://ghoapi.azureedge.net/api/RS_198?$filter=Dim1%20eq%20'SEX_BTSX'"
WHO_VEHICLES = "https://ghoapi.azureedge.net/api/RS_194"
WB_POPULATION = (
    "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
    f"?format=json&date={YEAR}&per_page=400"
)
SERIES_COLOURS = {"videos": "#2a78d6", "deaths": "#eb6834"}

csv.field_size_limit(sys.maxsize)


# Territories the pipeline's continent table leaves out.
EXTRA_CONTINENTS = {"PRI": "North America", "TTO": "North America"}


def continent_of(iso3: str) -> str:
    if iso3 in EXTRA_CONTINENTS:
        return EXTRA_CONTINENTS[iso3]
    country = pycountry.countries.get(alpha_3=iso3)
    if country:
        for name, codes in CONTINENT_CODES.items():
            if country.alpha_2 in codes:
                return name
    return ""


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def refresh_reference(path: Path) -> None:
    """Download road death rates, vehicles and population into one table."""
    rates = {
        row["SpatialDim"]: row["NumericValue"]
        for row in fetch_json(WHO_DEATH_RATE)["value"]
        if row["SpatialDimType"] == "COUNTRY" and row["TimeDim"] == YEAR
    }
    vehicles: dict[str, tuple[int, float]] = {}
    for row in fetch_json(WHO_VEHICLES)["value"]:
        if row["SpatialDimType"] != "COUNTRY" or row["NumericValue"] is None:
            continue
        if row["SpatialDim"] not in vehicles or row["TimeDim"] > vehicles[row["SpatialDim"]][0]:
            vehicles[row["SpatialDim"]] = (row["TimeDim"], row["NumericValue"])
    population = {
        row["countryiso3code"]: row["value"]
        for row in fetch_json(WB_POPULATION)[1]
        if row["countryiso3code"] and row["value"]
    }
    # Taiwan is in neither source; the WHO rate list decides which countries count.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "iso3", "country", "continent", f"population_{YEAR}",
            f"road_death_rate_{YEAR}", f"road_deaths_{YEAR}",
            "registered_vehicles", "vehicles_year",
        ])
        for iso3 in sorted(rates):
            if iso3 not in population:
                continue
            country = pycountry.countries.get(alpha_3=iso3)
            year, count = vehicles.get(iso3, ("", ""))
            writer.writerow([
                iso3, country.name if country else iso3, continent_of(iso3),
                population[iso3], rates[iso3],
                round(rates[iso3] * population[iso3] / 100_000),
                int(count) if count != "" else "", year,
            ])
    print(f"Reference: {len(rates)} WHO countries -> {path}")


def read_reference(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["iso3"]: row for row in csv.DictReader(handle)}


def count_mapping(path: Path) -> tuple[Counter, Counter, int, int]:
    """Videos and segments per country, videos placed in any country, and
    videos with no segment in any country.

    A compilation video counts once in every country it shows, so the
    per-country counts add up to more than the number of placed videos.
    """
    videos: dict[str, set[str]] = defaultdict(set)
    segments: Counter = Counter()
    unresolved: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ids = [v for v in re.split(r"[\[\],\s]+", row["videos"]) if v]
            if not row["iso3"]:
                unresolved.update(ids)
                continue
            videos[row["iso3"]].update(ids)
            # start_time holds one list of segment starts per video.
            segments[row["iso3"]] += len(re.findall(r"[-\d.]+", row["start_time"]))
    placed = set().union(*videos.values())
    counts = Counter({iso3: len(ids) for iso3, ids in videos.items()})
    return counts, segments, len(placed), len(unresolved - placed)


def share(count: float, total: float) -> float:
    return count / total if total else 0.0


def analyse(name: str, mapping: Path, reference: dict[str, dict], out: Path) -> dict:
    videos, segments, placed, unresolved = count_mapping(mapping)
    death_column = f"road_deaths_{YEAR}"
    deaths = {iso3: float(row[death_column]) for iso3, row in reference.items()}
    outside = sorted(set(videos) - set(deaths))
    total_videos = sum(videos[iso3] for iso3 in deaths)
    total_deaths = sum(deaths.values())
    total_segments = sum(segments[iso3] for iso3 in deaths)

    rows = []
    for iso3, ref in reference.items():
        video_share = share(videos.get(iso3, 0), total_videos)
        death_share = share(deaths[iso3], total_deaths)
        rows.append({
            "iso3": iso3,
            "country": ref["country"],
            "continent": ref["continent"],
            "videos": videos.get(iso3, 0),
            "segments": segments.get(iso3, 0),
            "road_deaths": int(deaths[iso3]),
            "video_share": round(video_share, 5),
            "segment_share": round(share(segments.get(iso3, 0), total_segments), 5),
            "death_share": round(death_share, 5),
            "representation": round(video_share / death_share, 3) if death_share else "",
            "videos_per_10k_deaths": round(videos.get(iso3, 0) / deaths[iso3] * 10_000, 2)
            if deaths[iso3] else "",
        })
    rows.sort(key=lambda row: (-row["videos"], -row["road_deaths"]))

    by_continent: dict[str, dict] = defaultdict(
        lambda: {"videos": 0, "segments": 0, "road_deaths": 0, "countries": 0, "covered": 0})
    for row in rows:
        group = by_continent[row["continent"] or "Other"]
        group["videos"] += row["videos"]
        group["segments"] += row["segments"]
        group["road_deaths"] += row["road_deaths"]
        group["countries"] += 1
        group["covered"] += row["videos"] > 0
    continents = []
    for continent, group in sorted(by_continent.items(), key=lambda item: -item[1]["road_deaths"]):
        video_share = share(group["videos"], total_videos)
        death_share = share(group["road_deaths"], total_deaths)
        continents.append({
            "continent": continent, **group,
            "video_share": round(video_share, 4),
            "segment_share": round(share(group["segments"], total_segments), 4),
            "death_share": round(death_share, 4),
            "representation": round(video_share / death_share, 3) if death_share else "",
        })

    covered = [row for row in rows if row["videos"]]
    rho, p_value = spearmanr([row["videos"] for row in rows], [row["road_deaths"] for row in rows])
    summary = {
        "dataset": name,
        "mapping": str(mapping),
        "reference_year": YEAR,
        "videos_placed": placed,
        "videos_unresolved": unresolved,
        "video_country_pairs": total_videos,
        "segments_placed": total_segments,
        "videos_outside_reference": {iso3: videos[iso3] for iso3 in outside},
        "countries_in_reference": len(rows),
        "countries_covered": len(covered),
        "countries_with_10_videos": sum(row["videos"] >= 10 for row in rows),
        "death_share_in_covered_countries": round(share(sum(r["road_deaths"] for r in covered), total_deaths), 4),
        "death_share_in_countries_with_10_videos": round(
            share(sum(r["road_deaths"] for r in rows if r["videos"] >= 10), total_deaths), 4),
        "video_share_top5_countries": round(share(sum(r["videos"] for r in rows[:5]), total_videos), 4),
        "spearman_videos_vs_deaths": {"rho": round(float(rho), 3), "p": float(p_value)},
        "most_under_represented": [
            {k: r[k] for k in ("country", "videos", "road_deaths", "death_share")}
            for r in sorted(rows, key=lambda r: -r["road_deaths"])
            if r["death_share"] >= 0.005 and r["representation"] < 0.25
        ],
        "continents": continents,
    }

    out.mkdir(parents=True, exist_ok=True)
    for file_name, table in ((f"coverage_{name}_by_country.csv", rows), (f"coverage_{name}_by_continent.csv", continents)):
        with (out / file_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (out / f"coverage_{name}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot_continents(name, continents, out)
    plot_countries(name, rows, out)
    print_summary(summary)
    return summary


def save(fig: go.Figure, path: Path, width: int, height: int) -> None:
    fig.update_layout(
        template="plotly_white", width=width, height=height,
        font=dict(family="Arial", size=14, color="#1f1f1e"),
        margin=dict(l=70, r=30, t=40, b=60),
    )
    fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")
    for suffix in (".png", ".pdf"):
        try:
            fig.write_image(path.with_suffix(suffix), width=width, height=height, scale=2 if suffix == ".png" else 1)
        except Exception as exc:  # kaleido needs a browser; HTML still exists
            print(f"Warning: could not write {path.with_suffix(suffix).name}: {exc}")


def plot_continents(name: str, continents: list[dict], out: Path) -> None:
    names = [c["continent"] for c in continents if c["continent"] != "Other"]
    data = {c["continent"]: c for c in continents}
    fig = go.Figure()
    for key, label in (("death_share", "Share of world road deaths (WHO 2021)"), ("video_share", f"Share of {name} videos")):
        colour = SERIES_COLOURS["deaths" if key == "death_share" else "videos"]
        fig.add_bar(
            name=label, x=names, y=[data[n][key] * 100 for n in names],
            marker=dict(color=colour, line=dict(color="white", width=2)),
            text=[f"{data[n][key] * 100:.{0 if data[n][key] >= 0.01 else 1}f}%" for n in names],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>",
        )
    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.05,
        yaxis=dict(title="Share (%)", gridcolor="#e6e5e0", rangemode="tozero"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    save(fig, out / f"coverage_{name}_by_continent", 1000, 520)


def plot_countries(name: str, rows: list[dict], out: Path) -> None:
    covered = [r for r in rows if r["videos"] and r["road_deaths"]]
    missing = [r for r in rows if not r["videos"] and r["road_deaths"]]
    # The largest by road deaths and by videos, placed so the labels do not collide.
    label = {r["iso3"] for r in sorted(rows, key=lambda r: -r["road_deaths"])[:8]}
    label |= {r["iso3"] for r in rows[:6]}
    position = {"CAN": "bottom center", "IDN": "bottom center", "FRA": "top right", "ITA": "top left",
                "IND": "top left", "CHN": "bottom center", "GBR": "top left", "DEU": "top right"}
    floor = 0.6  # where countries without videos sit on the log axis
    fig = go.Figure()
    fig.add_scatter(
        name="Countries without videos", mode="markers",
        x=[r["road_deaths"] for r in missing], y=[floor] * len(missing),
        marker=dict(size=8, color="#b0aea5", line=dict(color="white", width=1)),
        customdata=[[r["country"]] for r in missing],
        hovertemplate="%{customdata[0]}<br>%{x:,} road deaths<br>no videos<extra></extra>",
    )
    fig.add_scatter(
        name="Countries with videos", mode="markers+text",
        x=[r["road_deaths"] for r in covered], y=[r["videos"] for r in covered],
        text=[r["iso3"] if r["iso3"] in label else "" for r in covered],
        textposition=[position.get(r["iso3"], "top center") for r in covered],
        textfont=dict(size=11, color="#5f5e58"),
        marker=dict(size=9, color=SERIES_COLOURS["videos"], line=dict(color="white", width=1)),
        customdata=[[r["country"], r["representation"]] for r in covered],
        hovertemplate="%{customdata[0]}<br>%{x:,} road deaths<br>%{y} videos"
                      "<br>representation %{customdata[1]}<extra></extra>",
    )
    total_videos = sum(r["videos"] for r in rows)
    total_deaths = sum(r["road_deaths"] for r in rows)
    # The proportional line starts where it reaches one video.
    xs = [total_deaths / total_videos, max(r["road_deaths"] for r in rows)]
    fig.add_scatter(
        name="Proportional to road deaths", mode="lines",
        x=xs, y=[x * total_videos / total_deaths for x in xs],
        line=dict(color="#5f5e58", width=2, dash="dash"), hoverinfo="skip",
    )
    fig.update_layout(
        xaxis=dict(title=f"Road traffic deaths per year (WHO {YEAR}, log)", type="log", gridcolor="#e6e5e0"),
        yaxis=dict(title=f"{name.capitalize()} videos (log; grey = none)", type="log", gridcolor="#e6e5e0",
                   range=[-0.35, 3.1]),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    save(fig, out / f"coverage_{name}_by_country", 1000, 650)


def print_summary(summary: dict) -> None:
    s = summary
    print(f"== {s['dataset']}: {s['videos_placed']:,} videos placed in a country ({s['video_country_pairs']:,} "
          f"video-country pairs, {s['segments_placed']:,} segments), {s['videos_unresolved']:,} with no country")
    print(f"Countries covered: {s['countries_covered']} of {s['countries_in_reference']} "
          f"({s['countries_with_10_videos']} with 10+ videos)")
    print(f"World road deaths in covered countries: {s['death_share_in_covered_countries']:.0%}; "
          f"in countries with 10+ videos: {s['death_share_in_countries_with_10_videos']:.0%}")
    print(f"Top 5 countries hold {s['video_share_top5_countries']:.0%} of the videos; "
          f"Spearman videos vs deaths rho={s['spearman_videos_vs_deaths']['rho']}")
    if s["videos_outside_reference"]:
        print(f"Not in the WHO table: {s['videos_outside_reference']}")
    print(f"{'continent':15} {'videos':>7} {'segm.':>7} {'deaths':>7} {'index':>6} {'covered':>8}")
    for c in s["continents"]:
        print(f"{c['continent']:15} {c['video_share']:7.1%} {c['segment_share']:7.1%} {c['death_share']:7.1%} "
              f"{c['representation']:6} {c['covered']:>3}/{c['countries']}")
    print("Large road-death countries with under a quarter of their share: "
          + ", ".join(f"{r['country']} ({r['videos']})" for r in s["most_under_represented"]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crash-mapping", type=Path, default=HERE / "mapping.csv")
    parser.add_argument("--walking-mapping", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "_output" / "coverage")
    args = parser.parse_args()

    if args.refresh_reference or not args.reference.exists():
        refresh_reference(args.reference)
    reference = read_reference(args.reference)
    analyse("crash", args.crash_mapping, reference, args.out)
    if args.walking_mapping:
        analyse("walking", args.walking_mapping, reference, args.out)
    print(f"Tables and figures: {args.out}")


if __name__ == "__main__":
    main()
