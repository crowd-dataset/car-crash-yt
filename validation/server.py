"""Serve the two-reviewer validation page and store the answers.

Each reviewer gets one half of sample.json, so nobody labels a clip twice.
Progress is saved after every answer, and a reviewer who comes back with the
same name continues at the first clip they have not answered. Clips play from
YouTube; one that does not play moves to the end of that reviewer's queue while
just that snippet is downloaded with yt-dlp in the background (or, if that
fails, cut from a local copy in --video-dir). The snippet is deleted as soon as
the clip is answered.

    python validation/server.py                  # http://localhost:8765
    python validation/server.py --host 0.0.0.0   # reachable from other machines
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
SLOTS = 2
DOWNLOAD_TIMEOUT = 15 * 60
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm")

LOCK = threading.RLock()
SAMPLE: dict = {}
STATE: dict = {}
OPTIONS: argparse.Namespace
POOL: ThreadPoolExecutor


# --- state -----------------------------------------------------------------

def state_path() -> Path:
    return OPTIONS.data_dir / "state.json"


def snippet_path(item_id: str) -> Path:
    return OPTIONS.data_dir / "snippets" / f"{item_id}.mp4"


def save_state() -> None:
    path = state_path()
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(STATE, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def load_state() -> None:
    STATE.clear()
    if state_path().is_file():
        STATE.update(json.loads(state_path().read_text(encoding="utf-8")))
    STATE.setdefault("reviewers", [None] * SLOTS)
    STATE.setdefault("downloads", {})


def name_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def reviewer_slot(name: str) -> int | None:
    for slot, reviewer in enumerate(STATE["reviewers"]):
        if reviewer and reviewer["key"] == name_key(name):
            return slot
    return None


def is_closed(reviewer: dict, item_id: str) -> bool:
    return item_id in reviewer["answers"] or item_id in reviewer["unavailable"]


# --- snippets ---------------------------------------------------------------

def find_local_video(video_id: str) -> Path | None:
    for directory in OPTIONS.video_dir:
        for suffix in VIDEO_SUFFIXES:
            candidate = Path(directory) / f"{video_id}{suffix}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def yt_dlp_command() -> list[str]:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    if importlib.util.find_spec("yt_dlp"):
        return [sys.executable, "-m", "yt_dlp"]
    raise RuntimeError("yt-dlp is not installed")


def fetch_snippet(item_id: str) -> None:
    item = SAMPLE["items"][item_id]
    target = snippet_path(item_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".part.mp4")
    partial.unlink(missing_ok=True)
    start, end = item["play_start"], item["play_end"]
    # Only the clip's own seconds are fetched from YouTube; a local copy of the
    # full video is the fallback.
    attempts = []
    try:
        command = yt_dlp_command() + [
            "--quiet", "--no-warnings", "--no-playlist",
            "--extractor-args", "youtube:player_client=default,web_embedded,mweb",
            "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/bv*+ba/b",
            "--download-sections", f"*{start:.3f}-{end:.3f}",
            "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
            "-o", str(partial),
        ]
        if OPTIONS.cookies:
            command += ["--cookies", str(OPTIONS.cookies)]
        command.append(f"https://www.youtube.com/watch?v={item['video_id']}")
        attempts.append(("yt-dlp", command))
    except RuntimeError as error:
        outcome = {"status": "failed", "error": str(error)}
    local = find_local_video(item["video_id"])
    if local:
        attempts.append(("local", [
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}", "-i", str(local), "-vf", "scale=-2:'min(720,ih)'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-movflags", "+faststart", str(partial),
        ]))
    errors = []
    for source, command in attempts:
        partial.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT
            )
            if result.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError((result.stderr or result.stdout).strip()[-400:] or "no output file")
            os.replace(partial, target)
            outcome = {"status": "ready", "source": source}
            break
        except Exception as error:  # noqa: BLE001 - every failure ends the attempt
            partial.unlink(missing_ok=True)
            errors.append(f"{source}: {error}")
    else:
        if errors:
            outcome = {"status": "failed", "error": " | ".join(errors)[-800:]}
    with LOCK:
        STATE["downloads"][item_id] = {**outcome, "finished": time.time()}
        save_state()


def drop_snippet(item_id: str) -> None:
    """Delete a clip's downloaded snippet once it is no longer needed."""
    download = STATE["downloads"].get(item_id)
    if download and download.get("status") == "ready":
        snippet_path(item_id).unlink(missing_ok=True)
        download["status"] = "deleted"


def request_snippet(item_id: str) -> None:
    """Start a background fetch unless one is running or already finished."""
    status = STATE["downloads"].get(item_id, {}).get("status")
    if status == "pending" or (status == "ready" and snippet_path(item_id).is_file()):
        return
    STATE["downloads"][item_id] = {"status": "pending", "started": time.time()}
    POOL.submit(fetch_snippet, item_id)


# --- queue ------------------------------------------------------------------

def client_item(reviewer: dict, item_id: str) -> dict:
    item = SAMPLE["items"][item_id]
    deferred = item_id in reviewer["deferred"]
    download = STATE["downloads"].get(item_id, {})
    local = deferred and download.get("status") == "ready" and snippet_path(item_id).is_file()
    queue = reviewer["queue"]
    return {
        "id": item_id,
        "dataset": item["dataset"],
        "video_id": item["video_id"],
        "play_start": item["play_start"],
        "play_end": item["play_end"],
        "location": item["location"],
        "source": "local" if local else "youtube",
        "media_url": f"/media/{item_id}.mp4" if local else None,
        "position": queue.index(item_id) + 1,
        "answers": reviewer["answers"].get(item_id, {}).get("values"),
        "note": reviewer["answers"].get(item_id, {}).get("note", ""),
    }


def progress(reviewer: dict) -> dict:
    closed = sum(is_closed(reviewer, item_id) for item_id in reviewer["queue"])
    return {"done": closed, "total": len(reviewer["queue"])}


def next_item(reviewer: dict) -> dict:
    waiting = 0
    for item_id in reviewer["queue"]:
        if is_closed(reviewer, item_id):
            continue
        if item_id in reviewer["deferred"]:
            download = STATE["downloads"].get(item_id, {})
            if download.get("status") == "failed":
                reviewer["unavailable"][item_id] = download.get("error", "download failed")
                save_state()
                continue
            if download.get("status") != "ready":
                waiting += 1
                continue
        return {"state": "item", "item": client_item(reviewer, item_id), **progress(reviewer)}
    if waiting:
        return {"state": "waiting", "pending": waiting, **progress(reviewer)}
    return {"state": "done", **progress(reviewer)}


# --- HTTP -------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "CrowdValidation/1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if OPTIONS.verbose:
            super().log_message(format, *args)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def fail(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def reviewer(self, slot_text) -> dict | None:
        try:
            slot = int(slot_text)
            return STATE["reviewers"][slot]
        except (TypeError, ValueError, IndexError):
            return None

    # GET ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path in {"/", "/index.html"}:
            return self.send_file(HERE / "index.html", "text/html; charset=utf-8")
        if url.path == "/questions.json":
            return self.send_file(HERE / "questions.json", "application/json")
        if url.path.startswith("/media/"):
            item_id = url.path[len("/media/"):].removesuffix(".mp4")
            if item_id not in SAMPLE["items"]:
                return self.fail("unknown clip", 404)
            return self.send_file(snippet_path(item_id), "video/mp4")
        with LOCK:
            if url.path == "/api/next":
                reviewer = self.reviewer(query.get("slot"))
                if not reviewer:
                    return self.fail("unknown reviewer", 404)
                return self.send_json(next_item(reviewer))
            if url.path == "/api/item":
                reviewer = self.reviewer(query.get("slot"))
                item_id = query.get("id")
                if not reviewer or item_id not in reviewer["queue"]:
                    return self.fail("unknown clip", 404)
                return self.send_json({"item": client_item(reviewer, item_id), **progress(reviewer)})
            if url.path == "/api/history":
                reviewer = self.reviewer(query.get("slot"))
                if not reviewer:
                    return self.fail("unknown reviewer", 404)
                return self.send_json({"answered": reviewer["answer_order"]})
            if url.path == "/api/status":
                return self.send_json({
                    "reviewers": [
                        reviewer and {"name": reviewer["name"], **progress(reviewer),
                                      "deferred": len(reviewer["deferred"]),
                                      "unavailable": len(reviewer["unavailable"])}
                        for reviewer in STATE["reviewers"]
                    ],
                    "downloads": {
                        status: sum(d["status"] == status for d in STATE["downloads"].values())
                        for status in ("pending", "ready", "failed")
                    },
                })
        self.fail("not found", 404)

    def send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            return self.fail("not found", 404)
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    # POST -----------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            body = self.read_json()
        except json.JSONDecodeError:
            return self.fail("bad JSON")
        with LOCK:
            if url.path == "/api/start":
                return self.start(body)
            reviewer = self.reviewer(body.get("slot"))
            item_id = body.get("id")
            if not reviewer or item_id not in reviewer["queue"]:
                return self.fail("unknown clip", 404)
            if url.path == "/api/answer":
                values = body.get("values")
                if not isinstance(values, dict) or not values:
                    return self.fail("no answers")
                previous = reviewer["answers"].get(item_id)
                reviewer["answers"][item_id] = {
                    "values": values,
                    "note": str(body.get("note") or "")[:2000],
                    "seconds": body.get("seconds"),
                    "source": body.get("source"),
                    "first_saved": previous["first_saved"] if previous else time.time(),
                    "saved": time.time(),
                }
                if item_id not in reviewer["answer_order"]:
                    reviewer["answer_order"].append(item_id)
                drop_snippet(item_id)
                save_state()
                return self.send_json(next_item(reviewer))
            if url.path == "/api/defer":
                if is_closed(reviewer, item_id):
                    return self.fail("clip already answered")
                if item_id in reviewer["deferred"] and \
                        STATE["downloads"].get(item_id, {}).get("status") == "ready":
                    # The downloaded snippet does not play either.
                    reviewer["unavailable"][item_id] = "local snippet did not play"
                    drop_snippet(item_id)
                else:
                    reviewer["deferred"][item_id] = str(body.get("reason") or "")[:200]
                    reviewer["queue"].remove(item_id)
                    reviewer["queue"].append(item_id)
                    request_snippet(item_id)
                save_state()
                return self.send_json(next_item(reviewer))
        self.fail("not found", 404)

    def start(self, body: dict) -> None:
        name = " ".join(str(body.get("name") or "").split())
        if not name:
            return self.fail("Please type your name.")
        slot = reviewer_slot(name)
        resumed = slot is not None
        if slot is None:
            free = [s for s, reviewer in enumerate(STATE["reviewers"]) if reviewer is None]
            if not free:
                names = " and ".join(r["name"] for r in STATE["reviewers"])
                return self.fail(
                    f"Both reviewer places are taken ({names}). "
                    "Type one of these names to continue."
                )
            slot = free[0]
            STATE["reviewers"][slot] = {
                "name": name,
                "key": name_key(name),
                "created": time.time(),
                "queue": list(SAMPLE["queues"][slot]),
                "answers": {},
                "answer_order": [],
                "deferred": {},
                "unavailable": {},
            }
            save_state()
        self.send_json({"slot": slot, "name": STATE["reviewers"][slot]["name"],
                        "resumed": resumed, **progress(STATE["reviewers"][slot])})


def main() -> None:
    global OPTIONS, POOL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sample", type=Path, default=HERE / "sample.json")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--video-dir", action="append", default=[],
                        help="folder of full videos named <video_id>.mp4 (repeatable)")
    parser.add_argument("--cookies", type=Path, help="cookies.txt for yt-dlp (optional)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true",
                        help="fetch clips again whose download failed before")
    parser.add_argument("--verbose", action="store_true")
    OPTIONS = parser.parse_args()

    SAMPLE.update(json.loads(OPTIONS.sample.read_text(encoding="utf-8")))
    OPTIONS.data_dir.mkdir(parents=True, exist_ok=True)
    POOL = ThreadPoolExecutor(max_workers=OPTIONS.workers)
    with LOCK:
        load_state()
        for reviewer in STATE["reviewers"]:
            if not reviewer:
                continue
            for item_id in reviewer["deferred"]:
                if OPTIONS.retry_failed and item_id in reviewer["unavailable"] \
                        and item_id not in reviewer["answers"]:
                    del reviewer["unavailable"][item_id]
                    STATE["downloads"].pop(item_id, None)
                if is_closed(reviewer, item_id):
                    continue
                # Restart fetches that were running when the server stopped.
                download = STATE["downloads"].get(item_id, {})
                if download.get("status") == "failed" and not OPTIONS.retry_failed:
                    continue
                if download.get("status") != "ready" or not snippet_path(item_id).is_file():
                    STATE["downloads"].pop(item_id, None)
                    request_snippet(item_id)
        save_state()

    server = ThreadingHTTPServer((OPTIONS.host, OPTIONS.port), Handler)
    print(f"Validation page: http://{'localhost' if OPTIONS.host in {'127.0.0.1', '0.0.0.0'} else OPTIONS.host}:{OPTIONS.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
