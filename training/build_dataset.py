"""Mine indoor and warehouse3, label with Cosmos Reason, featurize YOLO boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
LABEL_DIR = OUT / "labels_raw"
FEATURE_DIR = OUT / "features"

PROMPT_VERSION = "v5"
FEATURE_VERSION = "v1"
LOCATIONS = ("indoor", "warehouse3")
MIN_CONF = 0.3
MACHINE_LABELS = {"humanoid", "agv", "forklift", "other", "none"}
DECISIONS = {"yield", "proceed"}

QUERIES = (
    ("scenario_person_walking_near_moving_robot", "person walking near a moving robot"),
    ("scenario_robot_or_agv_carrying_pallet", "robot or AGV carrying a pallet"),
    ("scenario_forklift_near_person", "forklift near a person"),
    ("scenario_empty_aisle_no_people", "empty aisle with no people"),
)
FEATURE_NAMES = (
    "has_detections",
    "max_persons",
    "max_machines",
    "co_presence",
    "min_gap",
    "gap_p10",
    "closing_rate",
    "max_iou",
    "machine_motion",
    "person_area_max",
) + tuple(name for name, _query in QUERIES)

SYSTEM_PROMPT = """Label captions headed Clip 1, Clip 2, and so on.
Return ONLY a JSON array, one object per clip, in order. No markdown.
{"clip": 1, "machine": "humanoid"|"agv"|"forklift"|"other"|"none", "machine_moving": false, "person_present": true, "person_in_path": false, "decision": "yield"|"proceed", "reason": "one short sentence"}
JSON true/false. agv = AGV, AMR, or pallet jack. forklift = forklift. humanoid = a humanoid robot.

yield when a person is in or about to enter the path of a moving machine, or within arm's reach of a machine.
Arm's reach includes near, beside, approaching, touching, climbing on, or operating that machine, even if it is stopped.
On the indoor shared floor, people plus a moving AGV, pallet jack, or humanoid are about to enter its path: yield, unless the caption says the people are far from it or already clear.
A wide indoor shot is still yield when a moving machine and people share the floor.
proceed when there is no person, no machine, or the person is only far across the room.
If the person is beside, near, next to, touching, climbing on, operating, or interacting with a machine at any point in the clip, the decision is yield. Do not mark those clips proceed.
"""

QUESTION_ALL = (
    "Label every clip. Beside, near, touching, climbing on, operating, or interacting with a machine is always yield, even if the machine is stopped. "
    "A moving AGV or humanoid sharing the indoor floor with people is yield. "
    "Proceed only when the person is far from every machine or no machine is present. Return only the JSON array."
)


class Client:
    def __init__(self) -> None:
        self.base = (os.environ.get("INGRESS_URL") or "").rstrip("/")
        self.user = os.environ.get("USERNAME") or ""
        self.password = os.environ.get("PASSWORD") or ""
        if not self.base or not self.user or not self.password:
            raise SystemExit("INGRESS_URL, USERNAME, and PASSWORD must be set")
        self.token = ""
        self.lock = threading.Lock()

    def login(self, force: bool = False) -> str:
        with self.lock:
            if self.token and not force:
                return self.token
            status, parsed, _raw = _http(
                "POST",
                self.base + "/api/v1/auth/login",
                {"username": self.user, "password": self.password},
                timeout=30,
            )
            if status != 200 or not isinstance(parsed, dict) or not parsed.get("access_token"):
                raise SystemExit(f"login failed status={status}")
            self.token = str(parsed["access_token"])
            return self.token

    def call(self, method: str, path: str, payload=None, timeout: int = 60):
        last_status = 0
        last_parsed = None
        for attempt in range(6):
            token = self.login(force=False)
            headers = {"Authorization": f"Bearer {token}"}
            status, parsed, raw = _http(method, self.base + path, payload, headers, timeout=timeout)
            last_status, last_parsed = status, parsed
            if status == 401:
                self.login(force=True)
                continue
            if status == 0 or status in {429, 500, 502, 503, 504}:
                time.sleep(2.0 * (attempt + 1))
                continue
            return status, parsed, raw
        return last_status, last_parsed, b""


def _http(method: str, url: str, payload=None, headers=None, timeout: int = 60):
    data = None if payload is None else json.dumps(payload).encode()
    hdrs = {} if headers is None else dict(headers)
    if payload is not None:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, _parse_json(raw), raw
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp else b""
        return exc.code, _parse_json(raw), raw
    except urllib.error.URLError:
        return 0, None, b""


def _parse_json(raw: bytes):
    try:
        return json.loads(raw.decode() or "null")
    except json.JSONDecodeError:
        return None


def _num(value, places: int = 6) -> float:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError("non-finite feature")
    return round(number, places)


def edge_gap(a, b) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def box_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def percentile_10(values: list[float]) -> float:
    if not values:
        return 1.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = 0.10 * (len(xs) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    weight = k - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / var_x


def neutral_features() -> dict:
    features = {
        "has_detections": 0,
        "max_persons": 0,
        "max_machines": 0,
        "co_presence": 0.0,
        "min_gap": 1.0,
        "gap_p10": 1.0,
        "closing_rate": 0.0,
        "max_iou": 0.0,
        "machine_motion": 0.0,
        "person_area_max": 0.0,
    }
    for name, _query in QUERIES:
        features[name] = 0.0
    return features


def _box(det: dict):
    bbox = det.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        conf = float(det.get("confidence") if det.get("confidence") is not None else 0.0)
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    label = str(det.get("label") or "").strip().lower()
    return label, conf, [x1, y1, x2, y2]


def _frame_area(shape, video_shape):
    for candidate in (shape, video_shape):
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
            try:
                a = float(candidate[0])
                b = float(candidate[1])
            except (TypeError, ValueError):
                continue
            if a > 0 and b > 0:
                return math.hypot(a, b), a * b
    return math.hypot(1080.0, 1920.0), 1080.0 * 1920.0


def features_from_detections(payload) -> dict:
    features = neutral_features()
    if not isinstance(payload, dict):
        return features
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return features
    features["has_detections"] = 1
    video_shape = payload.get("video_shape")
    max_persons = 0
    max_machines = 0
    co_frames = 0
    frame_gaps = []
    gap_times = []
    max_iou = 0.0
    person_area_max = 0.0
    motion = 0.0
    prev_centroid = None
    usable = 0
    ordered = sorted(
        [frame for frame in frames if isinstance(frame, dict)],
        key=lambda frame: (float(frame.get("time_sec") or 0.0), int(frame.get("frame_index") or 0)),
    )
    for frame in ordered:
        usable += 1
        diag, area = _frame_area(frame.get("shape"), video_shape)
        persons = []
        machines = []
        for det in frame.get("detections") or []:
            if not isinstance(det, dict):
                continue
            parsed = _box(det)
            if parsed is None or parsed[1] < MIN_CONF:
                continue
            label, _conf, box = parsed
            if label == "person":
                persons.append(box)
            elif label:
                machines.append(box)
        max_persons = max(max_persons, len(persons))
        max_machines = max(max_machines, len(machines))
        for box in persons:
            box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            person_area_max = max(person_area_max, box_area / area)
        if persons and machines:
            co_frames += 1
            gaps = [edge_gap(person, machine) / diag for person in persons for machine in machines]
            frame_gaps.append(min(1.0, max(0.0, min(gaps))))
            gap_times.append(float(frame.get("time_sec") or 0.0))
            for person in persons:
                for machine in machines:
                    max_iou = max(max_iou, box_iou(person, machine))
        if machines:
            largest = max(machines, key=lambda box: max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
            centroid = ((largest[0] + largest[2]) / 2.0, (largest[1] + largest[3]) / 2.0)
            if prev_centroid is not None:
                shift = math.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1]) / diag
                motion = max(motion, shift)
            prev_centroid = centroid
        else:
            prev_centroid = None
    if usable == 0:
        return neutral_features()
    features["max_persons"] = max_persons
    features["max_machines"] = max_machines
    features["co_presence"] = _num(co_frames / usable)
    if frame_gaps:
        features["min_gap"] = _num(min(frame_gaps))
        features["gap_p10"] = _num(min(1.0, max(0.0, percentile_10(frame_gaps))))
        features["closing_rate"] = _num(slope(gap_times, frame_gaps))
    features["max_iou"] = _num(min(1.0, max(0.0, max_iou)))
    features["machine_motion"] = _num(max(0.0, motion))
    features["person_area_max"] = _num(min(1.0, max(0.0, person_area_max)))
    return features


def self_check() -> None:
    assert edge_gap([0, 0, 10, 10], [5, 5, 15, 15]) == 0.0
    assert abs(edge_gap([0, 0, 10, 10], [20, 0, 30, 10]) - 10) < 1e-9
    assert abs(edge_gap([0, 0, 10, 10], [13, 14, 20, 20]) - 5) < 1e-9
    payload = {
        "video_shape": [100, 100],
        "frames": [
            {
                "frame_index": 0,
                "time_sec": 0.0,
                "shape": [100, 100],
                "detections": [
                    {"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                    {"label": "truck", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                ],
            },
            {
                "frame_index": 1,
                "time_sec": 1.0,
                "shape": [100, 100],
                "detections": [
                    {"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                    {"label": "truck", "confidence": 0.2, "bbox": [0, 0, 10, 10]},
                ],
            },
        ],
    }
    features = features_from_detections(payload)
    assert features["has_detections"] == 1
    assert features["min_gap"] == 0.0
    assert features["max_iou"] == 1.0
    assert features["co_presence"] == 0.5
    assert features["max_machines"] == 1
    empty = features_from_detections(None)
    assert empty["has_detections"] == 0 and empty["min_gap"] == 1.0
    assert all(empty[name] is not None for name in FEATURE_NAMES)


def explore_segments(client: Client) -> list[dict]:
    segments = []
    seen = set()
    for location in LOCATIONS:
        offset = 0
        total = None
        fetched = 0
        while True:
            status, parsed, _raw = client.call(
                "GET",
                f"/api/v1/videos/explore?location={urllib.parse.quote(location)}&limit=100&offset={offset}",
                timeout=90,
            )
            if status != 200 or not isinstance(parsed, dict):
                raise SystemExit(f"explore failed location={location} status={status}")
            chunks = parsed.get("chunks") or []
            total = int(parsed.get("total") or 0)
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                timeline = [item for item in (chunk.get("timeline") or []) if isinstance(item, dict)]
                expected = int(chunk.get("total_segments") or len(timeline))
                if len(timeline) < expected and chunk.get("original_video"):
                    timeline = _segments_from_tools(client, chunk) or timeline
                for item in timeline:
                    source = str(item.get("source") or "")
                    if not source or source in seen:
                        continue
                    seen.add(source)
                    segments.append(
                        {
                            "source": source,
                            "original_video": str(item.get("original_video") or chunk.get("original_video") or ""),
                            "segment_number": int(item.get("segment_number") or 0),
                            "segment_start_sec": float(item.get("segment_start_sec") or 0.0),
                            "segment_end_sec": float(item.get("segment_end_sec") or 0.0),
                            "location": str(item.get("location") or chunk.get("location") or location),
                            "camera_id": str(item.get("camera_id") or chunk.get("camera_id") or ""),
                            "reasoning_content": str(item.get("reasoning_content") or ""),
                        }
                    )
            fetched += len(chunks)
            offset += len(chunks)
            if not chunks or offset >= total:
                break
        print(f"explore {location} chunks={fetched} total_field={total}", flush=True)
    return segments


def _segments_from_tools(client: Client, chunk: dict) -> list[dict]:
    quoted = urllib.parse.quote(str(chunk.get("original_video") or ""), safe="")
    status, parsed, _raw = client.call("GET", f"/api/v1/tools/segments?original_video={quoted}", timeout=90)
    if status != 200 or not isinstance(parsed, dict):
        return []
    rows = [row for row in (parsed.get("segments") or []) if isinstance(row, dict)]
    return sorted(rows, key=lambda row: int(row.get("segment_number") or 0))


def scenario_scores(client: Client, segments: list[dict]) -> dict[str, dict[str, float]]:
    by_source = {row["source"]: {} for row in segments}
    by_key = {}
    for row in segments:
        by_key[(row["original_video"], row["segment_number"])] = row["source"]
    for feature, query in QUERIES:
        hits = 0
        for location in LOCATIONS:
            status, parsed, _raw = client.call(
                "POST",
                "/api/v1/search",
                {
                    "query": query,
                    "top_k": 50,
                    "min_similarity": 0.1,
                    "metadata_filters": {"location": location},
                    "include_public": True,
                },
                timeout=180,
            )
            if status != 200 or not isinstance(parsed, dict):
                raise SystemExit(f"search failed status={status} location={location}")
            for result in parsed.get("results") or []:
                if not isinstance(result, dict):
                    continue
                try:
                    score = float(result.get("similarity_score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                source = str(result.get("source") or "")
                if source not in by_source:
                    source = by_key.get((str(result.get("original_video") or ""), int(result.get("segment_number") or 0)), "")
                if not source or source not in by_source:
                    continue
                previous = by_source[source].get(feature, 0.0)
                if score > previous:
                    by_source[source][feature] = score
                    hits += 1
        print(f"search {feature} matched_rows={sum(1 for scores in by_source.values() if feature in scores)}", flush=True)
    return by_source


def extract_array(text: str):
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return data


def _clip_number(value, n: int):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1 <= value <= n:
        return value
    if isinstance(value, float) and value.is_integer() and 1 <= int(value) <= n:
        return int(value)
    return None


def valid_label(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("machine") not in MACHINE_LABELS:
        return False
    if obj.get("decision") not in DECISIONS:
        return False
    for key in ("machine_moving", "person_present", "person_in_path"):
        if not isinstance(obj.get(key), bool):
            return False
    reason = obj.get("reason")
    return isinstance(reason, str) and bool(reason.strip())


def assign_labels(segments: list[dict], objects) -> dict[int, dict | None]:
    ordered = sorted(segments, key=lambda row: (row["segment_number"], row["segment_start_sec"]))
    n = len(ordered)
    assigned: dict[int, dict | None] = {}
    if not isinstance(objects, list):
        return assigned
    numbered = []
    for index, obj in enumerate(objects):
        clip = _clip_number(obj.get("clip") if isinstance(obj, dict) else None, n)
        if clip is None:
            clip = index + 1 if index < n else None
        numbered.append((clip, obj))
    for clip, obj in numbered:
        if clip is None or clip in assigned:
            continue
        assigned[clip] = obj if valid_label(obj) else None
    return assigned


def synthesize(client: Client, original_video: str, question: str, max_segments: int) -> str:
    status, parsed, _raw = client.call(
        "POST",
        "/api/v1/videos/synthesize",
        {
            "original_video": original_video,
            "question": question,
            "max_segments": max_segments,
            "system_prompt": SYSTEM_PROMPT,
        },
        timeout=180,
    )
    if status != 200 or not isinstance(parsed, dict):
        print(f"synthesize status={status}", flush=True)
        return ""
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        synthesis = parsed.get("llm_synthesis")
        if isinstance(synthesis, dict) and isinstance(synthesis.get("response"), str):
            answer = synthesis["response"]
        else:
            answer = ""
    return answer


def label_parent(client: Client, original_video: str, segments: list[dict], relabel: bool) -> dict[int, dict | None]:
    ordered = sorted(segments, key=lambda row: (row["segment_number"], row["segment_start_sec"]))
    n = len(ordered)
    digest = hashlib.sha256(f"{PROMPT_VERSION}\n{original_video}".encode()).hexdigest()[:24]
    path = LABEL_DIR / f"{digest}.json"
    cached = None
    if path.exists() and not relabel:
        try:
            cached = json.loads(path.read_text())
        except json.JSONDecodeError:
            cached = None
        if not isinstance(cached, dict) or cached.get("prompt_version") != PROMPT_VERSION:
            cached = None
    responses = list(cached.get("responses") or []) if cached else []
    if not responses:
        answer = synthesize(client, original_video, QUESTION_ALL, min(n, 100))
        responses.append({"kind": "full", "answer": answer})
        assigned = assign_labels(ordered, extract_array(answer))
        if len(assigned) < n and n > 1:
            mid = max(1, n // 2)
            batches = ((1, mid), (mid + 1, n)) if mid < n else ((1, n),)
            for start, end in batches:
                if start > end:
                    continue
                question = (
                    f"Label only Clip {start} through Clip {end} inclusive. "
                    "Return one JSON object per requested clip and no others."
                )
                batch_answer = synthesize(client, original_video, question, min(end, 100))
                responses.append({"kind": "batch", "start": start, "end": end, "answer": batch_answer})
        path.write_text(
            json.dumps({"prompt_version": PROMPT_VERSION, "n_segments": n, "responses": responses})
        )
    assigned = {}
    for response in responses:
        part = assign_labels(ordered, extract_array(response.get("answer") or ""))
        if response.get("kind") == "batch":
            start = int(response.get("start") or 1)
            end = int(response.get("end") or n)
            part = {clip: obj for clip, obj in part.items() if start <= clip <= end}
        for clip, obj in part.items():
            if clip not in assigned or (assigned[clip] is None and obj is not None):
                assigned[clip] = obj
    return assigned


def apply_label(row: dict, obj) -> None:
    if not valid_label(obj):
        row["label_status"] = "unparsed"
        row["machine"] = ""
        row["machine_moving"] = False
        row["person_present"] = False
        row["person_in_path"] = False
        row["decision"] = ""
        row["reason"] = ""
        return
    row["label_status"] = "ok"
    row["machine"] = obj["machine"]
    row["machine_moving"] = obj["machine_moving"]
    row["person_present"] = obj["person_present"]
    row["person_in_path"] = obj["person_in_path"]
    row["decision"] = obj["decision"]
    row["reason"] = obj["reason"].strip()


def feature_cache_path(source: str) -> Path:
    digest = hashlib.sha256(f"{FEATURE_VERSION}\n{source}".encode()).hexdigest()[:24]
    return FEATURE_DIR / f"{digest}.json"


def featurize(client: Client, row: dict) -> dict:
    path = feature_cache_path(row["source"])
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except json.JSONDecodeError:
            cached = None
        if isinstance(cached, dict) and cached.get("feature_version") == FEATURE_VERSION:
            features = cached.get("features")
            if isinstance(features, dict) and all(name in features and features[name] is not None for name in FEATURE_NAMES):
                return features
    quoted = urllib.parse.quote(row["source"], safe="")
    status, parsed, _raw = client.call("GET", f"/api/v1/videos/detections?source={quoted}", timeout=120)
    if status == 404:
        features = neutral_features()
        path.write_text(json.dumps({"feature_version": FEATURE_VERSION, "features": features}))
        return features
    if status != 200:
        print(f"detections status={status}", flush=True)
        return neutral_features()
    features = features_from_detections(parsed)
    path.write_text(json.dumps({"feature_version": FEATURE_VERSION, "features": features}))
    return features


def build(relabel: bool) -> None:
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    client = Client()
    client.login()
    seg_cache = OUT / "segments.json"
    score_cache = OUT / "scenario_scores.json"
    if seg_cache.exists():
        segments = json.loads(seg_cache.read_text())
        print(f"segments={len(segments)} cached", flush=True)
    else:
        segments = explore_segments(client)
        print(f"segments={len(segments)}", flush=True)
        seg_cache.write_text(json.dumps(segments))
    if score_cache.exists():
        loaded = json.loads(score_cache.read_text())
        scores = {source: {name: float(value) for name, value in feats.items()} for source, feats in loaded.items()}
        print("scenario scores cached", flush=True)
    else:
        scores = scenario_scores(client, segments)
        score_cache.write_text(json.dumps(scores))
    groups: dict[str, list[dict]] = {}
    for row in segments:
        groups.setdefault(row["original_video"], []).append(row)
    labels: dict[str, dict[int, dict | None]] = {}
    parents = sorted(groups)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(label_parent, client, parent, groups[parent], relabel): parent for parent in parents
        }
        done = 0
        for future in as_completed(futures):
            parent = futures[future]
            labels[parent] = future.result()
            done += 1
            if done % 5 == 0 or done == len(parents):
                print(f"labeled {done}/{len(parents)}", flush=True)
    feature_rows: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(featurize, client, row): row["source"] for row in segments}
        done = 0
        for future in as_completed(futures):
            source = futures[future]
            feature_rows[source] = future.result()
            done += 1
            if done % 40 == 0 or done == len(segments):
                print(f"featurized {done}/{len(segments)}", flush=True)
    dataset = []
    for row in sorted(segments, key=lambda item: (item["location"], item["original_video"], item["segment_number"])):
        ordered = sorted(groups[row["original_video"]], key=lambda item: (item["segment_number"], item["segment_start_sec"]))
        clip = ordered.index(row) + 1
        obj = labels.get(row["original_video"], {}).get(clip)
        record = dict(row)
        record["prompt_version"] = PROMPT_VERSION
        apply_label(record, obj)
        features = feature_rows[row["source"]]
        for name in FEATURE_NAMES:
            record[name] = features[name]
        record["scenario_scores"] = {query: _num(scores[row["source"]].get(feature, 0.0)) for feature, query in QUERIES}
        for feature, _query in QUERIES:
            record[feature] = record["scenario_scores"][_query]
        if any(record[name] is None for name in FEATURE_NAMES):
            raise SystemExit("null feature")
        dataset.append(record)
    destination = OUT / "dataset.jsonl"
    with destination.open("w") as handle:
        for record in dataset:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    _summarize(dataset)


def _summarize(dataset: list[dict]) -> None:
    by_location = {}
    status = {"ok": 0, "unparsed": 0}
    decisions = {"yield": 0, "proceed": 0}
    for row in dataset:
        by_location[row["location"]] = by_location.get(row["location"], 0) + 1
        status[row["label_status"]] = status.get(row["label_status"], 0) + 1
        if row["label_status"] == "ok":
            decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    labeled = status.get("ok", 0)
    print(f"rows={len(dataset)} by_location={by_location}", flush=True)
    print(f"label_status ok={status.get('ok', 0)} unparsed={status.get('unparsed', 0)}", flush=True)
    print(f"yield={decisions.get('yield', 0)} proceed={decisions.get('proceed', 0)} labeled={labeled}", flush=True)
    if labeled:
        print(
            f"yield_frac={decisions.get('yield', 0) / labeled:.3f} proceed_frac={decisions.get('proceed', 0) / labeled:.3f}",
            flush=True,
        )
    detected = [row for row in dataset if row["has_detections"] == 1]
    gaps = {round(row["min_gap"], 4) for row in detected}
    yield_gaps = [row["min_gap"] for row in detected if row["decision"] == "yield"]
    proceed_gaps = [row["min_gap"] for row in detected if row["decision"] == "proceed"]
    print(f"has_detections={len(detected)} min_gap_unique={len(gaps)}", flush=True)
    if yield_gaps and proceed_gaps:
        print(
            f"min_gap_mean yield={sum(yield_gaps) / len(yield_gaps):.4f} proceed={sum(proceed_gaps) / len(proceed_gaps):.4f}",
            flush=True,
        )
    shown = {"yield": 0, "proceed": 0}
    for row in dataset:
        if row["label_status"] != "ok" or shown.get(row["decision"], 2) >= 2:
            continue
        shown[row["decision"]] = shown.get(row["decision"], 0) + 1
        caption = " ".join(row["reasoning_content"].split())[:180]
        print(f"sample {row['location']} {row['decision']} {row['machine']} | {caption}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()
    build(args.relabel)


if __name__ == "__main__":
    main()
