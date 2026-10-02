"""Near-miss desk. Retrieval API for search, captions, and detections. Canary for speech. W&B for the verdict."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8080"))
VSS_URL = (os.environ.get("VSS_URL") or os.environ.get("INGRESS_URL") or "").rstrip("/")
VSS_USERNAME = os.environ.get("VSS_USERNAME") or os.environ.get("USERNAME") or ""
VSS_PASSWORD = os.environ.get("VSS_PASSWORD") or os.environ.get("PASSWORD") or ""
WANDB_API_KEY = os.environ.get("WANDB_API_KEY") or ""
WANDB_TEAM = os.environ.get("WANDB_TEAM") or ""
WANDB_PROJECT = os.environ.get("WANDB_PROJECT") or ""
GPU_BEARER_TOKEN = os.environ.get("GPU_BEARER_TOKEN") or ""
CANARY_URL = (os.environ.get("CANARY_1B_URL") or "http://166.19.38.112:8004").rstrip("/")
WANDB_INFERENCE = "https://api.inference.wandb.ai/v1"
WANDB_MODEL = os.environ.get("WANDB_INFERENCE_MODEL") or "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_QUESTION = "person close to a moving vehicle"
SHOWN_HITS = 8

_token_lock = threading.Lock()
_token = ""
_weave_ready = False
_weave_lock = threading.Lock()

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Near-miss desk</title>
<style>
  :root { color-scheme: dark; --bg: #12140f; --panel: #1c2118; --ink: #e7ecdf; --muted: #9aa592; --line: #343b2e; --hot: #e2b340; --ok: #8fb573; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
  header { padding: 20px 24px 8px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0; letter-spacing: -0.02em; }
  p.lede { color: var(--muted); margin: 6px 0 0; }
  form { display: grid; grid-template-columns: 1fr 180px 180px auto auto; gap: 8px; padding: 16px 24px 8px; }
  input, select, button { background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; }
  button { cursor: pointer; }
  button.primary { background: var(--hot); color: #1a1608; border-color: transparent; font-weight: 650; }
  #mic[hidden] { display: none; }
  #status, #summary { padding: 8px 24px; }
  #summary { color: var(--ink); }
  .group { padding: 8px 24px 16px; }
  .group h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 12px 0; }
  .card { display: grid; grid-template-columns: 280px 1fr; gap: 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 10px; }
  video { width: 100%; background: #000; border-radius: 4px; }
  .meta { color: var(--muted); font-size: 13px; }
  .verdict { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 650; }
  .near_miss { background: #5a3418; color: var(--hot); }
  .clear { background: #24301c; color: var(--ok); }
  .unparsed { background: #2a2a2a; color: var(--muted); }
  .boxes { font-size: 13px; color: var(--muted); }
  @media (max-width: 800px) {
    form, .card { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>Near-miss desk</h1>
  <p class="lede">Person close to a moving vehicle, across the cameras already indexed.</p>
  <p class="lede"><a id="training-link" href="training">Yield-or-proceed training</a></p>
  <script>
  (function () {
    var path = location.pathname;
    var link = document.getElementById("training-link");
    if (path.charAt(path.length - 1) === "/") link.href = "training";
    else link.href = path + "/training";
  })();
  </script>
</header>
<form id="desk">
  <input id="question" name="question" placeholder="person close to a moving vehicle">
  <select id="location" name="location"><option value="">All locations</option></select>
  <select id="camera" name="camera_id"><option value="">All cameras</option></select>
  <button type="button" id="mic" hidden>Speak</button>
  <button class="primary" type="submit">Review</button>
</form>
<div id="status"></div>
<div id="summary"></div>
<div id="groups"></div>
<script>
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const groupsEl = document.getElementById("groups");
if (window.isSecureContext && navigator.mediaDevices) document.getElementById("mic").hidden = false;

async function loadFilters() {
  const res = await fetch("/api/filters");
  if (!res.ok) return;
  const data = await res.json();
  for (const [id, key] of [["location", "locations"], ["camera", "cameras"]]) {
    const sel = document.getElementById(id);
    for (const value of data[key] || []) {
      const opt = document.createElement("option");
      opt.value = value; opt.textContent = value;
      sel.appendChild(opt);
    }
  }
}
loadFilters().catch(() => {});

document.getElementById("mic").onclick = async () => {
  statusEl.textContent = "Listening…";
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
    const buf = await blob.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (const b of bytes) bin += String.fromCharCode(b);
    submit({ audio_base64: btoa(bin), audio_type: blob.type });
  };
  rec.start();
  setTimeout(() => rec.stop(), 4000);
};

document.getElementById("desk").onsubmit = (ev) => {
  ev.preventDefault();
  submit({
    question: document.getElementById("question").value,
    location: document.getElementById("location").value,
    camera_id: document.getElementById("camera").value,
  });
};

async function submit(body) {
  statusEl.textContent = "Reviewing…";
  summaryEl.textContent = "";
  groupsEl.innerHTML = "";
  try {
    const res = await fetch("/api/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) { statusEl.textContent = data.error || "Review failed"; return; }
    statusEl.textContent = data.warning || "";
    summaryEl.textContent = data.summary || "";
    if (!data.groups || !data.groups.length) { statusEl.textContent = "No matches."; return; }
    for (const group of data.groups) {
      const wrap = document.createElement("section");
      wrap.className = "group";
      const h = document.createElement("h2");
      h.textContent = group.location || "Unknown location";
      wrap.appendChild(h);
      for (const hit of group.hits) {
        const card = document.createElement("article");
        card.className = "card";
        const left = document.createElement("div");
        if (hit.media) {
          const video = document.createElement("video");
          video.controls = true; video.preload = "none"; video.src = hit.media;
          left.appendChild(video);
        }
        const right = document.createElement("div");
        const badge = document.createElement("span");
        badge.className = "verdict " + (hit.verdict || "unparsed");
        badge.textContent = (hit.verdict || "unparsed") + (hit.severity ? " · " + hit.severity : "");
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = [hit.camera_id, hit.score != null ? "score " + Number(hit.score).toFixed(2) : ""].filter(Boolean).join(" · ");
        const caption = document.createElement("p");
        caption.textContent = hit.caption || "";
        const action = document.createElement("p");
        action.textContent = hit.action || "";
        const boxes = document.createElement("div");
        boxes.className = "boxes";
        boxes.textContent = hit.labels && hit.labels.length ? hit.labels.join(", ") : "No boxes";
        right.append(badge, meta, caption, action, boxes);
        card.append(left, right);
        wrap.appendChild(card);
      }
      groupsEl.appendChild(wrap);
    }
  } catch (err) {
    statusEl.textContent = "Review failed";
  }
}
</script>
</body>
</html>
"""


def _env(name: str) -> str:
    return os.environ.get(name) or ""


def _redact(text: str) -> str:
    text = str(text)
    for secret in (VSS_PASSWORD, WANDB_API_KEY, GPU_BEARER_TOKEN, _token):
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", text)
    text = re.sub(r"https?://\S+", "[url]", text)
    return text[:300]


def _request(method: str, url: str, headers=None, data=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read() if exc.fp else b""
    except urllib.error.URLError as exc:
        raise RuntimeError("upstream unavailable") from exc


def _json_request(method: str, url: str, payload=None, headers=None, timeout=60):
    body = None if payload is None else json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    status, _, raw = _request(method, url, hdrs, body, timeout=timeout)
    try:
        parsed = json.loads(raw.decode() or "null")
    except json.JSONDecodeError:
        parsed = None
    return status, parsed


def login(force=False) -> str:
    global _token
    with _token_lock:
        if _token and not force:
            return _token
        if not VSS_URL or not VSS_USERNAME or not VSS_PASSWORD:
            raise RuntimeError("VSS credentials are not configured")
        status, parsed = _json_request(
            "POST",
            VSS_URL + "/api/v1/auth/login",
            {"username": VSS_USERNAME, "password": VSS_PASSWORD},
            timeout=30,
        )
        if status != 200 or not isinstance(parsed, dict) or not parsed.get("access_token"):
            raise RuntimeError("login failed")
        _token = parsed["access_token"]
        return _token


def vss(method: str, path: str, payload=None, timeout=60):
    token = login()
    headers = {"Authorization": f"Bearer {token}"}
    status, parsed = _json_request(method, VSS_URL + path, payload, headers, timeout=timeout)
    if status == 401:
        token = login(force=True)
        headers = {"Authorization": f"Bearer {token}"}
        status, parsed = _json_request(method, VSS_URL + path, payload, headers, timeout=timeout)
    return status, parsed


def filter_values(field: str) -> list[str]:
    status, parsed = vss("GET", f"/api/v1/metadata/values?field={urllib.parse.quote(field)}&limit=50")
    if status != 200 or not isinstance(parsed, dict):
        return []
    values = parsed.get("values") or []
    return [v for v in values if isinstance(v, str) and v.strip()]


def transcribe(audio: bytes, content_type: str) -> str:
    if not GPU_BEARER_TOKEN:
        raise RuntimeError("speech is not configured")
    boundary = "----nearmiss"
    filename = "speech.wav" if "wav" in content_type else "speech.webm"
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type or 'application/octet-stream'}\r\n\r\n".encode(),
        audio,
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="model"\r\n\r\n',
        b"nvidia/canary-1b\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    status, _, raw = _request(
        "POST",
        CANARY_URL + "/v1/audio/transcriptions",
        {
            "Authorization": f"Bearer {GPU_BEARER_TOKEN}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        b"".join(parts),
        timeout=90,
    )
    if status != 200:
        raise RuntimeError("transcription failed")
    try:
        parsed = json.loads(raw.decode() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("transcription failed") from exc
    text = parsed.get("text") or parsed.get("transcript") or ""
    if not str(text).strip():
        raise RuntimeError("transcription was empty")
    return str(text).strip()


def _labels_from_detections(payload) -> list[str]:
    labels = []
    items = payload
    if isinstance(payload, dict):
        items = payload.get("detections") or payload.get("objects") or payload.get("frames") or []
    if isinstance(items, dict):
        items = items.get("detections") or []
    if not isinstance(items, list):
        return []
    for item in items:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict):
            label = item.get("label") or item.get("class") or item.get("name")
            if label:
                labels.append(str(label))
            for obj in item.get("objects") or item.get("detections") or []:
                if isinstance(obj, dict) and (obj.get("label") or obj.get("class")):
                    labels.append(str(obj.get("label") or obj.get("class")))
    return sorted(set(labels))


def detections_for(source: str):
    quoted = urllib.parse.quote(source, safe="")
    status, parsed = vss("GET", f"/api/v1/videos/detections?source={quoted}")
    if status == 404:
        return [], []
    if status != 200:
        return [], []
    return parsed if isinstance(parsed, (dict, list)) else [], _labels_from_detections(parsed)


def wandb_headers() -> dict:
    headers = {
        "Authorization": f"Bearer {WANDB_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "OpenAI/Python 1.68.0",
    }
    if WANDB_TEAM and WANDB_PROJECT:
        headers["OpenAI-Project"] = f"{WANDB_TEAM}/{WANDB_PROJECT}"
    return headers


def verdict_for(question: str, caption: str, labels: list[str]) -> dict:
    if not WANDB_API_KEY:
        return {"verdict": "unparsed", "severity": "", "action": ""}
    prompt = (
        "Classify this video moment for a safety reviewer. "
        'Reply with JSON only: {"verdict":"near_miss"|"clear","severity":"high"|"medium"|"low","action":"one sentence"}. '
        f"Question: {question}\nCaption: {caption}\nObjects: {', '.join(labels) or 'none'}"
    )
    status, parsed = _json_request(
        "POST",
        WANDB_INFERENCE + "/chat/completions",
        {
            "model": WANDB_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0,
        },
        wandb_headers(),
        timeout=60,
    )
    if status != 200 or not isinstance(parsed, dict):
        print(f"wandb_chat_http={status}")
        return {"verdict": "unparsed", "severity": "", "action": ""}
    content = (((parsed.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return parse_verdict(content)


def parse_verdict(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {"verdict": "unparsed", "severity": "", "action": ""}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "unparsed", "severity": "", "action": ""}
    verdict = obj.get("verdict")
    severity = obj.get("severity")
    action = obj.get("action")
    if verdict not in {"near_miss", "clear"} or severity not in {"high", "medium", "low"} or not isinstance(action, str):
        return {"verdict": "unparsed", "severity": "", "action": ""}
    return {"verdict": verdict, "severity": severity, "action": action.strip()}


def log_weave(location: str, camera: str, verdict: str, severity: str) -> None:
    global _weave_ready
    if not WANDB_API_KEY or not WANDB_PROJECT:
        return
    try:
        import wandb
        os.makedirs("/tmp/wandb", exist_ok=True)
        os.environ["WANDB_DIR"] = "/tmp/wandb"
        os.environ["WANDB_CONFIG_DIR"] = "/tmp"
        os.environ["WANDB_CACHE_DIR"] = "/tmp/wandb-cache"
        with _weave_lock:
            if not _weave_ready:
                wandb.init(entity=WANDB_TEAM or None, project=WANDB_PROJECT, job_type="near-miss", reinit=True)
                _weave_ready = True
            wandb.log(
                {
                    "location": location or "",
                    "camera": camera or "",
                    "verdict": verdict or "",
                    "severity": severity or "",
                }
            )
    except Exception:
        print("weave_unavailable")


def search(question: str, location: str, camera_id: str):
    filters = {}
    warning = ""
    locations = filter_values("location")
    cameras = filter_values("camera_id")
    if location:
        if location in locations:
            filters["location"] = location
        else:
            warning = "That location is not in the index."
    if camera_id:
        if camera_id in cameras:
            filters["camera_id"] = camera_id
        else:
            warning = (warning + " " if warning else "") + "That camera is not in the index."
    status, parsed = vss(
        "POST",
        "/api/v1/search",
        {
            "query": question,
            "top_k": 15,
            "min_similarity": 0.3,
            "metadata_filters": filters,
            "include_public": True,
        },
        timeout=90,
    )
    if status != 200 or not isinstance(parsed, dict):
        raise RuntimeError("search failed")
    return parsed, warning.strip()


def summary_for(question: str, location: str, camera_id: str) -> str:
    filters = {}
    if location:
        filters["location"] = location
    if camera_id:
        filters["camera_id"] = camera_id
    status, parsed = vss(
        "POST",
        "/api/v1/agent/search-and-answer",
        {"query": question, "top_k": 10, "min_similarity": 0.3, "metadata_filters": filters},
        timeout=90,
    )
    if status != 200 or not isinstance(parsed, dict):
        return ""
    answer = parsed.get("answer") or ""
    if not answer and isinstance(parsed.get("llm_synthesis"), dict):
        answer = parsed["llm_synthesis"].get("response") or ""
    return str(answer).strip()


def _hit_source(row: dict) -> str:
    return row.get("source") or row.get("preview_source") or ""


def select_hits(payload: dict) -> list[dict]:
    rows = [r for r in (payload.get("results") or []) if isinstance(r, dict) and _hit_source(r)]
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(str(row.get("location") or "unknown"), []).append(row)
    chosen = []
    while len(chosen) < SHOWN_HITS and any(grouped.values()):
        for key in list(grouped):
            if grouped[key]:
                chosen.append(grouped[key].pop(0))
            if len(chosen) >= SHOWN_HITS:
                break
    return chosen


def review(question: str, location: str, camera_id: str) -> dict:
    payload, warning = search(question, location, camera_id)
    hits = []
    for row in select_hits(payload):
        source = _hit_source(row)
        labels_payload, labels = detections_for(source)
        decided = verdict_for(question, str(row.get("reasoning_content") or ""), labels)
        location_name = str(row.get("location") or "unknown")
        camera = str(row.get("camera_id") or "")
        log_weave(location_name, camera, decided.get("verdict") or "", decided.get("severity") or "")
        hits.append(
            {
                "location": location_name,
                "camera_id": camera,
                "score": row.get("similarity_score"),
                "caption": row.get("reasoning_content") or "",
                "verdict": decided.get("verdict") or "unparsed",
                "severity": decided.get("severity") or "",
                "action": decided.get("action") or "",
                "labels": labels,
                "detections": labels_payload,
                "media": "/api/media?source=" + urllib.parse.quote(source, safe=""),
            }
        )
    groups: dict[str, list] = {}
    for hit in hits:
        groups.setdefault(hit["location"], []).append(hit)
    text = ""
    try:
        text = summary_for(question, location if location else "", camera_id if camera_id else "")
    except RuntimeError:
        text = ""
    return {
        "question": question,
        "summary": text,
        "warning": warning,
        "groups": [{"location": key, "hits": value} for key, value in groups.items()],
    }


def _training_file(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, name), os.path.join(here, "..", "training", "out", name)):
        if os.path.isfile(path):
            with open(path) as handle:
                return json.load(handle)
    return None


def _public_predictions(rows: list) -> list[dict]:
    public = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "")
        public.append(
            {
                "camera_id": row.get("camera_id") or "",
                "caption": row.get("caption") or "",
                "correct": bool(row.get("correct")),
                "cosmos": row.get("cosmos") if isinstance(row.get("cosmos"), dict) else {},
                "location": row.get("location") or "",
                "media": "/api/media?source=" + urllib.parse.quote(source, safe=""),
                "model": row.get("model") if isinstance(row.get("model"), dict) else {},
                "segment_start_sec": row.get("segment_start_sec"),
            }
        )
    return public


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _training_document(metrics: dict, predictions: list[dict], wandb_url: str) -> str:
    mix = metrics.get("label_mix") or {}
    labeled = int(mix.get("labeled") or 0)
    yield_n = int(mix.get("yield") or 0)
    proceed_n = int(mix.get("proceed") or 0)
    yield_pct = (100.0 * yield_n / labeled) if labeled else 0.0
    proceed_pct = (100.0 * proceed_n / labeled) if labeled else 0.0
    model = metrics.get("model") or {}
    majority = metrics.get("majority") or {}
    distance = metrics.get("distance") or {}
    cards = sorted(
        predictions,
        key=lambda row: (bool(row.get("correct")), str(row.get("location") or ""), float(row.get("segment_start_sec") or 0)),
    )
    blocks = []
    for row in cards:
        cosmos = row.get("cosmos") or {}
        guessed = row.get("model") or {}
        agree = bool(row.get("correct"))
        mark = "agree" if agree else "disagree"
        media = str(row.get("media") or "")
        if media.startswith("/api/media"):
            media = media[1:]
        start = row.get("segment_start_sec")
        start_text = _fmt(start) if isinstance(start, (int, float)) else ""
        blocks.append(
            "<article class='card'>"
            f"<video controls preload='metadata' src='{_html(media)}'></video>"
            "<div>"
            f"<span class='mark {mark}'>{mark}</span>"
            f"<div class='meta'>{_html(row.get('location') or '')} · {_html(row.get('camera_id') or '')} · {start_text}s</div>"
            f"<p><b>Cosmos</b> {_html(cosmos.get('decision') or '')} · {_html(cosmos.get('machine') or '')}. {_html(cosmos.get('reason') or '')}</p>"
            f"<p><b>Model</b> {_html(guessed.get('decision') or '')} · p_yield {_fmt(guessed.get('p_yield'))}</p>"
            f"<p>{_html(row.get('caption') or '')}</p>"
            "</div></article>"
        )
    wandb_html = f"<a href='{_html(wandb_url)}'>W&amp;B run</a>" if wandb_url else "W&amp;B run link is not available."
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yield-or-proceed</title>
<style>
  :root {{ color-scheme: dark; --bg: #12140f; --panel: #1c2118; --ink: #e7ecdf; --muted: #9aa592; --line: #343b2e; --hot: #e2b340; --ok: #8fb573; --bad: #e07a5f; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 15px/1.45 "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
  header, .limits {{ padding: 20px 24px 8px; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0; }}
  a {{ color: var(--hot); }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 8px 24px 16px; }}
  .stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }}
  .stat span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
  .grid {{ padding: 0 24px 24px; }}
  .card {{ display: grid; grid-template-columns: 280px 1fr; gap: 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 10px; }}
  video {{ width: 100%; background: #000; border-radius: 4px; }}
  .meta {{ color: var(--muted); font-size: 13px; margin: 6px 0; }}
  .mark {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 650; }}
  .agree {{ background: #24301c; color: var(--ok); }}
  .disagree {{ background: #3a221c; color: var(--bad); }}
  @media (max-width: 800px) {{ .stats, .card {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>Yield-or-proceed</h1>
  <p><a href="./">Near-miss desk</a> · {wandb_html}</p>
</header>
<section class="stats">
  <div class="stat"><span>Dataset</span>{int(metrics.get("dataset_size") or 0)} clips · train {int(metrics.get("train_size") or 0)} · test {int(metrics.get("test_size") or 0)}</div>
  <div class="stat"><span>Label mix</span>yield {yield_n} ({yield_pct:.0f}%) · proceed {proceed_n} ({proceed_pct:.0f}%) · unparsed {int(mix.get("unparsed") or 0)}</div>
  <div class="stat"><span>Model macro F1</span>{_fmt(model.get("macro_f1"))}</div>
  <div class="stat"><span>Baselines</span>majority {_fmt(majority.get("macro_f1"))} ({_html(metrics.get("majority_class") or "")}) · distance {_fmt(distance.get("macro_f1"))} (min_gap &lt; {_fmt(metrics.get("distance_threshold"))})</div>
</section>
<p class="limits">{_html(_baseline_note(model, distance))}</p>
<section class="limits">
  <p>About 240 clips, from only two scenes. One is synthetic. This is a working data engine and a baseline model, not a deployable robot policy.</p>
  <p>Labels come from Cosmos Reason reading captions, not from humans or pixels. The model learns to agree with Cosmos.</p>
  <p>YOLO cannot name robots, so geometry treats every non-person box as a machine.</p>
</section>
<div class="grid">
{''.join(blocks)}
</div>
</body>
</html>
"""


def _baseline_note(model: dict, distance: dict) -> str:
    try:
        model_f1 = float(model.get("macro_f1"))
        distance_f1 = float(distance.get("macro_f1"))
    except (TypeError, ValueError):
        return ""
    if distance_f1 > model_f1:
        return "On this test split the distance rule is ahead of the model on macro F1."
    if distance_f1 == model_f1:
        return "On this test split the distance rule ties the model on macro F1."
    return "On this test split the model is ahead of the distance rule on macro F1."


def _html(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        message = _redact(fmt % args)
        print(f"{self.address_string()} {message}")

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _training_payload(self):
        metrics = _training_file("metrics.json")
        predictions = _training_file("predictions.json")
        if not isinstance(metrics, dict) or not isinstance(predictions, list):
            return None
        wandb_info = _training_file("wandb.json")
        wandb_url = ""
        if isinstance(wandb_info, dict):
            wandb_url = str(wandb_info.get("wandb_url") or "")
        return metrics, _public_predictions(predictions), wandb_url

    def _training_page(self):
        payload = self._training_payload()
        if payload is None:
            self._send(503, b"training outputs are not available", "text/plain; charset=utf-8")
            return
        metrics, predictions, wandb_url = payload
        self._send(200, _training_document(metrics, predictions, wandb_url).encode(), "text/html; charset=utf-8")

    def _training_api(self):
        payload = self._training_payload()
        if payload is None:
            self._json(503, {"error": "training outputs are not available"})
            return
        metrics, predictions, wandb_url = payload
        self._json(200, {"metrics": metrics, "predictions": predictions, "wandb_url": wandb_url})

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/health":
            self._json(200, {"ok": True})
            return
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path in ("/showcase", "/showcase/", "/showcase.html"):
            showcase = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showcase.html")
            with open(showcase, "rb") as handle:
                self._send(200, handle.read(), "text/html; charset=utf-8")
            return
        if path == "/training":
            self._training_page()
            return
        if path == "/api/training":
            self._training_api()
            return
        if path == "/api/filters":
            try:
                self._json(200, {"locations": filter_values("location"), "cameras": filter_values("camera_id")})
            except RuntimeError as exc:
                self._json(502, {"error": _redact(exc)})
            return
        if path == "/api/media":
            self._proxy_media()
            return
        self._json(404, {"error": "not found"})

    def _proxy_media(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        source = (query.get("source") or [""])[0]
        if not source:
            self._json(400, {"error": "missing source"})
            return
        try:
            token = login()
        except RuntimeError as exc:
            self._json(502, {"error": _redact(exc)})
            return
        url = VSS_URL + "/api/v1/videos/stream?source=" + urllib.parse.quote(source, safe="") + "&token=" + urllib.parse.quote(token)
        headers = {}
        if self.headers.get("Range"):
            headers["Range"] = self.headers["Range"]
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as exc:
            self._json(exc.code, {"error": "playback failed"})
            return
        except Exception:
            self._json(502, {"error": "playback failed"})
            return
        self.send_response(resp.status)
        for key in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            value = resp.headers.get(key)
            if value:
                self.send_header(key, value)
        self.end_headers()
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            self.wfile.write(chunk)
        resp.close()

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path != "/api/review":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "expected JSON"})
            return
        if not isinstance(body, dict):
            self._json(400, {"error": "expected JSON"})
            return
        question = str(body.get("question") or "").strip()
        if body.get("audio_base64"):
            import base64
            try:
                audio = base64.b64decode(body["audio_base64"])
                question = transcribe(audio, str(body.get("audio_type") or "audio/webm"))
            except Exception as exc:
                self._json(200, {"question": question, "summary": "", "warning": _redact(exc), "groups": [], "speech_error": True})
                return
        if not question:
            question = DEFAULT_QUESTION
        try:
            result = review(question, str(body.get("location") or "").strip(), str(body.get("camera_id") or "").strip())
        except RuntimeError as exc:
            self._json(502, {"error": _redact(exc)})
            return
        self._json(200, result)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"listening {PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
