"""Split by parent video, fit a logistic yield model, log the run to W&B."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DATASET = OUT / "dataset.jsonl"
SEED = 42
C_GRID = (0.1, 1.0, 10.0)
LABELS = ("proceed", "yield")

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
    "scenario_person_walking_near_moving_robot",
    "scenario_robot_or_agv_carrying_pallet",
    "scenario_forklift_near_person",
    "scenario_empty_aisle_no_people",
)


def load_rows() -> list[dict]:
    rows = []
    with DATASET.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: (row["original_video"], row["segment_number"], row["source"]))
    return rows


def labeled(row: dict) -> bool:
    return row.get("label_status") == "ok" and row.get("decision") in LABELS


def vector(row: dict) -> list[float]:
    values = []
    for name in FEATURE_NAMES:
        value = row.get(name)
        if value is None:
            raise SystemExit(f"missing feature {name}")
        values.append(float(value))
    return values


def num(value) -> float:
    return round(float(value), 6)


def pack_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()
    return {
        "n": len(y_true),
        "accuracy": num(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": num(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)) if y_true else 0.0,
        "yield_precision": num(precision_score(y_true, y_pred, labels=["yield"], average="macro", zero_division=0)) if y_true else 0.0,
        "yield_recall": num(recall_score(y_true, y_pred, labels=["yield"], average="macro", zero_division=0)) if y_true else 0.0,
        "confusion_matrix": matrix,
        "labels": list(LABELS),
    }


def by_location(rows: list[dict], preds: list[str]) -> dict:
    grouped = defaultdict(lambda: {"y": [], "p": []})
    for row, pred in zip(rows, preds):
        grouped[row["location"]]["y"].append(row["decision"])
        grouped[row["location"]]["p"].append(pred)
    return {location: pack_metrics(parts["y"], parts["p"]) for location, parts in sorted(grouped.items())}


def majority_label(rows: list[dict]) -> str:
    counts = Counter(row["decision"] for row in rows)
    if counts["yield"] > counts["proceed"]:
        return "yield"
    return "proceed"


def tune_distance(rows: list[dict]) -> float:
    best_t = 0.0
    best_f1 = -1.0
    y_true = [row["decision"] for row in rows]
    candidates = [index / 100 for index in range(0, 101)]
    for row in rows:
        candidates.append(float(row["min_gap"]))
    for threshold in sorted(set(round(value, 6) for value in candidates)):
        pred = ["yield" if float(row["min_gap"]) < threshold else "proceed" for row in rows]
        score = f1_score(y_true, pred, labels=list(LABELS), average="macro", zero_division=0)
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_t = threshold
    return num(best_t)


def predict_distance(rows: list[dict], threshold: float) -> list[str]:
    return ["yield" if float(row["min_gap"]) < threshold else "proceed" for row in rows]


def video_table(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if labeled(row):
            groups[row["original_video"]].append(row)
    videos = []
    for original, items in groups.items():
        counts = Counter(item["decision"] for item in items)
        if counts["yield"] > counts["proceed"]:
            majority = "yield"
        else:
            majority = "proceed"
        videos.append(
            {
                "id": original,
                "location": items[0]["location"],
                "n": len(items),
                "majority": majority,
                "labels": set(counts),
            }
        )
    videos.sort(key=lambda video: video["id"])
    return videos


def _cover(videos: list[dict]) -> tuple[set, set]:
    labels = set()
    locations = set()
    for video in videos:
        labels |= set(video["labels"])
        locations.add(video["location"])
    return labels, locations


def split_videos(videos: list[dict]) -> tuple[set, set]:
    rng = random.Random(SEED)
    strata = defaultdict(list)
    for video in videos:
        strata[(video["location"], video["majority"])].append(video)
    train = []
    test = []
    for key in sorted(strata):
        group = strata[key][:]
        rng.shuffle(group)
        total = sum(video["n"] for video in group)
        target = int(round(0.25 * total))
        chosen = []
        got = 0
        for index, video in enumerate(group):
            if index == len(group) - 1:
                break
            if chosen and got >= target:
                break
            chosen.append(video)
            got += video["n"]
        chosen_ids = {video["id"] for video in chosen}
        for video in group:
            if video["id"] in chosen_ids:
                test.append(video)
            else:
                train.append(video)

    def move(src, dst, predicate) -> bool:
        candidates = [video for video in src if predicate(video)]
        candidates.sort(key=lambda video: (video["n"], video["id"]))
        for pick in candidates:
            rest = [video for video in src if video["id"] != pick["id"]]
            src_labels, src_locations = _cover(src)
            rest_labels, rest_locations = _cover(rest)
            if rest_labels < src_labels or rest_locations < src_locations:
                continue
            src.remove(pick)
            dst.append(pick)
            return True
        return False

    for location in ("indoor", "warehouse3"):
        if location not in _cover(test)[1]:
            move(train, test, lambda video, location=location: video["location"] == location)
        if location not in _cover(train)[1]:
            move(test, train, lambda video, location=location: video["location"] == location)
    for label in LABELS:
        if label not in _cover(test)[0]:
            move(train, test, lambda video, label=label: label in video["labels"])
        if label not in _cover(train)[0]:
            move(test, train, lambda video, label=label: label in video["labels"])
    train_ids = {video["id"] for video in train}
    test_ids = {video["id"] for video in test}
    if train_ids & test_ids:
        raise SystemExit("split leaked a parent video")
    return train_ids, test_ids


def choose_c(train_rows: list[dict]) -> float:
    groups = [row["original_video"] for row in train_rows]
    n_groups = len(set(groups))
    if n_groups < 2:
        return 1.0
    n_splits = min(3, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    x = np.asarray([vector(row) for row in train_rows], dtype=float)
    y = np.asarray([row["decision"] for row in train_rows])
    group_ids = np.asarray(groups)
    best_c = C_GRID[0]
    best_score = -1.0
    for c_value in C_GRID:
        scores = []
        for train_index, test_index in splitter.split(x, y, group_ids):
            if len(set(y[train_index])) < 2:
                scores.append(0.0)
                continue
            model = _pipeline(c_value)
            model.fit(x[train_index], y[train_index])
            pred = model.predict(x[test_index])
            scores.append(f1_score(y[test_index], pred, labels=list(LABELS), average="macro", zero_division=0))
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score + 1e-12:
            best_score = score
            best_c = c_value
    return float(best_c)


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def export_model(pipeline: Pipeline, c_value: float) -> dict:
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    classes = [str(label) for label in clf.classes_]
    if classes != list(LABELS):
        raise SystemExit(f"unexpected class order {classes}")
    scales = [float(value) if float(value) != 0.0 else 1.0 for value in scaler.scale_]
    return {
        "C": c_value,
        "classes": list(LABELS),
        "coefficients": [num(value) for value in clf.coef_[0]],
        "feature_names": list(FEATURE_NAMES),
        "intercept": num(float(clf.intercept_[0])),
        "means": [num(value) for value in scaler.mean_],
        "scales": [num(value) for value in scales],
        "threshold": 0.5,
    }


def score_row(model: dict, row: dict) -> float:
    total = float(model["intercept"])
    for index, name in enumerate(model["feature_names"]):
        scale = float(model["scales"][index]) or 1.0
        total += float(model["coefficients"][index]) * ((float(row[name]) - float(model["means"][index])) / scale)
    return sigmoid(total)


def decide(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "yield"
    return "proceed"


def write_json(path: Path, payload) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def log_wandb(model: dict, metrics: dict, predictions: list[dict], prompt_version: str) -> dict:
    import wandb

    wandb_dir = OUT / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CONFIG_DIR"] = "/tmp"
    os.environ["WANDB_CACHE_DIR"] = str(OUT / "wandb-cache")
    entity = os.environ.get("WANDB_TEAM") or ""
    project = os.environ.get("WANDB_PROJECT") or ""
    if not entity or not project or not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_TEAM, WANDB_PROJECT, and WANDB_API_KEY must be set")
    run = wandb.init(
        entity=entity,
        project=project,
        job_type="train",
        config={
            "C": model["C"],
            "features": list(FEATURE_NAMES),
            "label_prompt_version": prompt_version,
            "seed": SEED,
            "test_size": metrics["test_size"],
            "train_size": metrics["train_size"],
        },
        settings=wandb.Settings(silent=True),
    )
    # A dataset logged in this same run is not recorded as an input. Use the
    # already committed version first so the model artifact has lineage.
    try:
        run.use_artifact(f"{entity}/{project}/yield-dataset:latest", type="dataset")
    except Exception:
        pass
    summary = {
        "model/accuracy": metrics["model"]["accuracy"],
        "model/macro_f1": metrics["model"]["macro_f1"],
        "model/yield_precision": metrics["model"]["yield_precision"],
        "model/yield_recall": metrics["model"]["yield_recall"],
        "majority/accuracy": metrics["majority"]["accuracy"],
        "majority/macro_f1": metrics["majority"]["macro_f1"],
        "majority/yield_precision": metrics["majority"]["yield_precision"],
        "majority/yield_recall": metrics["majority"]["yield_recall"],
        "distance/accuracy": metrics["distance"]["accuracy"],
        "distance/macro_f1": metrics["distance"]["macro_f1"],
        "distance/yield_precision": metrics["distance"]["yield_precision"],
        "distance/yield_recall": metrics["distance"]["yield_recall"],
        "distance/threshold": metrics["distance_threshold"],
    }
    wandb.log(summary)
    run.summary.update(summary)
    y_true = [1 if row["cosmos"]["decision"] == "yield" else 0 for row in predictions]
    y_pred = [1 if row["model"]["decision"] == "yield" else 0 for row in predictions]
    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=["proceed", "yield"],
            )
        }
    )
    coef_table = wandb.Table(data=[[name, coef] for name, coef in zip(model["feature_names"], model["coefficients"])], columns=["feature", "coefficient"])
    wandb.log({"coefficients": wandb.plot.bar(coef_table, "feature", "coefficient", title="Coefficients")})
    pred_table = wandb.Table(columns=["caption", "cosmos_label", "model_probability", "correct"])
    for row in predictions:
        pred_table.add_data(
            row["caption"],
            row["cosmos"]["decision"],
            row["model"]["p_yield"],
            "correct" if row["correct"] else "incorrect",
        )
    wandb.log({"test_predictions": pred_table})
    dataset_art = wandb.Artifact("yield-dataset", type="dataset")
    dataset_art.add_file(str(DATASET), name="dataset.jsonl")
    logged_dataset = run.log_artifact(dataset_art)
    logged_dataset.wait()
    version = logged_dataset.version or "latest"
    try:
        run.use_artifact(f"{entity}/{project}/yield-dataset:{version}", type="dataset")
    except Exception:
        pass
    model_art = wandb.Artifact("yield-model", type="model")
    model_art.add_file(str(OUT / "model.json"), name="model.json")
    model_art.add_file(str(OUT / "metrics.json"), name="metrics.json")
    logged_model = run.log_artifact(model_art)
    logged_model.wait()
    url = run.url or f"https://wandb.ai/{entity}/{project}/runs/{run.id}"
    info = {
        "dataset_artifact": f"yield-dataset:{logged_dataset.version}",
        "model_artifact": f"yield-model:{logged_model.version}",
        "run_id": run.id,
        "wandb_url": url,
    }
    run.finish()
    return info


def main() -> None:
    rows = load_rows()
    usable = [row for row in rows if labeled(row)]
    if len({row["decision"] for row in usable}) < 2:
        raise SystemExit("need both yield and proceed labels")
    train_ids, test_ids = split_videos(video_table(usable))
    train_rows = [row for row in usable if row["original_video"] in train_ids]
    test_rows = [row for row in usable if row["original_video"] in test_ids]
    train_rows.sort(key=lambda row: (row["original_video"], row["segment_number"], row["source"]))
    test_rows.sort(key=lambda row: (row["original_video"], row["segment_number"], row["source"]))
    if set(row["original_video"] for row in train_rows) & set(row["original_video"] for row in test_rows):
        raise SystemExit("segment split leaked a parent video")
    test_labels = {row["decision"] for row in test_rows}
    test_locations = {row["location"] for row in test_rows}
    if test_labels != set(LABELS) or test_locations != {"indoor", "warehouse3"}:
        raise SystemExit(f"test split missing coverage labels={sorted(test_labels)} locations={sorted(test_locations)}")
    c_value = choose_c(train_rows)
    pipeline = _pipeline(c_value)
    pipeline.fit(np.asarray([vector(row) for row in train_rows], dtype=float), np.asarray([row["decision"] for row in train_rows]))
    model = export_model(pipeline, c_value)
    probabilities = [score_row(model, row) for row in test_rows]
    sklearn_prob = pipeline.predict_proba(np.asarray([vector(row) for row in test_rows], dtype=float))[:, 1]
    delta = max(abs(float(left) - float(right)) for left, right in zip(probabilities, sklearn_prob))
    print(f"score_delta={delta:.6f}", flush=True)
    model_pred = [decide(num(prob), model["threshold"]) for prob in probabilities]
    majority = majority_label(train_rows)
    majority_pred = [majority for _row in test_rows]
    threshold = tune_distance(train_rows)
    distance_pred = predict_distance(test_rows, threshold)
    y_test = [row["decision"] for row in test_rows]
    model_metrics = pack_metrics(y_test, model_pred)
    model_metrics["by_location"] = by_location(test_rows, model_pred)
    majority_metrics = pack_metrics(y_test, majority_pred)
    majority_metrics["by_location"] = by_location(test_rows, majority_pred)
    distance_metrics = pack_metrics(y_test, distance_pred)
    distance_metrics["by_location"] = by_location(test_rows, distance_pred)
    counts = Counter(row.get("label_status") for row in rows)
    decision_counts = Counter(row["decision"] for row in rows if row.get("label_status") == "ok")
    prompt_versions = sorted({str(row.get("prompt_version") or "") for row in rows})
    prompt_version = prompt_versions[0] if len(prompt_versions) == 1 else ",".join(prompt_versions)
    metrics = {
        "C": c_value,
        "dataset_size": len(rows),
        "distance": distance_metrics,
        "distance_threshold": threshold,
        "label_mix": {
            "labeled": int(decision_counts["yield"] + decision_counts["proceed"]),
            "proceed": int(decision_counts["proceed"]),
            "unparsed": int(counts.get("unparsed", 0)),
            "yield": int(decision_counts["yield"]),
        },
        "majority": majority_metrics,
        "majority_class": majority,
        "model": model_metrics,
        "prompt_version": prompt_version,
        "seed": SEED,
        "test_size": len(test_rows),
        "test_videos": len(test_ids),
        "train_size": len(train_rows),
        "train_videos": len(train_ids),
    }
    predictions = []
    for row, probability, pred in zip(test_rows, probabilities, model_pred):
        probability = num(probability)
        predictions.append(
            {
                "camera_id": row.get("camera_id") or "",
                "caption": row.get("reasoning_content") or "",
                "correct": pred == row["decision"],
                "cosmos": {
                    "decision": row["decision"],
                    "machine": row.get("machine") or "",
                    "reason": row.get("reason") or "",
                },
                "location": row.get("location") or "",
                "model": {"decision": pred, "p_yield": probability},
                "segment_start_sec": row.get("segment_start_sec"),
                "source": row["source"],
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "model.json", model)
    write_json(OUT / "predictions.json", predictions)
    write_json(OUT / "metrics.json", metrics)
    info = log_wandb(model, metrics, predictions, prompt_version)
    write_json(OUT / "wandb.json", info)
    print(
        f"train={metrics['train_size']} test={metrics['test_size']} C={c_value} "
        f"model_f1={model_metrics['macro_f1']} majority_f1={majority_metrics['macro_f1']} "
        f"distance_f1={distance_metrics['macro_f1']}",
        flush=True,
    )
    print(f"wandb {entity_path(info)}", flush=True)
    print(f"artifacts {info['dataset_artifact']} {info['model_artifact']}", flush=True)


def entity_path(info: dict) -> str:
    url = info.get("wandb_url") or ""
    marker = "wandb.ai/"
    if marker in url:
        return url.split(marker, 1)[1]
    return info.get("run_id") or ""


if __name__ == "__main__":
    main()
