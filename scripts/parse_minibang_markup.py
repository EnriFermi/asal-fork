from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TIME_RANGE_RE = re.compile(
    r"(?<!\d)(?P<start>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<end>\d+(?:\.\d+)?)(?!\d)",
    re.IGNORECASE,
)


def _col_to_idx(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if match is None:
        return 0
    out = 0
    for ch in match.group(1):
        out = out * 26 + ord(ch) - ord("A") + 1
    return out - 1


def _read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("x:si", NS):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//x:t", NS)))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            sheet_candidates = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet"))
            if not sheet_candidates:
                raise FileNotFoundError(f"No worksheet XML found in {path}")
            sheet_name = sheet_candidates[0]

        root = ET.fromstring(zf.read(sheet_name))
        rows: list[list[str]] = []
        for row in root.findall(".//x:sheetData/x:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("x:c", NS):
                idx = _col_to_idx(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t", "")
                if cell_type == "s":
                    v = cell.find("x:v", NS)
                    value = shared_strings[int(v.text)] if v is not None and v.text is not None else ""
                elif cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//x:t", NS))
                else:
                    v = cell.find("x:v", NS)
                    value = v.text if v is not None and v.text is not None else ""
                values[idx] = value
            if values:
                rows.append([values.get(i, "") for i in range(max(values) + 1)])
        return rows


def _clean_description(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\s,;]+$", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    elif text.endswith(")") and text.count(")") > text.count("("):
        text = text[:-1].strip()
    text = re.sub(r"^[\s,;:\\–—-]+", "", text)
    text = re.sub(r"[\s,;]+$", "", text)
    return text.strip()


def _is_no_marker(text: str) -> bool:
    norm = text.strip().lower()
    norm = re.sub(r"[\s.]+$", "", norm)
    return norm in {"", "no", "none", "нет"}


def _parse_events(
    text: str,
    *,
    event_type: str,
    trial_id: str,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if _is_no_marker(raw):
        return []

    matches = list(TIME_RANGE_RE.finditer(raw))
    events: list[dict[str, Any]] = []
    if not matches:
        if event_type == "minibang":
            warnings.append(
                {
                    "trial_id": trial_id,
                    "event_type": event_type,
                    "message": "non-empty minibang cell has no parseable time ranges",
                    "raw_text": raw,
                }
            )
        return events

    for idx, match in enumerate(matches):
        start_sec = float(match.group("start"))
        end_sec = float(match.group("end"))
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        desc = _clean_description(raw[match.end() : next_start])
        range_raw = raw[match.start() : next_start].strip()
        if end_sec <= start_sec:
            warnings.append(
                {
                    "trial_id": trial_id,
                    "event_type": event_type,
                    "message": "invalid non-positive time range",
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "raw_text": range_raw,
                }
            )
            continue
        events.append(
            {
                "event_index": len(events),
                "event_type": event_type,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "mid_sec": 0.5 * (start_sec + end_sec),
                "description": desc,
                "raw_text": range_raw,
            }
        )
    return events


def _trial_index(raw_trial_id: str, fallback: int) -> int:
    match = re.search(r"(\d+)", raw_trial_id)
    return int(match.group(1)) if match is not None else fallback


def _load_video_durations(dataset_root: Path | None, traj_ids: list[str]) -> dict[str, float]:
    if dataset_root is None:
        return {}
    durations: dict[str, float] = {}
    for traj_id in traj_ids:
        path = dataset_root / traj_id / "frame_times.csv"
        if not path.exists():
            continue
        try:
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            last = rows[-1]
            if "video_sec" in last:
                durations[traj_id] = float(last["video_sec"])
        except Exception:
            continue
    return durations


def _load_manifest_metadata(dataset_root: Path | None) -> dict[str, dict[str, Any]]:
    if dataset_root is None:
        return {}
    path = dataset_root / "manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    rows = payload.get("trajectories", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        traj_id = row.get("traj_id", None)
        if traj_id is None:
            continue
        out[str(traj_id)] = {
            "optimization_iter": row.get("iter", None),
            "loss": row.get("loss", None),
            "saturation_T": row.get("saturation_T", None),
            "source": row.get("source", ""),
            "selection_idx": row.get("selection_idx", None),
            "param_hash": row.get("param_hash", ""),
        }
    return out


def _parse_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _parse_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        out = int(float(value))
    except Exception:
        return None
    return out


def _load_metric_plot_metadata(dataset_root: Path | None) -> dict[str, dict[str, Any]]:
    if dataset_root is None:
        return {}
    path = dataset_root / "metric_plots" / "metric_plot_summary.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            traj_id = row.get("traj_id", "")
            if not traj_id:
                continue
            out[traj_id] = {
                "longrun_metric_loss": _parse_float_or_none(row.get("longrun_metric_loss")),
                "longrun_metric_score": _parse_float_or_none(row.get("longrun_metric_score")),
                "longrun_msc_t": _parse_float_or_none(row.get("longrun_msc_t")),
                "delta_h_loss_scalar": _parse_float_or_none(row.get("delta_h_loss_scalar")),
                "delta_h_score_scalar": _parse_float_or_none(row.get("delta_h_score_scalar")),
                "delta_h_msc_scalar": _parse_float_or_none(row.get("delta_h_msc_scalar")),
                "delta_h_max": _parse_float_or_none(row.get("delta_h_max")),
                "delta_h_mean": _parse_float_or_none(row.get("delta_h_mean")),
                "delta_h_max_step": _parse_float_or_none(row.get("delta_h_max_step")),
                "delta_h_max_tau_step": _parse_float_or_none(row.get("delta_h_max_tau_step")),
                "cluster_tv_lag_max": _parse_float_or_none(row.get("cluster_tv_lag_max")),
                "cluster_entropy_norm_min": _parse_float_or_none(row.get("cluster_entropy_norm_min")),
                "cluster_entropy_norm_max": _parse_float_or_none(row.get("cluster_entropy_norm_max")),
                "cluster_mass_n_clusters": _parse_int_or_none(row.get("cluster_mass_n_clusters")),
            }
    return out


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "max": None,
            "std": None,
        }
    xs = sorted(float(x) for x in values)

    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {
        "count": len(xs),
        "min": xs[0],
        "q25": q(0.25),
        "median": q(0.50),
        "mean": statistics.fmean(xs),
        "q75": q(0.75),
        "max": xs[-1],
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def _hist(values: list[float], edges: list[float]) -> list[dict[str, Any]]:
    counts = [0 for _ in range(len(edges) - 1)]
    overflow = 0
    for value in values:
        placed = False
        for i in range(len(edges) - 1):
            left, right = edges[i], edges[i + 1]
            if (value >= left and value < right) or (i == len(edges) - 2 and value == right):
                counts[i] += 1
                placed = True
                break
        if not placed:
            overflow += 1
    rows = [
        {
            "range": f"{edges[i]:g}-{edges[i + 1]:g}",
            "start": edges[i],
            "end": edges[i + 1],
            "count": counts[i],
        }
        for i in range(len(counts))
    ]
    if overflow:
        rows.append({"range": f">={edges[-1]:g}", "start": edges[-1], "end": None, "count": overflow})
    return rows


def _count_distribution(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _finite_xy(rows: list[tuple[Any, Any]]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for raw_x, raw_y in rows:
        try:
            x = float(raw_x)
            y = float(raw_y)
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx <= 0.0 or sy <= 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _correlation(rows: list[tuple[Any, Any]]) -> dict[str, Any]:
    xs, ys = _finite_xy(rows)
    pearson = _pearson_r(xs, ys)
    spearman = _pearson_r(_rankdata(xs), _rankdata(ys)) if len(xs) >= 2 else None
    result: dict[str, Any] = {
        "n": len(xs),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "kendall_tau": None,
        "pearson_p": None,
        "spearman_p": None,
        "kendall_p": None,
    }
    try:
        from scipy import stats as scipy_stats  # type: ignore

        if len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
            pr = scipy_stats.pearsonr(xs, ys)
            sr = scipy_stats.spearmanr(xs, ys)
            kr = scipy_stats.kendalltau(xs, ys)
            result["pearson_r"] = float(pr.statistic)
            result["pearson_p"] = float(pr.pvalue)
            result["spearman_r"] = float(sr.statistic)
            result["spearman_p"] = float(sr.pvalue)
            result["kendall_tau"] = float(kr.statistic)
            result["kendall_p"] = float(kr.pvalue)
    except Exception:
        pass
    return result


def _roc_curve(labels_raw: list[Any], scores_raw: list[Any]) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    for raw_label, raw_score in zip(labels_raw, scores_raw):
        try:
            label = 1 if int(raw_label) > 0 else 0
            score = float(raw_score)
        except Exception:
            continue
        if math.isfinite(score):
            labels.append(label)
            scores.append(score)

    positives = int(sum(labels))
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return {
            "n": len(labels),
            "positives": positives,
            "negatives": negatives,
            "auc": None,
            "fpr": [],
            "tpr": [],
            "thresholds": [],
        }

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [float("inf")]
    tp = 0
    fp = 0
    i = 0
    while i < len(order):
        score = scores[order[i]]
        j = i
        while j < len(order) and scores[order[j]] == score:
            if labels[order[j]]:
                tp += 1
            else:
                fp += 1
            j += 1
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
        thresholds.append(score)
        i = j

    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * 0.5 * (tpr[i] + tpr[i - 1])

    return {
        "n": len(labels),
        "positives": positives,
        "negatives": negatives,
        "auc": float(auc),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def _roc_auc_for_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if len(trial["minibangs"]) > 0 else 0 for trial in trials]

    def values(field: str, sign: float = 1.0) -> list[float | None]:
        out: list[float | None] = []
        for trial in trials:
            value = _parse_float_or_none(trial.get(field))
            out.append(None if value is None else sign * value)
        return out

    specs = [
        ("optimization_iter", "optimization_iter", 1.0),
        ("minus_loss", "loss", -1.0),
        ("saturation_T", "saturation_T", 1.0),
        ("minus_longrun_metric_loss", "longrun_metric_loss", -1.0),
        ("longrun_metric_score", "longrun_metric_score", 1.0),
        ("longrun_msc_t", "longrun_msc_t", 1.0),
        ("delta_h_max", "delta_h_max", 1.0),
        ("cluster_tv_lag_max", "cluster_tv_lag_max", 1.0),
    ]
    curves: dict[str, Any] = {}
    for name, field, sign in specs:
        curve = _roc_curve(labels, values(field, sign))
        curve["source_field"] = field
        curve["score_transform"] = "negate" if sign < 0 else "identity"
        curves[name] = curve
    return {
        "target": "has_minibang",
        "positive_definition": "minibang_count > 0",
        "curves": curves,
    }


def _make_stats(markup: dict[str, Any]) -> dict[str, Any]:
    trials = markup["trials"]
    minibangs = [
        dict(event, trial_id=trial["trial_id"], traj_id=trial["traj_id"], trial_index=trial["trial_index"])
        for trial in trials
        for event in trial["minibangs"]
    ]
    starts = [float(e["start_sec"]) for e in minibangs]
    ends = [float(e["end_sec"]) for e in minibangs]
    mids = [float(e["mid_sec"]) for e in minibangs]
    durations = [float(e["duration_sec"]) for e in minibangs]
    counts_per_trial = [len(trial["minibangs"]) for trial in trials]
    iter_count_rows = [(trial.get("optimization_iter"), len(trial["minibangs"])) for trial in trials]
    loss_count_rows = [(trial.get("loss"), len(trial["minibangs"])) for trial in trials]
    saturation_count_rows = [(trial.get("saturation_T"), len(trial["minibangs"])) for trial in trials]
    longrun_loss_count_rows = [(trial.get("longrun_metric_loss"), len(trial["minibangs"])) for trial in trials]
    longrun_score_count_rows = [(trial.get("longrun_metric_score"), len(trial["minibangs"])) for trial in trials]
    longrun_msc_t_count_rows = [(trial.get("longrun_msc_t"), len(trial["minibangs"])) for trial in trials]
    delta_h_max_count_rows = [(trial.get("delta_h_max"), len(trial["minibangs"])) for trial in trials]
    cluster_tv_count_rows = [(trial.get("cluster_tv_lag_max"), len(trial["minibangs"])) for trial in trials]

    video_durations = [
        float(trial["video_duration_sec"])
        for trial in trials
        if trial.get("video_duration_sec") is not None
    ]
    normalized_mid = []
    normalized_start = []
    for trial in trials:
        duration = trial.get("video_duration_sec")
        if duration is None or float(duration) <= 0.0:
            continue
        for event in trial["minibangs"]:
            normalized_start.append(float(event["start_sec"]) / float(duration))
            normalized_mid.append(float(event["mid_sec"]) / float(duration))

    return {
        "source_path": markup["source_path"],
        "time_unit": "video_seconds",
        "n_trials": len(trials),
        "n_minibangs": len(minibangs),
        "n_trials_with_minibangs": sum(1 for c in counts_per_trial if c > 0),
        "n_trials_without_minibangs": sum(1 for c in counts_per_trial if c == 0),
        "minibangs_per_trial": {
            "summary": _summary([float(x) for x in counts_per_trial]),
            "count_distribution": _count_distribution(counts_per_trial),
        },
        "duration_sec": {
            "summary": _summary(durations),
            "histogram_10sec_bins": _hist(durations, [0, 10, 20, 30, 40, 50, 60]),
        },
        "start_sec": {
            "summary": _summary(starts),
            "histogram_20sec_bins": _hist(starts, [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]),
        },
        "mid_sec": {
            "summary": _summary(mids),
            "histogram_20sec_bins": _hist(mids, [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]),
        },
        "end_sec": {
            "summary": _summary(ends),
            "histogram_20sec_bins": _hist(ends, [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]),
        },
        "normalized_start_fraction": {
            "summary": _summary(normalized_start),
            "histogram_20pct_bins": _hist(normalized_start, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        },
        "normalized_mid_fraction": {
            "summary": _summary(normalized_mid),
            "histogram_20pct_bins": _hist(normalized_mid, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        },
        "video_duration_sec": _summary(video_durations),
        "optimization_correlations": {
            "target": "minibang_count_per_trial",
            "optimization_iter_vs_minibang_count": _correlation(iter_count_rows),
            "loss_vs_minibang_count": _correlation(loss_count_rows),
            "saturation_T_vs_minibang_count": _correlation(saturation_count_rows),
            "longrun_metric_loss_vs_minibang_count": _correlation(longrun_loss_count_rows),
            "longrun_metric_score_vs_minibang_count": _correlation(longrun_score_count_rows),
            "longrun_msc_t_vs_minibang_count": _correlation(longrun_msc_t_count_rows),
            "delta_h_max_vs_minibang_count": _correlation(delta_h_max_count_rows),
            "cluster_tv_lag_max_vs_minibang_count": _correlation(cluster_tv_count_rows),
            "notes": [
                "loss is copied from optimization manifest; lower loss is better.",
                "saturation_T is metadata derived from optimization iteration.",
                "longrun metrics are loaded from metric_plots/metric_plot_summary.csv when available.",
            ],
        },
        "roc_auc": _roc_auc_for_trials(trials),
        "n_interesting_non_minibang_ranges": sum(len(trial["interesting_non_minibangs"]) for trial in trials),
        "n_trials_with_interesting_text": sum(1 for trial in trials if trial.get("interesting_text", "").strip()),
        "parse_warnings": markup["parse_warnings"],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _hist_md(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| range | count |", "|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['range']} | {row['count']} |")
    return lines


def _write_stats_md(path: Path, stats: dict[str, Any]) -> None:
    dur = stats["duration_sec"]["summary"]
    mid = stats["mid_sec"]["summary"]
    start = stats["start_sec"]["summary"]
    norm_mid = stats["normalized_mid_fraction"]["summary"]
    video_duration = stats["video_duration_sec"]
    opt_corr = stats.get("optimization_correlations", {})
    iter_corr = opt_corr.get("optimization_iter_vs_minibang_count", {})
    loss_corr = opt_corr.get("loss_vs_minibang_count", {})
    sat_corr = opt_corr.get("saturation_T_vs_minibang_count", {})
    longrun_loss_corr = opt_corr.get("longrun_metric_loss_vs_minibang_count", {})
    longrun_score_corr = opt_corr.get("longrun_metric_score_vs_minibang_count", {})
    longrun_msc_corr = opt_corr.get("longrun_msc_t_vs_minibang_count", {})
    dh_max_corr = opt_corr.get("delta_h_max_vs_minibang_count", {})
    tv_corr = opt_corr.get("cluster_tv_lag_max_vs_minibang_count", {})
    roc_curves = stats.get("roc_auc", {}).get("curves", {})
    lines = [
        "# Minibang Markup Stats",
        "",
        f"- trials: `{stats['n_trials']}`",
        f"- minibangs: `{stats['n_minibangs']}`",
        f"- trials with minibangs: `{stats['n_trials_with_minibangs']}`",
        f"- trials without minibangs: `{stats['n_trials_without_minibangs']}`",
        f"- interesting non-minibang ranges: `{stats['n_interesting_non_minibang_ranges']}`",
        f"- trials with interesting text: `{stats['n_trials_with_interesting_text']}`",
        f"- video duration, sec: `{_fmt(video_duration['median'])}`",
        "",
        "## Duration, sec",
        "",
        f"- min/q25/median/mean/q75/max: `{_fmt(dur['min'])}` / `{_fmt(dur['q25'])}` / `{_fmt(dur['median'])}` / `{_fmt(dur['mean'])}` / `{_fmt(dur['q75'])}` / `{_fmt(dur['max'])}`",
        "",
        *_hist_md(stats["duration_sec"]["histogram_10sec_bins"]),
        "",
        "## Start Time, sec",
        "",
        f"- min/q25/median/mean/q75/max: `{_fmt(start['min'])}` / `{_fmt(start['q25'])}` / `{_fmt(start['median'])}` / `{_fmt(start['mean'])}` / `{_fmt(start['q75'])}` / `{_fmt(start['max'])}`",
        "",
        *_hist_md(stats["start_sec"]["histogram_20sec_bins"]),
        "",
        "## Midpoint Time, sec",
        "",
        f"- min/q25/median/mean/q75/max: `{_fmt(mid['min'])}` / `{_fmt(mid['q25'])}` / `{_fmt(mid['median'])}` / `{_fmt(mid['mean'])}` / `{_fmt(mid['q75'])}` / `{_fmt(mid['max'])}`",
        "",
        *_hist_md(stats["mid_sec"]["histogram_20sec_bins"]),
        "",
        "## Midpoint Position, Fraction Of Video",
        "",
        f"- min/q25/median/mean/q75/max: `{_fmt(norm_mid['min'])}` / `{_fmt(norm_mid['q25'])}` / `{_fmt(norm_mid['median'])}` / `{_fmt(norm_mid['mean'])}` / `{_fmt(norm_mid['q75'])}` / `{_fmt(norm_mid['max'])}`",
        "",
        *_hist_md(stats["normalized_mid_fraction"]["histogram_20pct_bins"]),
        "",
        "## Optimization Correlations",
        "",
        "| x | y | n | Pearson r | Pearson p | Spearman r | Spearman p | Kendall tau | Kendall p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| optimization_iter | minibang_count | {iter_corr.get('n', '')} | "
            f"{_fmt(iter_corr.get('pearson_r'))} | {_fmt(iter_corr.get('pearson_p'))} | "
            f"{_fmt(iter_corr.get('spearman_r'))} | {_fmt(iter_corr.get('spearman_p'))} | "
            f"{_fmt(iter_corr.get('kendall_tau'))} | {_fmt(iter_corr.get('kendall_p'))} |"
        ),
        (
            f"| loss | minibang_count | {loss_corr.get('n', '')} | "
            f"{_fmt(loss_corr.get('pearson_r'))} | {_fmt(loss_corr.get('pearson_p'))} | "
            f"{_fmt(loss_corr.get('spearman_r'))} | {_fmt(loss_corr.get('spearman_p'))} | "
            f"{_fmt(loss_corr.get('kendall_tau'))} | {_fmt(loss_corr.get('kendall_p'))} |"
        ),
        (
            f"| saturation_T | minibang_count | {sat_corr.get('n', '')} | "
            f"{_fmt(sat_corr.get('pearson_r'))} | {_fmt(sat_corr.get('pearson_p'))} | "
            f"{_fmt(sat_corr.get('spearman_r'))} | {_fmt(sat_corr.get('spearman_p'))} | "
            f"{_fmt(sat_corr.get('kendall_tau'))} | {_fmt(sat_corr.get('kendall_p'))} |"
        ),
        (
            f"| longrun_metric_loss | minibang_count | {longrun_loss_corr.get('n', '')} | "
            f"{_fmt(longrun_loss_corr.get('pearson_r'))} | {_fmt(longrun_loss_corr.get('pearson_p'))} | "
            f"{_fmt(longrun_loss_corr.get('spearman_r'))} | {_fmt(longrun_loss_corr.get('spearman_p'))} | "
            f"{_fmt(longrun_loss_corr.get('kendall_tau'))} | {_fmt(longrun_loss_corr.get('kendall_p'))} |"
        ),
        (
            f"| longrun_metric_score | minibang_count | {longrun_score_corr.get('n', '')} | "
            f"{_fmt(longrun_score_corr.get('pearson_r'))} | {_fmt(longrun_score_corr.get('pearson_p'))} | "
            f"{_fmt(longrun_score_corr.get('spearman_r'))} | {_fmt(longrun_score_corr.get('spearman_p'))} | "
            f"{_fmt(longrun_score_corr.get('kendall_tau'))} | {_fmt(longrun_score_corr.get('kendall_p'))} |"
        ),
        (
            f"| longrun_msc_t | minibang_count | {longrun_msc_corr.get('n', '')} | "
            f"{_fmt(longrun_msc_corr.get('pearson_r'))} | {_fmt(longrun_msc_corr.get('pearson_p'))} | "
            f"{_fmt(longrun_msc_corr.get('spearman_r'))} | {_fmt(longrun_msc_corr.get('spearman_p'))} | "
            f"{_fmt(longrun_msc_corr.get('kendall_tau'))} | {_fmt(longrun_msc_corr.get('kendall_p'))} |"
        ),
        (
            f"| delta_h_max | minibang_count | {dh_max_corr.get('n', '')} | "
            f"{_fmt(dh_max_corr.get('pearson_r'))} | {_fmt(dh_max_corr.get('pearson_p'))} | "
            f"{_fmt(dh_max_corr.get('spearman_r'))} | {_fmt(dh_max_corr.get('spearman_p'))} | "
            f"{_fmt(dh_max_corr.get('kendall_tau'))} | {_fmt(dh_max_corr.get('kendall_p'))} |"
        ),
        (
            f"| cluster_tv_lag_max | minibang_count | {tv_corr.get('n', '')} | "
            f"{_fmt(tv_corr.get('pearson_r'))} | {_fmt(tv_corr.get('pearson_p'))} | "
            f"{_fmt(tv_corr.get('spearman_r'))} | {_fmt(tv_corr.get('spearman_p'))} | "
            f"{_fmt(tv_corr.get('kendall_tau'))} | {_fmt(tv_corr.get('kendall_p'))} |"
        ),
        "",
        "## ROC-AUC",
        "",
        "Positive label: `minibang_count > 0`.",
        "",
        "| score | n | positives | negatives | AUC | transform |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, curve in sorted(
        roc_curves.items(),
        key=lambda kv: (float("-inf") if kv[1].get("auc") is None else -float(kv[1]["auc"]), kv[0]),
    ):
        lines.append(
            f"| {name} | {curve.get('n', '')} | {curve.get('positives', '')} | "
            f"{curve.get('negatives', '')} | {_fmt(curve.get('auc'))} | {curve.get('score_transform', '')} |"
        )
    if stats.get("roc_auc", {}).get("plot_path"):
        lines.extend(["", f"![ROC curves]({Path(str(stats['roc_auc']['plot_path'])).name})"])
    if stats.get("msc_t_kde", {}).get("plot_path"):
        kde = stats["msc_t_kde"]
        lines.extend(
            [
                "",
                "## MSC-T KDE",
                "",
                f"- score: `{kde.get('score', 'longrun_msc_t')}`",
                f"- positive trials: `{kde.get('n_positive_trials', '')}`",
                f"- negative trials: `{kde.get('n_negative_trials', '')}`",
                "",
                f"![MSC-T KDE]({Path(str(kde['plot_path'])).name})",
            ]
        )
    lines.extend(
        [
        "",
        "## Minibangs Per Trial",
        "",
        "| minibangs | trials |",
        "|---:|---:|",
        ]
    )
    for count, n_trials in stats["minibangs_per_trial"]["count_distribution"].items():
        lines.append(f"| {count} | {n_trials} |")
    if stats["parse_warnings"]:
        lines.extend(["", "## Parse Warnings", ""])
        for warning in stats["parse_warnings"]:
            lines.append(f"- `{warning.get('trial_id')}` {warning.get('event_type')}: {warning.get('message')} ({warning.get('raw_text')})")
    path.write_text("\n".join(lines) + "\n")


def build_markup(xlsx_path: Path, dataset_root: Path | None) -> dict[str, Any]:
    rows = _read_xlsx_first_sheet(xlsx_path)
    warnings: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    raw_trials: list[tuple[int, str, list[str]]] = []
    for i, row in enumerate(rows):
        raw_trial_id = str(row[0]).strip() if row else f"trial_{i:04d}"
        if not raw_trial_id:
            raw_trial_id = f"trial_{i:04d}"
        raw_trials.append((_trial_index(raw_trial_id, i), raw_trial_id, row))

    traj_ids = [f"traj_{idx:05d}" for idx, _raw, _row in raw_trials]
    video_durations = _load_video_durations(dataset_root, traj_ids)
    manifest_metadata = _load_manifest_metadata(dataset_root)
    metric_plot_metadata = _load_metric_plot_metadata(dataset_root)

    for idx, raw_trial_id, row in raw_trials:
        traj_id = f"traj_{idx:05d}"
        meta = {**manifest_metadata.get(traj_id, {}), **metric_plot_metadata.get(traj_id, {})}
        minibang_text = row[1] if len(row) > 1 else ""
        interesting_text = row[2] if len(row) > 2 else ""
        notes = row[3] if len(row) > 3 else ""
        trials.append(
            {
                "trial_id": raw_trial_id,
                "trial_index": idx,
                "traj_id": traj_id,
                "video_duration_sec": video_durations.get(traj_id),
                "optimization_iter": meta.get("optimization_iter"),
                "loss": meta.get("loss"),
                "saturation_T": meta.get("saturation_T"),
                "source": meta.get("source", ""),
                "selection_idx": meta.get("selection_idx"),
                "param_hash": meta.get("param_hash", ""),
                "longrun_metric_loss": meta.get("longrun_metric_loss"),
                "longrun_metric_score": meta.get("longrun_metric_score"),
                "longrun_msc_t": meta.get("longrun_msc_t"),
                "delta_h_loss_scalar": meta.get("delta_h_loss_scalar"),
                "delta_h_score_scalar": meta.get("delta_h_score_scalar"),
                "delta_h_msc_scalar": meta.get("delta_h_msc_scalar"),
                "delta_h_max": meta.get("delta_h_max"),
                "delta_h_mean": meta.get("delta_h_mean"),
                "delta_h_max_step": meta.get("delta_h_max_step"),
                "delta_h_max_tau_step": meta.get("delta_h_max_tau_step"),
                "cluster_tv_lag_max": meta.get("cluster_tv_lag_max"),
                "cluster_entropy_norm_min": meta.get("cluster_entropy_norm_min"),
                "cluster_entropy_norm_max": meta.get("cluster_entropy_norm_max"),
                "cluster_mass_n_clusters": meta.get("cluster_mass_n_clusters"),
                "minibang_text": str(minibang_text or "").strip(),
                "interesting_text": str(interesting_text or "").strip(),
                "notes": str(notes or "").strip(),
                "minibangs": _parse_events(
                    str(minibang_text or ""),
                    event_type="minibang",
                    trial_id=raw_trial_id,
                    warnings=warnings,
                ),
                "interesting_non_minibangs": _parse_events(
                    str(interesting_text or ""),
                    event_type="interesting_non_minibang",
                    trial_id=raw_trial_id,
                    warnings=warnings,
                ),
            }
        )

    def flat_events(key: str) -> list[dict[str, Any]]:
        rows_out: list[dict[str, Any]] = []
        for trial in trials:
            for event in trial[key]:
                item = {
                    "trial_id": trial["trial_id"],
                    "trial_index": trial["trial_index"],
                    "traj_id": trial["traj_id"],
                    "video_duration_sec": trial["video_duration_sec"],
                    "optimization_iter": trial.get("optimization_iter"),
                    "loss": trial.get("loss"),
                    "saturation_T": trial.get("saturation_T"),
                    "source": trial.get("source", ""),
                    "selection_idx": trial.get("selection_idx"),
                    "param_hash": trial.get("param_hash", ""),
                    "longrun_metric_loss": trial.get("longrun_metric_loss"),
                    "longrun_metric_score": trial.get("longrun_metric_score"),
                    "longrun_msc_t": trial.get("longrun_msc_t"),
                    "delta_h_loss_scalar": trial.get("delta_h_loss_scalar"),
                    "delta_h_score_scalar": trial.get("delta_h_score_scalar"),
                    "delta_h_msc_scalar": trial.get("delta_h_msc_scalar"),
                    "delta_h_max": trial.get("delta_h_max"),
                    "delta_h_mean": trial.get("delta_h_mean"),
                    "delta_h_max_step": trial.get("delta_h_max_step"),
                    "delta_h_max_tau_step": trial.get("delta_h_max_tau_step"),
                    "cluster_tv_lag_max": trial.get("cluster_tv_lag_max"),
                    "cluster_entropy_norm_min": trial.get("cluster_entropy_norm_min"),
                    "cluster_entropy_norm_max": trial.get("cluster_entropy_norm_max"),
                    "cluster_mass_n_clusters": trial.get("cluster_mass_n_clusters"),
                    **event,
                }
                rows_out.append(item)
        return rows_out

    return {
        "source_path": str(xlsx_path),
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "time_unit": "video_seconds",
        "schema": {
            "column_1": "trial_id",
            "column_2": "minibang_text",
            "column_3": "interesting_non_minibang_text",
            "column_4": "notes",
        },
        "n_trials": len(trials),
        "minibangs": flat_events("minibangs"),
        "interesting_non_minibangs": flat_events("interesting_non_minibangs"),
        "trials": trials,
        "parse_warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse hand-written FlowLenia minibang XLSX markup into JSON.")
    parser.add_argument("xlsx_path", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root with traj_XXXXX/frame_times.csv.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--stats-md", type=Path, default=None)
    parser.add_argument("--roc-png", type=Path, default=None)
    parser.add_argument("--kde-png", type=Path, default=None)
    return parser.parse_args()


def _ensure_matplotlib_cache() -> None:
    import os
    import tempfile

    mpl_cache = Path(tempfile.gettempdir()) / "flowlenia_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))


def _write_roc_plot(path: Path, stats: dict[str, Any]) -> None:
    curves = stats.get("roc_auc", {}).get("curves", {})
    curves = {
        name: curve
        for name, curve in curves.items()
        if curve.get("auc") is not None and curve.get("fpr") and curve.get("tpr")
    }
    if not curves:
        return
    _ensure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(curves.items(), key=lambda kv: -float(kv[1]["auc"]))
    fig, ax = plt.subplots(figsize=(7.6, 6.0), constrained_layout=True)
    for name, curve in ordered:
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=1.8,
            label=f"{name} (AUC={float(curve['auc']):.3f})",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: has minibang")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_msc_t_kde_plot(path: Path, markup: dict[str, Any]) -> dict[str, Any]:
    rows: list[tuple[float, int]] = []
    for trial in markup.get("trials", []):
        score = _parse_float_or_none(trial.get("longrun_msc_t"))
        if score is not None:
            rows.append((score, len(trial.get("minibangs", []))))

    positives = [score for score, count in rows if count > 0]
    negatives = [score for score, count in rows if count == 0]
    summary: dict[str, Any] = {
        "score": "longrun_msc_t",
        "n": len(rows),
        "n_positive_trials": len(positives),
        "n_negative_trials": len(negatives),
        "plot_path": str(path),
    }
    if len(rows) < 3 or len(positives) < 2 or len(negatives) < 2:
        summary["status"] = "not_enough_points"
        return summary

    _ensure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde  # type: ignore

    all_scores = np.asarray([score for score, _count in rows], dtype=np.float64)
    x_min = float(np.nanmin(all_scores))
    x_max = float(np.nanmax(all_scores))
    pad = 0.08 * max(x_max - x_min, 1e-9)
    xs = np.linspace(x_min - pad, x_max + pad, 512)

    kde_neg = gaussian_kde(np.asarray(negatives, dtype=np.float64))
    kde_pos = gaussian_kde(np.asarray(positives, dtype=np.float64))
    ys_neg = kde_neg(xs)
    ys_pos = kde_pos(xs)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    ax.fill_between(xs, ys_neg, color="#4c78a8", alpha=0.22)
    ax.plot(xs, ys_neg, color="#4c78a8", linewidth=2.0, label=f"no minibang (n={len(negatives)})")
    ax.fill_between(xs, ys_pos, color="#f58518", alpha=0.25)
    ax.plot(xs, ys_pos, color="#f58518", linewidth=2.0, label=f"has minibang (n={len(positives)})")

    weighted_scores = np.asarray([score for score, count in rows for _ in range(count)], dtype=np.float64)
    if weighted_scores.size >= 2 and np.nanstd(weighted_scores) > 0.0:
        kde_weighted = gaussian_kde(weighted_scores)
        ax.plot(
            xs,
            kde_weighted(xs),
            color="#e45756",
            linestyle="--",
            linewidth=2.0,
            label=f"minibang-count weighted (events={weighted_scores.size})",
        )

    ax.set_title("KDE of minibang markup by longrun MSC-T")
    ax.set_xlabel("longrun_msc_t")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)

    summary["status"] = "ok"
    summary["positive_score_summary"] = _summary(positives)
    summary["negative_score_summary"] = _summary(negatives)
    return summary


def main() -> None:
    args = parse_args()
    xlsx_path = args.xlsx_path
    output_json = args.output_json or xlsx_path.with_suffix(".json")
    stats_json = args.stats_json or xlsx_path.with_name(f"{xlsx_path.stem}_stats.json")
    stats_md = args.stats_md or xlsx_path.with_name(f"{xlsx_path.stem}_stats.md")
    roc_png = args.roc_png or stats_json.with_name(f"{stats_json.stem}_roc.png")
    kde_png = args.kde_png or stats_json.with_name(f"{stats_json.stem}_msc_t_kde.png")

    markup = build_markup(xlsx_path, args.dataset_root)
    stats = _make_stats(markup)
    stats["roc_auc"]["plot_path"] = str(roc_png)
    stats["msc_t_kde"] = _write_msc_t_kde_plot(kde_png, markup)

    output_json.write_text(json.dumps(markup, indent=2, sort_keys=False) + "\n")
    stats_json.write_text(json.dumps(stats, indent=2, sort_keys=False) + "\n")
    _write_roc_plot(roc_png, stats)
    _write_stats_md(stats_md, stats)

    print(f"Wrote markup JSON: {output_json}")
    print(f"Wrote stats JSON:  {stats_json}")
    print(f"Wrote stats MD:    {stats_md}")
    print(f"Wrote ROC plot:    {roc_png}")
    print(f"Wrote KDE plot:    {kde_png}")
    print(
        f"Parsed {stats['n_minibangs']} minibangs across "
        f"{stats['n_trials_with_minibangs']}/{stats['n_trials']} trials."
    )


if __name__ == "__main__":
    main()
