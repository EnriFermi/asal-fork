from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


_REPO_ROOT = Path(__file__).resolve().parent.parent

FIGURE_SUFFIXES = (".png", ".pdf", ".svg")

EXPECTED_MAIN_FIGURES = (
    "figures/synthetic_calibration_grid.png",
    "figures/synthetic_msc_by_scale.png",
    "figures/synthetic_delta_h_heatmaps.png",
    "figures/c1_flow_lenia_paired_contrast.png",
    "figures/c1_flow_lenia_tau_profiles.png",
    "figures/c1_flow_lenia_delta_h_heatmaps.png",
    "figures/c1_flow_lenia_delta_h_eval_optimized_vs_random_median.png",
    "figures/c1_flow_lenia_delta_h_eval_optimized_vs_random_grid.png",
    "figures/c2_branching_sensitivity.png",
    "figures/c2_delta_h_branching_correlation.png",
    "figures/c5_flow_lenia_frustration_contrast.png",
    "figures/c5_flow_lenia_embedding_vs_mspd.png",
    "figures/c1_plife_plus_paired_contrast.png",
    "figures/c1_plife_plus_tau_profiles.png",
    "figures/c1_plife_plus_delta_h_heatmaps.png",
    "figures/c1_plife_plus_delta_h_eval_optimized_vs_random_median.png",
    "figures/c1_plife_plus_delta_h_eval_optimized_vs_random_grid.png",
    "figures/c5_plife_plus_frustration_contrast.png",
    "figures/c5_plife_plus_embedding_vs_mspd.png",
    "figures/c6_cross_substrate_effects.png",
    "c4_nnopt_vs_mspd/c4_nnopt_vs_mspd_dual_axis.png",
)

SUPPORTING_TABLES = (
    "cross_substrate_summary.csv",
    "paper_suite_metrics_summary.json",
    "visualization_summary.json",
    "synthetic_calibration/per_family_scores.csv",
    "synthetic_calibration/tau_profiles.csv",
    "synthetic_calibration/role_recovery.csv",
    "synthetic_calibration/event_localization.csv",
    "c2_branching/branching_scores.csv",
    "c2_branching/branching_delta_h_correlation.csv",
    "c4_nnopt_vs_mspd/objective_run_scores.csv",
    "c4_nnopt_vs_mspd/hypothesis_tests.csv",
    "c4_nnopt_vs_mspd/c4_nnopt_vs_mspd_summary.json",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - local environment dependent.
        raise RuntimeError("PyYAML is required to read the suite config. Run inside the suite conda env.") from exc
    with path.open("r") as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def _resolve(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def _output_root(config_path: Path, *, smoke: bool) -> Path:
    cfg = _load_yaml(config_path)
    if smoke:
        smoke_cfg = cfg.get("smoke", {}) if isinstance(cfg.get("smoke", {}), dict) else {}
        raw = smoke_cfg.get("output_root")
        if raw:
            return _resolve(raw) or (_REPO_ROOT / "analysis/results/paper_suite_smoke")
    meta = cfg.get("meta", {}) if isinstance(cfg.get("meta", {}), dict) else {}
    return _resolve(meta.get("output_root", "analysis/results/paper_suite")) or (_REPO_ROOT / "analysis/results/paper_suite")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _iter_summary_figures(output_root: Path) -> Iterable[Path]:
    for summary in (
        output_root / "visualization_summary.json",
        output_root / "synthetic_calibration" / "visualization_summary.json",
        output_root / "c4_nnopt_vs_mspd" / "c4_nnopt_vs_mspd_summary.json",
        output_root / "c4_nnopt_vs_mspd" / "nnopt_vs_mspd_summary.json",
        output_root / "nnopt_vs_mspd" / "nnopt_vs_mspd_summary.json",
    ):
        if not summary.exists():
            continue
        payload = _read_json(summary)
        figure_paths = payload.get("figure_paths", {})
        if isinstance(figure_paths, dict):
            for raw in figure_paths.values():
                path = _resolve(str(raw)) if raw else None
                if path is not None:
                    yield path
        raw_figure = payload.get("figure")
        path = _resolve(str(raw_figure)) if raw_figure else None
        if path is not None:
            yield path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _iter_all_figures(output_root: Path, *, out_dir: Path) -> Iterable[Path]:
    if not output_root.exists():
        return
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        if _is_under(path, out_dir):
            continue
        if path.suffix.lower() in FIGURE_SUFFIXES:
            yield path


def _stable_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _sanitize(value: str) -> str:
    value = value.strip().replace(os.sep, "__").replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "figure"


def _dest_name(path: Path, output_root: Path, used: set[str]) -> str:
    rel = _stable_rel(path, output_root)
    if rel.startswith("figures/"):
        base = Path(rel).name
    else:
        base = _sanitize(rel)
    if base not in used:
        used.add(base)
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    digest = hashlib.sha256(path.resolve().as_posix().encode("utf-8")).hexdigest()[:8]
    name = f"{stem}__{digest}{suffix}"
    used.add(name)
    return name


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> str:
    if dst.exists() and not overwrite:
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_latex_snippets(path: Path, rows: list[dict[str, Any]]) -> None:
    copied = [row for row in rows if row.get("kind") == "figure" and row.get("status") in {"copied", "exists"}]
    copied.sort(key=lambda row: str(row.get("dest_name", "")))
    lines = [
        "% Auto-generated by scripts/collect_paper_suite_figures.py",
        "% Copy this directory into Overleaf and include files by filename.",
        "",
    ]
    for row in copied:
        dest = str(row["dest_name"])
        label = Path(dest).stem.lower().replace("_", "-")
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"  \centering",
                rf"  \includegraphics[width=\linewidth]{{{dest}}}",
                rf"  \caption{{TODO: {Path(dest).stem.replace('_', ' ')}}}",
                rf"  \label{{fig:{label}}}",
                r"\end{figure}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect(
    config_path: str | Path,
    *,
    output_dir: str | Path | None,
    smoke: bool = False,
    clean: bool = False,
    overwrite: bool = True,
    include_tables: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    config = Path(config_path)
    if not config.is_absolute():
        config = _REPO_ROOT / config
    output_root = _output_root(config, smoke=smoke)
    out_dir = _resolve(output_dir) if output_dir else output_root / "paper_figures_for_overleaf"
    assert out_dir is not None

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found: dict[Path, set[str]] = {}
    for path in _iter_summary_figures(output_root):
        found.setdefault(path, set()).add("summary")
    for path in _iter_all_figures(output_root, out_dir=out_dir):
        found.setdefault(path, set()).add("recursive")
    for rel in EXPECTED_MAIN_FIGURES:
        path = output_root / rel
        found.setdefault(path, set()).add("expected")

    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    missing_expected: list[str] = []
    for src in sorted(found, key=lambda p: _stable_rel(p, output_root)):
        sources = ",".join(sorted(found[src]))
        rel = _stable_rel(src, output_root)
        if not src.exists():
            status = "missing"
            dest_name = ""
            dest_path = ""
            if "expected" in found[src]:
                missing_expected.append(rel)
        else:
            dest_name = _dest_name(src, output_root, used_names)
            dest = out_dir / dest_name
            status = _copy_file(src, dest, overwrite=overwrite)
            dest_path = str(dest)
        rows.append(
            {
                "kind": "figure",
                "status": status,
                "source_tags": sources,
                "source_path": str(src),
                "source_rel": rel,
                "dest_name": dest_name,
                "dest_path": dest_path,
                "size_bytes": src.stat().st_size if src.exists() else "",
            }
        )

    if include_tables:
        table_dir = out_dir / "tables"
        missing_supporting: list[str] = []
        for rel in SUPPORTING_TABLES:
            src = output_root / rel
            status = "missing"
            dest_name = ""
            dest_path = ""
            if src.exists():
                dest_name = _sanitize(rel)
                dest = table_dir / dest_name
                status = _copy_file(src, dest, overwrite=overwrite)
                dest_path = str(dest)
            else:
                missing_supporting.append(rel)
            rows.append(
                {
                    "kind": "table",
                    "status": status,
                    "source_tags": "supporting",
                    "source_path": str(src),
                    "source_rel": rel,
                    "dest_name": f"tables/{dest_name}" if dest_name else "",
                    "dest_path": dest_path,
                    "size_bytes": src.stat().st_size if src.exists() else "",
                }
            )
    else:
        missing_supporting = []

    _write_csv(
        out_dir / "manifest.csv",
        rows,
        fieldnames=["kind", "status", "source_tags", "source_rel", "source_path", "dest_name", "dest_path", "size_bytes"],
    )
    _write_latex_snippets(out_dir / "latex_include_snippets.tex", rows)

    copied_figures = [row for row in rows if row["kind"] == "figure" and row["status"] in {"copied", "exists"}]
    missing_figures = [row for row in rows if row["kind"] == "figure" and row["status"] == "missing"]
    summary = {
        "status": "ok",
        "output_root": str(output_root),
        "output_dir": str(out_dir),
        "n_figures": len(copied_figures),
        "n_missing_figures": len(missing_figures),
        "missing_expected": missing_expected,
        "missing_supporting": missing_supporting,
        "manifest": str(out_dir / "manifest.csv"),
        "latex_include_snippets": str(out_dir / "latex_include_snippets.tex"),
    }
    (out_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if strict and (missing_expected or missing_supporting):
        message_parts = []
        if missing_expected:
            message_parts.append("missing expected paper figures:\n  - " + "\n  - ".join(missing_expected))
        if missing_supporting:
            message_parts.append("missing supporting paper tables/summaries:\n  - " + "\n  - ".join(missing_supporting))
        raise RuntimeError(
            "Paper figure collection is incomplete. "
            "Run the relevant metrics/visualization tasks first, then rerun this collector.\n"
            + "\n".join(message_parts)
        )
    if not copied_figures:
        raise RuntimeError(f"No figures copied from {output_root}. Run visualization first.")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect paper-suite figures into one Overleaf-ready directory.")
    parser.add_argument("config", help="Path to experiments/paper_suite/config.yaml")
    parser.add_argument("--output-dir", default=None, help="Destination directory. Default: <output_root>/paper_figures_for_overleaf")
    parser.add_argument("--smoke", action="store_true", help="Collect from smoke output_root.")
    parser.add_argument("--clean", action="store_true", help="Delete destination directory before copying.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing copied files.")
    parser.add_argument("--no-tables", action="store_true", help="Do not copy supporting CSV/JSON tables.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail on missing expected figures/tables.")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            collect(
                args.config,
                output_dir=args.output_dir,
                smoke=args.smoke,
                clean=args.clean,
                overwrite=not args.no_overwrite,
                include_tables=not args.no_tables,
                strict=not args.allow_missing,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
