"""Copy current measured artifacts; full publication requires complete valid runs."""

import csv
import json
import re
import shutil
from pathlib import Path

from compbench.datasets.ensemble import variables
from compbench.execution.validation import validate
from compbench.figures.io import FIGURES, RENDERER_VERSION
from compbench.results import save


def export_figures(cfg, opts, tier, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    workflows = [
        "rate_distortion",
        "error_statistics",
        "kappa_study",
        "quantization_residual",
        "residual_layer_ablation",
        "case_study_hurricane",
        "sequential_throughput",
        "weak_scaling",
        "energy_budget_closure",
        "trajectory_simulation",
    ]
    source = Path(opts["output_root"]) / tier
    manifest = {}
    sources = {}
    skipped = {}
    for workflow in workflows:
        root = source / workflow
        if not (root / "run.json").exists():
            if tier == "full":
                raise ValueError(f"Missing full workflow: {workflow}")
            continue
        run = json.loads((root / "run.json").read_text())
        selected = run["specification"]["fields"]
        if tier == "full" and selected:
            raise ValueError(f"Diagnostic selection in full {workflow}")
        try:
            validate(cfg, opts, workflow, tier, selected, allow_bound_violations=tier == "sample")
        except (ValueError, FileNotFoundError) as exc:
            if tier == "full":
                raise
            skipped[workflow] = str(exc)
            continue
        for path in sorted((root / "figures").glob("*.pdf")):
            name = path.name
            main_panels = {stem + ".pdf" for stems in FIGURES.values() for stem in stems}
            if name not in main_panels:
                if re.fullmatch(
                    r"(histogram|spectrum)_(pl|sfc)_cr_[0-9]+\.pdf", name
                ) or re.fullmatch(
                    r"casestudy_(u_component_of_wind|divergence)_850_cr(10|20|50)(_delta)?\.pdf",
                    name,
                ):
                    name = "figs_si/" + name
                else:
                    continue
            sidecar = path.with_suffix(".json")
            if (
                not sidecar.exists()
                or json.loads(sidecar.read_text()).get("renderer_version") != RENDERER_VERSION
            ):
                if tier == "full":
                    raise ValueError(
                        f"Figure must be rerendered using the current renderer: {path}"
                    )
                continue
            target = destination / name
            if target.exists() and path.read_bytes() != target.read_bytes():
                raise ValueError(
                    f"Export would replace a different artifact: {target}; use a fresh destination"
                )
            sources[name] = path
            manifest[name] = dict(
                workflow=workflow,
                run=str(root),
                tier=tier,
                validation=json.loads((root / "validation.json").read_text()),
            )
    if tier == "full":
        required = {stem + ".pdf" for stems in FIGURES.values() for stem in stems}
        required |= {
            f"figs_si/{kind}_{group}_cr_{ratio}.pdf"
            for kind in ["histogram", "spectrum"]
            for group in ["pl", "sfc"]
            for ratio in cfg["error_statistics"]["ratios"]
        }
        required |= {
            f"figs_si/casestudy_{variable}_850_cr{ratio}{suffix}.pdf"
            for variable in ["u_component_of_wind", "divergence"]
            for ratio in [10, 20, 50]
            for suffix in ["", "_delta"]
        }
        missing = required - set(manifest)
        if missing:
            raise ValueError(f"Missing full-run figures: {sorted(missing)}")
    for name, path in sources.items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination / name)
        shutil.copy2(path.with_suffix(".json"), (destination / name).with_suffix(".json"))
    save(
        destination / "panel_map.json",
        {
            number: [
                dict(file=stem + ".pdf", available=stem + ".pdf" in manifest) for stem in stems
            ]
            for number, stems in FIGURES.items()
        },
    )
    with (destination / "table_3_variables.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["short_name", "source_variable"])
        pressure, surface = variables(cfg, "energy_budget_closure")
        writer.writerows({**pressure, **surface}.items())
    save(
        destination / "manifest.json",
        dict(tier=tier, artifacts=manifest, skipped=skipped),
    )
    (destination / "README.txt").write_text(
        "Sample source-panel PDFs. See panel_map.json for figure mapping and manifest.json for bound failures and coverage.\n"
        if tier == "sample"
        else "Full validated source-panel PDFs. See panel_map.json and manifest.json.\n"
    )
    return destination
