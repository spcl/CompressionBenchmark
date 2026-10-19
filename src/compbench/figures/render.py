"""Dispatch source-panel rendering from saved benchmark measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compbench.figures.error_statistics import histograms
from compbench.figures.io import finish
from compbench.figures.quality import rd
from compbench.results import record_batches, records


def plot(cfg, opts, workflow, tier):
    output = Path(opts["output_root"]) / tier / workflow
    # Separate legacy scripts ran in fresh processes: do not leak styles between figures.
    with plt.rc_context(matplotlib.rcParamsDefault):
        if workflow == "rate_distortion":
            rd((r for _, batch in record_batches(output) for r in batch), output, cfg, tier)
        elif workflow in ("kappa_study", "quantization_residual"):
            from compbench.figures import ebcc_comparisons

            name, draw = {
                "kappa_study": ("method_price", ebcc_comparisons.kappa_comparison),
                "quantization_residual": (
                    "ablation_quantization",
                    ebcc_comparisons.component_comparison,
                ),
            }[workflow]
            finish(draw(records(output), cfg), output, name, tier)
        elif workflow == "residual_layer_ablation":
            from compbench.figures.residual_layer_ablation import spike

            stats = json.loads((output / "measurements.json").read_text())
            stats["bound"] = cfg["residual_layer_ablation"]["bound"]
            for r in stats["records"]:
                r.update(
                    bytes=r["size"], compression_ratio=r["nbytes"] / r["size"], psnr_db=r["psnr"]
                )
            finish(spike(np.load(output / "arrays.npz"), stats), output, "method_spike", tier)
        elif workflow == "error_statistics":
            histograms(output, cfg, tier)
        elif workflow == "case_study_hurricane":
            from compbench.figures.hurricane import hurricane

            hurricane(output, cfg, tier)
        elif workflow == "energy_budget_closure":
            from compbench.figures.energy_budget_closure import plot_energy

            plot_energy(cfg, tier, output)
        elif workflow == "trajectory_simulation":
            from compbench.figures.trajectory import plot_trajectory

            plot_trajectory(cfg, tier, output)
        elif workflow in ("sequential_throughput", "weak_scaling"):
            from compbench.figures.scaling import plot_scaling

            plot_scaling(cfg, Path(opts["output_root"]) / tier, tier)
    return (
        output.parent / "weak_scaling" if workflow == "sequential_throughput" else output
    ) / "figures"
