"""Deterministic figure-panel output and provenance."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RENDERER_VERSION = 4
COLORS = dict(
    ebcc="tab:blue",
    sz="tab:orange",
    sz3="tab:green",
    hpez="tab:brown",
    sperr="tab:red",
    mgard="tab:cyan",
)


# These are pre-TeX plot assets, not assembled numbered manuscript figures.
FIGURES = {
    2: ("method_price",),
    3: ("cr_vs_ssim_lowres", "cr_vs_psnr"),
    4: ("histogram_t2m_cr_10", "histogram_t2m_cr_100"),
    5: ("spectrum_z_500_cr_10_zoom", "spectrum_z_500_cr_100_zoom"),
    6: ("weak_scaling_landscape",),
    7: ("casestudy_u_component_of_wind_850_cr100", "casestudy_u_component_of_wind_850_cr100_delta"),
    8: ("casestudy_divergence_850_cr100", "casestudy_divergence_850_cr100_delta"),
    9: ("casestudy_u_component_of_wind_850_cr_vs_ssim", "casestudy_divergence_850_cr_vs_ssim"),
    10: ("ens_energy_budget_delta_ebcc_3d",),
    11: ("energy_budget_rd_3d",),
    12: ("trajectory_rmse_rd_3d",),
    13: tuple(
        f"Fukushima.{name}_traj"
        for name in ("ens_member0", "ens_member1", "ens_ebcc_3d_rel_0_01", "ens_ebcc_3d_rel_0_05")
    ),
    14: tuple(
        f"Fukushima.{name}_density"
        for name in ("ens_member0", "ens_member1", "ens_ebcc_3d_rel_0_01", "ens_ebcc_3d_rel_0_05")
    ),
    15: ("ablation_quantization",),
    16: ("method_spike",),
}


def asset(name):
    root = Path(__file__).resolve().parents[3] / "assets"
    if not root.exists():
        root = Path(sys.prefix) / "share/compbench/assets"
    return root / name


def finish(fig, output, name, tier):
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        folder / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    provenance(output, name, tier)
    return folder / f"{name}.pdf"


def provenance(output, name, tier, panels=()):
    from importlib.metadata import version

    from compbench.results import save

    report = output / "validation.json"
    save(
        output / "figures" / f"{name}.json",
        dict(
            renderer_version=RENDERER_VERSION,
            environment={key: version(key) for key in ("matplotlib", "numpy", "cartopy", "pillow")},
            tier=tier,
            panels=[str(p) for p in panels],
            validation=json.loads(report.read_text()) if report.exists() else None,
            note="Sample inputs; full-dataset results require a full run."
            if tier == "sample"
            else "",
        ),
    )


def configure_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt
