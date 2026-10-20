from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

NPZ: Final[Path] = Path("results/geometry_showcase_data.npz")
OUTPUT: Final[Path] = Path("results/fig_geometry_showcase.pdf")
CURVED_RUN: Final[str] = "baseline_csgo_I_T_Tab_rvicreg_learned_seed456"
FLAT_RUN: Final[str] = "baseline_csgo_I_T_Tab_vicreg_seed456"
CLASS_NAMES: Final[tuple[str, ...]] = (
    "Classified",
    "Consumer",
    "Covert",
    "Industrial",
    "Mil-spec",
    "Restricted",
)
CLASS_COLORS: Final[tuple[str, ...]] = (
    "#D1495B",
    "#EDAE49",
    "#0F7173",
    "#30638E",
    "#8E4D9D",
    "#66A182",
)
FACTORS: Final[int] = 64
FACTOR_DIM: Final[int] = 4
RING_RADII: Final[tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0, 5.0)
RING_COLOR: Final[str] = "#7A7F87"


def load_arrays(runs: Path, run: str) -> tuple[FloatArray, IntArray, FloatArray]:
    path = next(iter(runs.glob(f"{run}/*/pretrain_train.npz")))
    archive = np.load(path)
    z = archive["projector_unmasked"].astype(np.float64)
    labels = archive["labels"].astype(np.int64)
    kappas = archive["kappas"].astype(np.float64)
    return z, labels, kappas


def load_bundled(npz: Path) -> tuple[FloatArray, IntArray, FloatArray, FloatArray, IntArray]:
    archive = np.load(npz)
    return (
        archive["curved_projector"].astype(np.float64),
        archive["curved_labels"].astype(np.int64),
        archive["curved_kappas"].astype(np.float64),
        archive["flat_projector"].astype(np.float64),
        archive["flat_labels"].astype(np.int64),
    )


def best_factor(z: FloatArray, kappas: FloatArray) -> int:
    factors = z.reshape(z.shape[0], FACTORS, FACTOR_DIM)
    norms = np.linalg.norm(factors, axis=2)
    sqrt_k = np.sqrt(np.abs(kappas))[None, :]
    utilization = np.tanh(sqrt_k * norms).max(axis=0)
    return int(np.argmax(utilization))


def pca_directions(tangent: FloatArray) -> tuple[FloatArray, float]:
    centered = tangent - tangent.mean(axis=0, keepdims=True)
    _, singular, rows = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ rows[:2].T
    norms = np.linalg.norm(projected, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    explained = float(np.sum(singular[:2] ** 2) / np.sum(singular**2))
    return cast(FloatArray, projected / norms[:, None]), explained


def draw_classes(
    ax: Axes,
    points: FloatArray,
    labels: IntArray,
    size: float,
) -> None:
    order = np.argsort(-np.bincount(labels, minlength=len(CLASS_NAMES)))
    for class_index in order:
        mask = labels == class_index
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=size,
            color=CLASS_COLORS[class_index],
            alpha=0.85,
            linewidth=0.0,
            zorder=4,
        )


def draw_flat_disk(
    ax: Axes,
    directions: FloatArray,
    radii: FloatArray,
    labels: IntArray,
) -> None:
    limit = float(np.max(radii)) * 1.06
    points = directions * radii[:, None]
    for ring in RING_RADII:
        if ring < limit:
            ax.add_patch(
                Circle(
                    (0.0, 0.0),
                    ring,
                    facecolor="none",
                    edgecolor=RING_COLOR,
                    linewidth=0.7,
                    linestyle=(0, (4, 3)),
                    zorder=2,
                )
            )
            ax.text(
                0.04,
                ring - 0.045 * limit,
                f"{ring:.0f}",
                fontsize=6.6,
                color=RING_COLOR,
                ha="left",
                zorder=5,
            )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            limit,
            facecolor="none",
            edgecolor="#B9BDC4",
            linewidth=1.0,
            zorder=2,
        )
    )
    draw_classes(ax, points, labels, 7.5)
    ax.set_xlim(-limit * 1.04, limit * 1.04)
    ax.set_ylim(-limit * 1.04, limit * 1.04)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "(a) flat metric (VICReg)\nequal-distance rings: evenly spaced",
        fontsize=9.0,
        fontweight="bold",
    )


def draw_poincare_disk(
    ax: Axes,
    directions: FloatArray,
    geodesic_radii: FloatArray,
    labels: IntArray,
    kappa: float,
) -> None:
    sqrt_k = float(np.sqrt(abs(kappa)))
    boundary = 1.0 / sqrt_k
    ball_radii = np.tanh(sqrt_k * geodesic_radii / 2.0) / sqrt_k
    points = directions * ball_radii[:, None]
    for ring in RING_RADII:
        euclidean = float(np.tanh(sqrt_k * ring / 2.0) / sqrt_k)
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                euclidean,
                facecolor="none",
                edgecolor=RING_COLOR,
                linewidth=0.7,
                linestyle=(0, (4, 3)),
                zorder=2,
            )
        )
        ax.text(
            0.02,
            euclidean - 0.05 * boundary,
            f"{ring:.0f}",
            fontsize=6.6,
            color=RING_COLOR,
            ha="left",
            zorder=5,
        )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            boundary,
            facecolor="none",
            edgecolor="#0F7173",
            linewidth=1.6,
            zorder=3,
        )
    )
    draw_classes(ax, points, labels, 7.5)
    ax.set_xlim(-boundary * 1.1, boundary * 1.1)
    ax.set_ylim(-boundary * 1.1, boundary * 1.1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        "(b) learned hyperbolic metric (R-VICReg)\nsame rings: crowded at the boundary",
        fontsize=9.0,
        fontweight="bold",
    )


def draw_hyperboloid(
    axis: Axes3D,
    directions: FloatArray,
    geodesic_radii: FloatArray,
    labels: IntArray,
    kappa: float,
    factor: int,
) -> None:
    sqrt_k = float(np.sqrt(abs(kappa)))
    unit_radii = np.tanh(sqrt_k * geodesic_radii / 2.0)
    unit_points = directions * unit_radii[:, None]
    norm_sq = np.sum(unit_points**2, axis=1)
    denominator = 1.0 - norm_sq
    lifted_xy = 2.0 * unit_points / denominator[:, None] / sqrt_k
    lifted_z = (1.0 + norm_sq) / denominator / sqrt_k
    max_unit = float(np.quantile(unit_radii, 0.94))
    grid_r = np.tanh(np.linspace(0.0, np.arctanh(max_unit), 36))
    grid_theta = np.linspace(0.0, 2.0 * np.pi, 90)
    mesh_r, mesh_theta = np.meshgrid(grid_r, grid_theta)
    mesh_den = 1.0 - mesh_r**2
    surface_x = 2.0 * mesh_r * np.cos(mesh_theta) / mesh_den / sqrt_k
    surface_y = 2.0 * mesh_r * np.sin(mesh_theta) / mesh_den / sqrt_k
    surface_z = (1.0 + mesh_r**2) / mesh_den / sqrt_k
    axis.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        color="#0F7173",
        alpha=0.08,
        linewidth=0.0,
        antialiased=True,
    )
    axis.plot_wireframe(
        surface_x,
        surface_y,
        surface_z,
        rstride=15,
        cstride=6,
        color="#0F7173",
        linewidth=0.3,
        alpha=0.4,
    )
    boundary = 1.0 / sqrt_k
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    axis.plot(
        boundary * np.cos(theta),
        boundary * np.sin(theta),
        np.zeros_like(theta),
        color="#0F7173",
        linewidth=1.3,
        alpha=0.9,
    )
    ball_points = unit_points / sqrt_k
    keep = np.isfinite(lifted_z) & (unit_radii <= max_unit)
    rng = np.random.default_rng(456)
    ray_indices = rng.choice(np.flatnonzero(keep), size=30, replace=False)
    for index in ray_indices:
        axis.plot(
            (ball_points[index, 0], lifted_xy[index, 0]),
            (ball_points[index, 1], lifted_xy[index, 1]),
            (0.0, lifted_z[index]),
            color="#9AA1A9",
            linewidth=0.5,
            alpha=0.6,
        )
    order = np.argsort(-np.bincount(labels, minlength=len(CLASS_NAMES)))
    for class_index in order:
        mask = (labels == class_index) & keep
        axis.scatter(
            lifted_xy[mask, 0],
            lifted_xy[mask, 1],
            lifted_z[mask],
            s=8,
            color=CLASS_COLORS[class_index],
            alpha=0.9,
            depthshade=False,
        )
        axis.scatter(
            ball_points[mask, 0],
            ball_points[mask, 1],
            np.zeros(int(mask.sum())),
            s=5,
            color=CLASS_COLORS[class_index],
            alpha=0.45,
            depthshade=False,
        )
    axis.set_title(
        rf"(c) hyperboloid ($\kappa_{{{factor}}}={kappa:.3f}$)"
        + "\nsheet above, Poincar\u00e9 shadow below",
        fontsize=9.0,
        fontweight="bold",
    )
    axis.set_axis_off()
    axis.view_init(elev=17, azim=42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=NPZ)
    args = parser.parse_args()
    runs: Path | None = args.runs
    npz: Path = args.npz

    plt.rcParams.update({"font.size": 9.0})
    if runs is not None:
        curved_z, labels, kappas = load_arrays(runs, CURVED_RUN)
        flat_z, flat_labels, _ = load_arrays(runs, FLAT_RUN)
    else:
        curved_z, labels, kappas, flat_z, flat_labels = load_bundled(npz)
    factor = best_factor(curved_z, kappas)
    kappa = float(kappas[factor])
    tangent = curved_z.reshape(-1, FACTORS, FACTOR_DIM)[:, factor, :]
    sqrt_k = float(np.sqrt(abs(kappa)))
    tangent_norms = np.linalg.norm(tangent, axis=1)
    ball_norms = np.clip(np.tanh(sqrt_k * tangent_norms), 0.0, 1.0 - 1e-12)
    factor_radii = cast(FloatArray, 2.0 * np.arctanh(ball_norms) / sqrt_k)
    directions, explained = pca_directions(tangent)
    flat_directions, flat_explained = pca_directions(flat_z)
    flat_radii = cast(FloatArray, np.linalg.norm(flat_z, axis=1) / np.sqrt(flat_z.shape[1]) * 2.0)

    figure: Figure = plt.figure(figsize=(10.6, 4.0))
    flat_axis = figure.add_subplot(1, 3, 1)
    disk_axis = figure.add_subplot(1, 3, 2)
    hyper_axis = cast(Axes3D, figure.add_subplot(1, 3, 3, projection="3d"))
    draw_flat_disk(flat_axis, flat_directions, flat_radii, flat_labels)
    draw_poincare_disk(disk_axis, directions, factor_radii, labels, kappa)
    draw_hyperboloid(hyper_axis, directions, factor_radii, labels, kappa, factor)
    handles = [
        Line2D((0,), (0,), marker="o", linestyle="", color=color, markersize=5, label=name)
        for name, color in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        "CSGO trimodal (image+text+tabular) projector embeddings, seed 456: two learned geometries",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.01, right=0.995, top=0.80, bottom=0.10, wspace=0.10)
    figure.savefig(OUTPUT, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(figure)
    shell_fraction = float(np.mean(factor_radii >= 2.0))
    median_radius = float(np.median(factor_radii))
    hyperbolic_volume = (np.sinh(sqrt_k * median_radius) / sqrt_k) ** (FACTOR_DIM - 1)
    euclidean_volume = median_radius ** (FACTOR_DIM - 1)
    print("factor %d kappa %.6f" % (factor, kappa))
    print("tangent PCA explained variance: curved %.4f, flat %.4f" % (explained, flat_explained))
    print("median geodesic radius %.4f; fraction beyond r=2: %.4f" % (median_radius, shell_fraction))
    print(
        "shell area ratio at median radius (hyperbolic vs flat, per 4-D factor): %.2f"
        % (hyperbolic_volume / euclidean_volume)
    )
    print("output", OUTPUT)


if __name__ == "__main__":
    main()
