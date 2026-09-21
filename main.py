#!/usr/bin/env python3
"""Unified GBSC search and fixed-parameter reproduction entry point."""

from __future__ import annotations

import os

# Dataset-level multiprocessing is used, so keep each worker's BLAS bounded.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist
from sklearn.cluster._spectral import discretize
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import _deterministic_vector_sign_flip

from utils import clustering_metrics, format_value, load_dataset, plot_clustering


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (PROJECT_ROOT / "../data/D-Spec").resolve()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"
SEEDS = (42, 3407, 4079, 2024, 0)
DELTA_POWERS = tuple(range(-5, 6))
DATASETS = {
    "spiral": DATA_ROOT / "spiral.mat",
    "4C": DATA_ROOT / "4C.mat",
    "AC": DATA_ROOT / "AC.mat",
    "RingG": DATA_ROOT / "RingG.mat",
    "complex9": DATA_ROOT / "complex9.mat",
    "cure-t2-4k": DATA_ROOT / "cure-t2-4k.mat",
    "landsat": DATA_ROOT / "landsat.mat",
    "spam": DATA_ROOT / "spam.mat",
    "waveform3": DATA_ROOT / "waveform3.mat",
    "pendigits": DATA_ROOT / "pendigits.mat",
    "USPS": DATA_ROOT / "USPS.mat",
    "letters": DATA_ROOT / "letters.mat",
    "MNIST": DATA_ROOT / "mnist.mat",
    "skin": DATA_ROOT / "skin.mat",
    "covertype": DATA_ROOT / "covertype.mat",
}


@dataclass
class PreparedDataset:
    name: str
    labels: np.ndarray
    projected: np.ndarray
    sample_ball: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    n_samples: int
    n_features: int
    n_clusters: int
    preparation_seconds: float


@dataclass
class AffinityGeometry:
    backend: str
    centers: np.ndarray
    radii: np.ndarray
    dense_distances: np.ndarray | None = None
    rows: np.ndarray | None = None
    columns: np.ndarray | None = None
    neighbor_distances: np.ndarray | None = None


def _diameter_pair_small(points: np.ndarray) -> tuple[int, int]:
    gram = points @ points.T
    diagonal = np.diag(gram)
    squared = diagonal[:, None] + diagonal[None, :] - 2 * gram
    flat = int(np.nanargmax(squared))
    return tuple(map(int, np.unravel_index(flat, squared.shape)))


def _cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    return float(
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def _diameter_pair_hull(points: np.ndarray) -> tuple[int, int]:
    try:
        hull_indices = ConvexHull(points).vertices
    except QhullError:
        axis = int(np.argmax(np.ptp(points, axis=0)))
        return int(np.argmin(points[:, axis])), int(np.argmax(points[:, axis]))

    polygon = points[hull_indices]
    count = len(polygon)
    if count <= 2:
        return int(hull_indices[0]), int(hull_indices[-1])

    best_i, best_j, best_distance = 0, 1, -1.0
    opposite = 1
    for index in range(count):
        following = (index + 1) % count
        while True:
            next_opposite = (opposite + 1) % count
            current_area = abs(_cross(polygon[index], polygon[following], polygon[opposite]))
            next_area = abs(
                _cross(polygon[index], polygon[following], polygon[next_opposite])
            )
            if next_area <= current_area:
                break
            opposite = next_opposite
        for candidate_i in (index, following):
            distance = float(np.sum((polygon[candidate_i] - polygon[opposite]) ** 2))
            if distance > best_distance:
                best_i, best_j, best_distance = candidate_i, opposite, distance
    return int(hull_indices[best_i]), int(hull_indices[best_j])


def farthest_pair(points: np.ndarray) -> tuple[int, int]:
    if len(points) <= 2048:
        return _diameter_pair_small(points)
    return _diameter_pair_hull(points)


def ball_statistics(projected: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, float, float]:
    points = projected[indices]
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(distances.max())
    distance_sum = float(distances.sum())
    density = float(len(indices) / distance_sum) if distance_sum else float(len(indices))
    return center, radius, density


def split_ball(projected: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points = projected[indices]
    first_local, second_local = farthest_pair(points)
    if first_local == second_local:
        return None
    first_distance = np.linalg.norm(points - points[first_local], axis=1)
    second_distance = np.linalg.norm(points - points[second_local], axis=1)
    first_mask = first_distance < second_distance
    if not first_mask.any() or first_mask.all():
        return None
    return indices[first_mask], indices[~first_mask]


def generate_balls(projected: np.ndarray) -> list[np.ndarray]:
    balls = [np.arange(len(projected), dtype=np.int64)]
    while True:
        divided: list[np.ndarray] = []
        for indices in balls:
            if len(indices) < 8:
                divided.append(indices)
                continue
            children = split_ball(projected, indices)
            if children is None:
                divided.append(indices)
                continue
            first, second = children
            _, _, parent_density = ball_statistics(projected, indices)
            _, _, first_density = ball_statistics(projected, first)
            _, _, second_density = ball_statistics(projected, second)
            weighted = (
                len(first) * first_density + len(second) * second_density
            ) / len(indices)
            if weighted > parent_density:
                divided.extend((first, second))
            else:
                divided.append(indices)
        if len(divided) == len(balls):
            break
        balls = divided

    radii = [
        ball_statistics(projected, indices)[1]
        for indices in balls
        if len(indices) >= 2
    ]
    if not radii:
        return balls
    threshold = max(float(np.mean(radii)), float(np.median(radii)))
    while True:
        divided = []
        for indices in balls:
            if len(indices) < 2 or ball_statistics(projected, indices)[1] <= 2 * threshold:
                divided.append(indices)
                continue
            children = split_ball(projected, indices)
            divided.extend(children if children is not None else (indices,))
        if len(divided) == len(balls):
            break
        balls = divided
    return balls


def prepare_dataset(name: str, path: Path) -> PreparedDataset:
    started = perf_counter()
    dataset = load_dataset(path, name=name)
    if dataset.n_features == 2:
        projected = np.asarray(dataset.features, dtype=np.float64)
    else:
        projected = PCA(n_components=2, random_state=42).fit_transform(dataset.features)
        projected = np.asarray(projected, dtype=np.float64)
    balls = generate_balls(projected)

    centers = np.empty((len(balls), 2), dtype=np.float64)
    radii = np.empty(len(balls), dtype=np.float64)
    sample_ball = np.empty(dataset.n_samples, dtype=np.int64)
    for ball_index, indices in enumerate(balls):
        center, ball_radius, _ = ball_statistics(projected, indices)
        centers[ball_index] = center
        radii[ball_index] = ball_radius
        sample_ball[indices] = ball_index

    return PreparedDataset(
        name=name,
        labels=dataset.labels,
        projected=projected,
        sample_ball=sample_ball,
        centers=centers,
        radii=radii,
        n_samples=dataset.n_samples,
        n_features=dataset.n_features,
        n_clusters=dataset.n_clusters,
        preparation_seconds=perf_counter() - started,
    )


def build_geometry(
    prepared: PreparedDataset, dense_limit: int, neighbors: int
) -> AffinityGeometry:
    centers, radii = prepared.centers, prepared.radii
    if len(centers) <= dense_limit:
        return AffinityGeometry(
            backend="full",
            centers=centers,
            radii=radii,
            dense_distances=cdist(centers, centers),
        )

    neighbor_count = min(neighbors + 1, len(centers))
    distances, indices = NearestNeighbors(
        n_neighbors=neighbor_count, algorithm="kd_tree", n_jobs=1
    ).fit(centers).kneighbors(centers)
    rows = np.repeat(np.arange(len(centers), dtype=np.int64), neighbor_count - 1)
    return AffinityGeometry(
        backend=f"knn-{neighbor_count - 1}",
        centers=centers,
        radii=radii,
        rows=rows,
        columns=indices[:, 1:].reshape(-1),
        neighbor_distances=distances[:, 1:].reshape(-1),
    )


def build_affinity(geometry: AffinityGeometry, delta: float):
    denominator = 2 * delta**2
    if geometry.backend == "full":
        exponent = (
            geometry.radii[:, None]
            + geometry.radii[None, :]
            - geometry.dense_distances
        ) / denominator
        np.fill_diagonal(exponent, -np.inf)
        exponent -= float(np.max(exponent))
        affinity = np.exp(np.clip(exponent, -700.0, 0.0))
        np.fill_diagonal(affinity, 0.0)
        return affinity

    rows = geometry.rows
    columns = geometry.columns
    exponent = (
        geometry.radii[rows]
        + geometry.radii[columns]
        - geometry.neighbor_distances
    ) / denominator
    exponent -= float(np.max(exponent))
    values = np.exp(np.clip(exponent, -700.0, 0.0))
    affinity = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(geometry.centers), len(geometry.centers)),
    )
    affinity = affinity.maximum(affinity.T)
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity


def spectral_labels(affinity, n_clusters: int, seed: int) -> np.ndarray:
    if sparse.issparse(affinity):
        embedding = spectral_embedding(
            affinity,
            n_components=n_clusters,
            eigen_solver="arpack",
            random_state=seed,
            eigen_tol=1e-5,
            drop_first=False,
        )
        maps = embedding
    else:
        normalized_laplacian, degree_sqrt = laplacian(
            affinity, normed=True, return_diag=True
        )
        _, eigenvectors = eigh(
            normalized_laplacian,
            subset_by_index=(0, n_clusters - 1),
            check_finite=False,
            driver="evr",
        )
        embedding = _deterministic_vector_sign_flip(eigenvectors.T / degree_sqrt)
        maps = embedding.T

    row_norms = np.linalg.norm(maps, axis=1)
    degenerate = (~np.isfinite(row_norms)) | (row_norms == 0)
    if degenerate.any():
        maps = maps.copy()
        maps[degenerate] = 0.0
        maps[degenerate, 0] = 1.0
    return discretize(maps, random_state=seed)


def cluster_prepared(
    prepared: PreparedDataset,
    geometry: AffinityGeometry,
    delta: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], float]:
    started = perf_counter()
    affinity = build_affinity(geometry, delta)
    ball_labels = spectral_labels(affinity, prepared.n_clusters, seed)
    predicted = ball_labels[prepared.sample_ball]
    fit_seconds = perf_counter() - started
    return predicted, clustering_metrics(prepared.labels, predicted), fit_seconds


def save_search(
    prepared: PreparedDataset,
    output_dir: Path,
    dense_limit: int,
    neighbors: int,
) -> dict[str, object]:
    geometry_started = perf_counter()
    geometry = build_geometry(prepared, dense_limit, neighbors)
    geometry_seconds = perf_counter() - geometry_started
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for power in DELTA_POWERS:
        delta = float(2.0**power)
        predicted, metrics, fit_seconds = cluster_prepared(
            prepared, geometry, delta, seed=42
        )
        row = {
            "power": power,
            "delta": delta,
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
            "f1": metrics["f1"],
            "time": prepared.preparation_seconds + geometry_seconds + fit_seconds,
        }
        rows.append(row)
        score = (float(row["nmi"]), float(row["ari"]), float(row["f1"]), -power)
        if best is None or score > best["score"]:
            best = {"score": score, "row": row, "labels": predicted.copy()}
        print(
            f"[{prepared.name}] 2^{power:>2} NMI={format_value(row['nmi'])} "
            f"ARI={format_value(row['ari'])} F1={format_value(row['f1'])}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("power", "delta", "nmi", "ari", "f1", "time"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "power": row["power"],
                    "delta": f"{row['delta']:.5f}",
                    "nmi": format_value(row["nmi"]),
                    "ari": format_value(row["ari"]),
                    "f1": format_value(row["f1"]),
                    "time": format_value(row["time"]),
                }
            )

    best_row = best["row"]
    best_labels = np.asarray(best["labels"], dtype=np.int64)
    np.save(output_dir / "best_labels_seed42.npy", best_labels)
    np.savetxt(
        output_dir / "best_labels_seed42.csv",
        best_labels,
        fmt="%d",
        delimiter=",",
        header="label",
        comments="",
    )
    plot_clustering(
        prepared.projected,
        prepared.labels,
        best_labels,
        output_dir / "plots",
        seed=42,
    )
    best_parameters = {
        "dataset": prepared.name,
        "best_power": int(best_row["power"]),
        "best_delta": float(best_row["delta"]),
        "selection_seed": 42,
        "selection_metric": "nmi_then_ari_then_f1",
        "affinity_backend": geometry.backend,
        "n_balls": int(len(prepared.centers)),
        "search_nmi": float(best_row["nmi"]),
        "search_ari": float(best_row["ari"]),
        "search_f1": float(best_row["f1"]),
    }
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(best_parameters, handle, indent=2, ensure_ascii=False)
    return best_parameters


def reproduce_dataset(
    name: str,
    path: Path,
    output_dir: Path,
    best_parameters: dict[str, object],
    dense_limit: int,
    neighbors: int,
) -> dict[str, object]:
    delta = float(best_parameters["best_delta"])
    runs: list[dict[str, object]] = []
    metadata: PreparedDataset | None = None
    backend = ""
    n_balls = 0
    for seed in SEEDS:
        prepared = prepare_dataset(name, path)
        metadata = prepared
        geometry_started = perf_counter()
        geometry = build_geometry(prepared, dense_limit, neighbors)
        geometry_seconds = perf_counter() - geometry_started
        _, metrics, fit_seconds = cluster_prepared(prepared, geometry, delta, seed)
        total_seconds = prepared.preparation_seconds + geometry_seconds + fit_seconds
        backend = geometry.backend
        n_balls = len(prepared.centers)
        runs.append(
            {
                "seed": seed,
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
                "f1": metrics["f1"],
                "time": total_seconds,
            }
        )
        print(
            f"[{name}] seed={seed} NMI={format_value(metrics['nmi'])} "
            f"ARI={format_value(metrics['ari'])} F1={format_value(metrics['f1'])} "
            f"time={format_value(total_seconds)}",
            flush=True,
        )

    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "nmi", "ari", "f1", "time"),
            lineterminator="\n",
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "seed": run["seed"],
                    "nmi": format_value(run["nmi"]),
                    "ari": format_value(run["ari"]),
                    "f1": format_value(run["f1"]),
                    "time": format_value(run["time"]),
                }
            )

    summary: dict[str, object] = {
        "dataset": name,
        "n_samples": metadata.n_samples,
        "n_features": metadata.n_features,
        "n_clusters": metadata.n_clusters,
        "n_balls": n_balls,
        "affinity_backend": backend,
        "best_power": int(best_parameters["best_power"]),
        "best_delta": delta,
    }
    for metric in ("nmi", "ari", "f1", "time"):
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def run_dataset(
    name: str,
    path_string: str,
    output_root_string: str,
    mode: str,
    dense_limit: int,
    neighbors: int,
) -> dict[str, object] | None:
    path = Path(path_string)
    output_dir = Path(output_root_string) / name
    if mode in {"search", "all"}:
        prepared = prepare_dataset(name, path)
        print(
            f"[{name}] n={prepared.n_samples} d={prepared.n_features} "
            f"k={prepared.n_clusters} balls={len(prepared.centers)}",
            flush=True,
        )
        best_parameters = save_search(prepared, output_dir, dense_limit, neighbors)
    else:
        parameter_path = output_dir / "best_params.json"
        if not parameter_path.is_file():
            raise FileNotFoundError(f"Run search first: {parameter_path}")
        with parameter_path.open(encoding="utf-8") as handle:
            best_parameters = json.load(handle)

    if mode in {"reproduce", "all"}:
        return reproduce_dataset(
            name, path, output_dir, best_parameters, dense_limit, neighbors
        )
    return None


def aggregate(output_root: Path, destination: Path) -> None:
    fieldnames = (
        "dataset",
        "n_samples",
        "n_features",
        "n_clusters",
        "n_balls",
        "affinity_backend",
        "best_power",
        "best_delta",
        "nmi_mean",
        "nmi_std",
        "ari_mean",
        "ari_std",
        "f1_mean",
        "f1_std",
        "time_mean",
        "time_std",
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name in DATASETS:
            summary_path = output_root / name / "summary.json"
            if not summary_path.is_file():
                continue
            with summary_path.open(encoding="utf-8") as summary_file:
                summary = json.load(summary_file)
            row = {key: summary[key] for key in fieldnames}
            row["best_delta"] = f"{float(row['best_delta']):.5f}"
            for key in fieldnames:
                if key.endswith(("_mean", "_std")):
                    row[key] = format_value(row[key])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("search", "reproduce", "all"))
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASETS), default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dense-limit", type=int, default=6000)
    parser.add_argument("--neighbors", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.datasets or list(DATASETS)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                run_dataset,
                name,
                str(DATASETS[name]),
                str(output_root),
                args.mode,
                args.dense_limit,
                args.neighbors,
            ): name
            for name in selected
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                failures[name] = f"{type(error).__name__}: {error}"
                print(f"[{name}] FAILED: {failures[name]}", flush=True)
                traceback.print_exception(error)

    if args.mode in {"reproduce", "all"}:
        aggregate(output_root, PROJECT_ROOT / "gbsc.csv")
    if failures:
        with (output_root / "failures.json").open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2, ensure_ascii=False)
        return 1
    failure_path = output_root / "failures.json"
    if failure_path.exists():
        failure_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
