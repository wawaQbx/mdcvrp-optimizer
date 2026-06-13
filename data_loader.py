"""
data_loader.py
==============
Loader for the SimMD (INCOM2024 / MIM2025) Multi-Depot Capacitated Vehicle
Routing Problem (MDCVRP) instances.

Each instance is a paired ``.vrp`` (TSPLIB-style) and ``.yaml`` file. This module:

  * Parses the ``.vrp`` file with **vrplib** for coordinates, demands and capacity.
    The competition files use a *non-standard* ``DEPOT_SECTION`` (each row is
    ``id x y`` instead of a bare node index), which vanilla vrplib cannot read,
    so the section is sanitised on the fly before parsing.
  * Parses the ``.yaml`` file with **pyyaml** and uses it as the authoritative
    source of metadata (depot count, capacity, dimension), cross-checking it
    against the ``.vrp`` header.
  * Builds a high-performance, symmetric Euclidean **distance matrix** with NumPy.
  * Returns a structured dictionary ready to feed straight into Google OR-Tools,
    with the start/end depot index lists derived *dynamically* from the metadata.

Convention used by these instances: the first ``DEPOTS`` nodes (1-indexed in the
file, i.e. indices ``0 .. DEPOTS-1`` 0-indexed) are the depots; every remaining
node is a customer. Depots always carry demand 0.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import vrplib
import yaml


# --------------------------------------------------------------------------- #
#  .vrp sanitising + parsing
# --------------------------------------------------------------------------- #
def _sanitize_vrp_text(text: str) -> str:
    """Rewrite the non-standard DEPOT_SECTION so vrplib can parse the file.

    The SimMD files write each depot row as ``id x y``. Standard VRPLIB expects
    the DEPOT_SECTION to contain only bare node indices terminated by ``-1``.
    We keep just the first token (the index) of every depot row and leave the
    rest of the file untouched.
    """
    out: list[str] = []
    in_depot = False
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("DEPOT_SECTION"):
            in_depot = True
            out.append(line)
            continue
        if in_depot:
            if stripped == "":
                out.append(line)
                continue
            if stripped.startswith("-1") or upper.startswith("EOF"):
                in_depot = False
                out.append(line)
                continue
            # depot row "id x y" -> keep only the node index
            out.append(stripped.split()[0])
        else:
            out.append(line)
    return "\n".join(out)


def _read_vrp(vrp_path: str) -> dict:
    """Parse a (possibly non-standard) .vrp file via vrplib.

    vrplib only reads from a path, so the sanitised text is written to a
    temporary file. ``compute_edge_weights=False`` skips vrplib's own distance
    matrix (we build our own with NumPy).
    """
    with open(vrp_path, "r") as fh:
        clean = _sanitize_vrp_text(fh.read())

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".vrp", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(clean)
        tmp.close()
        return vrplib.read_instance(tmp.name, compute_edge_weights=False)
    finally:
        os.unlink(tmp.name)


def _read_yaml(yaml_path: str) -> dict:
    """Parse the companion .yaml metadata file with pyyaml."""
    with open(yaml_path, "r") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
#  Distance matrix
# --------------------------------------------------------------------------- #
def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Symmetric Euclidean distance matrix computed with NumPy.

    Uses the identity ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b for a vectorised,
    high-performance computation (no Python-level loops).

    Parameters
    ----------
    coords : (N, 2) float array of node coordinates.

    Returns
    -------
    (N, N) float array ``D`` with ``D[i, j]`` the Euclidean distance.
    """
    coords = np.asarray(coords, dtype=np.float64)
    sq = np.sum(coords * coords, axis=1)
    # d2[i,j] = sq[i] + sq[j] - 2 * coords[i] . coords[j]
    d2 = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)
    np.maximum(d2, 0.0, out=d2)          # clip tiny negative round-off
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, 0.0)
    return dist


# --------------------------------------------------------------------------- #
#  Public data container
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    """A fully parsed MDCVRP instance, OR-Tools ready."""

    name: str
    num_nodes: int
    coords: np.ndarray            # (N, 2) float
    demands: np.ndarray           # (N,)   int
    capacity: int
    num_depots: int
    depot_indices: list[int]      # 0-indexed depot nodes (dynamic, from metadata)
    customer_indices: list[int]   # 0-indexed customer nodes
    distance_matrix: np.ndarray   # (N, N) float, symmetric
    total_demand: int
    min_vehicles: int             # ceil(total_demand / capacity)
    source_dir: str = ""
    comment: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a plain dictionary (handy for serialisation / inspection)."""
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "coords": self.coords,
            "demands": self.demands,
            "capacity": self.capacity,
            "num_depots": self.num_depots,
            "depot_indices": self.depot_indices,
            "customer_indices": self.customer_indices,
            "distance_matrix": self.distance_matrix,
            "total_demand": self.total_demand,
            "min_vehicles": self.min_vehicles,
            "source_dir": self.source_dir,
            "comment": self.comment,
        }

    def vehicle_start_end(self, num_vehicles: int) -> tuple[list[int], list[int]]:
        """Build per-vehicle start/end depot index lists for OR-Tools.

        Vehicles are distributed round-robin across the available depots, so a
        fleet of ``num_vehicles`` is shared as evenly as possible among the
        ``depot_indices``. Each vehicle starts and ends at the same depot.

        Returns ``(starts, ends)`` — two length-``num_vehicles`` lists of
        0-indexed depot node ids (here ``starts == ends``).
        """
        depots = self.depot_indices
        starts = [depots[i % len(depots)] for i in range(num_vehicles)]
        ends = list(starts)
        return starts, ends


# --------------------------------------------------------------------------- #
#  Loader
# --------------------------------------------------------------------------- #
def load_instance(
    vrp_path: str,
    yaml_path: Optional[str] = None,
    *,
    strict: bool = False,
) -> Instance:
    """Load and structure a single MDCVRP instance.

    Parameters
    ----------
    vrp_path : path to the ``.vrp`` file.
    yaml_path : path to the companion ``.yaml``. If ``None`` it is inferred by
        swapping the extension. If the file is absent, metadata falls back to
        the ``.vrp`` header.
    strict : if ``True``, raise on any .vrp/.yaml metadata mismatch instead of
        silently trusting the YAML.

    Returns
    -------
    Instance
    """
    vrp = _read_vrp(vrp_path)

    if yaml_path is None:
        candidate = os.path.splitext(vrp_path)[0] + ".yaml"
        yaml_path = candidate if os.path.exists(candidate) else None
    ymeta = _read_yaml(yaml_path) if yaml_path else {}

    name = ymeta.get("name") or vrp.get("name") or os.path.basename(vrp_path)

    # --- metadata, YAML authoritative with .vrp fallback ------------------- #
    num_depots = int(ymeta.get("depots", vrp.get("depots", 1)))
    capacity = int(ymeta.get("capacity", vrp.get("capacity")))
    dimension = int(ymeta.get("dimension", vrp.get("dimension", len(vrp["node_coord"]))))

    # cross-check
    if strict:
        if vrp.get("depots") is not None and int(vrp["depots"]) != num_depots:
            raise ValueError(f"{name}: depot mismatch vrp={vrp['depots']} yaml={num_depots}")
        if vrp.get("capacity") is not None and int(vrp["capacity"]) != capacity:
            raise ValueError(f"{name}: capacity mismatch vrp={vrp['capacity']} yaml={capacity}")

    coords = np.asarray(vrp["node_coord"], dtype=np.float64)
    demands = np.asarray(vrp["demand"], dtype=np.int64)
    num_nodes = coords.shape[0]

    if demands.shape[0] != num_nodes:
        raise ValueError(f"{name}: demand/coord length mismatch")

    # --- dynamic depot index list ----------------------------------------- #
    # Prefer vrplib's parsed depot array; fall back to the convention that the
    # first `num_depots` nodes are depots.
    if "depot" in vrp and len(vrp["depot"]) > 0:
        depot_indices = [int(i) for i in np.asarray(vrp["depot"]).ravel().tolist()]
    else:
        depot_indices = list(range(num_depots))
    depot_set = set(depot_indices)
    customer_indices = [i for i in range(num_nodes) if i not in depot_set]

    # sanity: depots should have zero demand
    if np.any(demands[depot_indices] != 0):
        # not fatal, but force depot demand to 0 so capacity accounting is clean
        demands = demands.copy()
        demands[depot_indices] = 0

    distance_matrix = euclidean_distance_matrix(coords)

    total_demand = int(demands.sum())
    min_vehicles = int(np.ceil(total_demand / capacity)) if capacity > 0 else len(customer_indices)

    # feasibility guard: no single customer may exceed vehicle capacity
    max_demand = int(demands.max()) if num_nodes else 0
    if capacity > 0 and max_demand > capacity:
        raise ValueError(
            f"{name}: infeasible — max customer demand {max_demand} > capacity {capacity}"
        )

    return Instance(
        name=name,
        num_nodes=num_nodes,
        coords=coords,
        demands=demands,
        capacity=capacity,
        num_depots=num_depots,
        depot_indices=depot_indices,
        customer_indices=customer_indices,
        distance_matrix=distance_matrix,
        total_demand=total_demand,
        min_vehicles=min_vehicles,
        source_dir=os.path.dirname(vrp_path),
        comment=str(ymeta.get("comment", vrp.get("comment", ""))),
        meta={"dimension": dimension},
    )


# --------------------------------------------------------------------------- #
#  Discovery helpers
# --------------------------------------------------------------------------- #
def _scail_sort_key(path: str):
    """Natural sort key: order by the integer following 'SCAIL' in the name."""
    base = os.path.basename(path)
    digits = ""
    idx = base.upper().find("SCAIL")
    if idx >= 0:
        for ch in base[idx + 5:]:
            if ch.isdigit():
                digits += ch
            else:
                break
    return (int(digits) if digits else 0, base)


def discover_instances(directory: str) -> list[tuple[str, Optional[str]]]:
    """Find all (vrp, yaml) pairs in *directory*, naturally sorted.

    The yaml path is ``None`` when no companion file exists.
    """
    import glob

    pairs: list[tuple[str, Optional[str]]] = []
    for vrp in sorted(glob.glob(os.path.join(directory, "*.vrp")), key=_scail_sort_key):
        yml = os.path.splitext(vrp)[0] + ".yaml"
        pairs.append((vrp, yml if os.path.exists(yml) else None))
    return pairs


# --------------------------------------------------------------------------- #
#  CLI / smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a single MDCVRP instance.")
    ap.add_argument("vrp", help="path to a .vrp file")
    ap.add_argument("--yaml", default=None, help="path to companion .yaml")
    args = ap.parse_args()

    inst = load_instance(args.vrp, args.yaml)
    print(f"name           : {inst.name}")
    print(f"comment        : {inst.comment}")
    print(f"nodes          : {inst.num_nodes}")
    print(f"depots ({inst.num_depots})    : {inst.depot_indices}")
    print(f"customers      : {len(inst.customer_indices)}")
    print(f"capacity       : {inst.capacity}")
    print(f"total demand   : {inst.total_demand}")
    print(f"min vehicles   : {inst.min_vehicles}")
    print(f"dist matrix    : {inst.distance_matrix.shape} "
          f"(symmetric={np.allclose(inst.distance_matrix, inst.distance_matrix.T)})")
    print(f"sample dists   : {inst.distance_matrix[0, 1:5]}")
