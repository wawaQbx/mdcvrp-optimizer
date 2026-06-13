"""
solver.py
=========
End-to-end MDCVRP benchmarking pipeline for the SimMD instance sets.

For every ``.vrp`` instance (paired with its ``.yaml``) found in the configured
directories this script:

  1. Loads it via :mod:`data_loader`.
  2. Solves it with a **Greedy Nearest-Neighbour baseline** (cluster-first,
     route-second capacitated NN).
  3. Solves it with **Google OR-Tools** routing (``PATH_CHEAPEST_ARC`` first
     solution + ``GUIDED_LOCAL_SEARCH`` metaheuristic, default 30 s/instance),
     modelling the multiple depots via per-vehicle start/end indices and a
     capacity dimension.
  4. Saves a side-by-side matplotlib route plot (greedy vs OR-Tools) to
     ``output_plots/`` with each vehicle route in a different colour.
  5. Tracks distance metrics and finally prints a formatted ASCII benchmark
     table (and writes a CSV) comparing the two methods and the optimisation
     rate.

Run ``python solver.py --help`` for options. A quick smoke test:

    python solver.py --limit 2 --time-limit 5 --max-dim 200
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless / no display
from matplotlib import colormaps
from matplotlib.figure import Figure

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tabulate import tabulate

import data_loader
from data_loader import Instance


# --------------------------------------------------------------------------- #
#  Generic route helpers
# --------------------------------------------------------------------------- #
def route_distance(route: list[int], dist: np.ndarray) -> float:
    """Total Euclidean length of a single route (list of node indices)."""
    if len(route) < 2:
        return 0.0
    idx = np.asarray(route)
    return float(dist[idx[:-1], idx[1:]].sum())


def total_distance(routes: list[list[int]], dist: np.ndarray) -> float:
    """Total length over all routes, measured with the exact float matrix."""
    return float(sum(route_distance(r, dist) for r in routes))


def nonempty_routes(routes: list[list[int]]) -> list[list[int]]:
    """Routes that actually visit at least one customer (len > 2)."""
    return [r for r in routes if len(r) > 2]


# --------------------------------------------------------------------------- #
#  Greedy Nearest-Neighbour baseline
# --------------------------------------------------------------------------- #
def greedy_nearest_neighbor(inst: Instance) -> list[list[int]]:
    """Cluster-first, route-second capacitated nearest-neighbour heuristic.

    * Cluster: assign every customer to its nearest depot.
    * Route: within each depot's cluster, repeatedly open a route at the depot
      and greedily append the nearest unvisited customer that still fits the
      vehicle capacity; close the route (return to depot) when nothing fits.

    Returns a list of routes, each of the form ``[depot, c1, ..., ck, depot]``.
    """
    dist = inst.distance_matrix
    demands = inst.demands
    cap = inst.capacity
    depots = inst.depot_indices

    # --- cluster customers to nearest depot ------------------------------- #
    customers = np.asarray(inst.customer_indices)
    routes: list[list[int]] = []
    if customers.size == 0:
        return routes

    # distances from each depot to each customer -> nearest depot per customer
    depot_arr = np.asarray(depots)
    dmat = dist[np.ix_(depot_arr, customers)]          # (D, C)
    nearest_depot = depot_arr[np.argmin(dmat, axis=0)]  # (C,)

    for d in depots:
        cluster = customers[nearest_depot == d]
        unvisited = cluster.copy()
        while unvisited.size:
            route = [int(d)]
            load = 0
            current = d
            while unvisited.size:
                feasible = demands[unvisited] <= (cap - load)
                if not feasible.any():
                    break
                cand = unvisited[feasible]
                nearest = cand[np.argmin(dist[current, cand])]
                route.append(int(nearest))
                load += int(demands[nearest])
                current = int(nearest)
                unvisited = unvisited[unvisited != nearest]
            route.append(int(d))  # back to depot
            routes.append(route)
    return routes


# --------------------------------------------------------------------------- #
#  OR-Tools solver
# --------------------------------------------------------------------------- #
@dataclass
class ORToolsResult:
    routes: list[list[int]]
    distance: Optional[float]
    num_vehicles: int            # fleet size offered
    vehicles_used: int           # routes that served customers
    status: int                  # routing.status() code
    solved: bool


def _fleet_size(inst: Instance, buffer: float) -> int:
    """Choose a fleet size: ceil(min_vehicles * buffer), with sane floors."""
    mv = inst.min_vehicles
    n = max(int(math.ceil(mv * buffer)), mv + 1, inst.num_depots)
    return n


def solve_ortools(
    inst: Instance,
    *,
    time_limit: float = 30.0,
    vehicle_buffer: float = 1.3,
    scale: int = 100,
    first_solution: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
    log_search: bool = False,
) -> ORToolsResult:
    """Solve the MDCVRP with OR-Tools constraint-solver routing."""
    num_nodes = inst.num_nodes
    num_vehicles = _fleet_size(inst, vehicle_buffer)
    starts, ends = inst.vehicle_start_end(num_vehicles)

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # Integer distance matrix (OR-Tools needs integral arc costs). Scaling
    # preserves `log10(scale)` decimal places of the true Euclidean distance.
    int_matrix = np.round(inst.distance_matrix * scale).astype(np.int64).tolist()

    def distance_callback(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int_matrix[f][t]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Capacity dimension via a unary demand callback.
    demands = inst.demands.tolist()

    def demand_callback(from_index):
        return int(demands[manager.IndexToNode(from_index)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                                  # null capacity slack
        [int(inst.capacity)] * num_vehicles,  # per-vehicle capacities
        True,                               # start cumul to zero
        "Capacity",
    )

    # Search parameters.
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution
    )
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic
    )
    params.time_limit.FromMilliseconds(int(time_limit * 1000))
    params.log_search = log_search

    solution = routing.SolveWithParameters(params)
    status = routing.status()

    if solution is None:
        return ORToolsResult([], None, num_vehicles, 0, status, False)

    # Extract routes.
    routes: list[list[int]] = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        # skip vehicles that go straight start->end (no customers)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # final (end) depot
        routes.append(route)

    dist = total_distance(routes, inst.distance_matrix)
    return ORToolsResult(routes, dist, num_vehicles, len(routes), status, True)


# --------------------------------------------------------------------------- #
#  Visualisation
#
#  All figures are produced through the matplotlib *object-oriented* API
#  (``Figure`` + the Agg backend) rather than ``pyplot``. pyplot keeps global
#  state that is not thread-safe; the OO API creates fully independent figures,
#  so rendering can be off-loaded to background worker threads (see the async
#  image-generation mechanism in ``run_batch``).
# --------------------------------------------------------------------------- #
@dataclass
class PlotPayload:
    """Lightweight snapshot of the geometry needed to draw an instance.

    Carrying only the coordinates (and depot/customer indices) — never the
    full N×N distance matrix — keeps queued async plot tasks cheap in memory
    during a large batch run.
    """
    name: str
    coords: np.ndarray
    depot_indices: list[int]
    customer_indices: list[int]
    num_nodes: int
    num_depots: int
    capacity: int

    @classmethod
    def from_instance(cls, inst: Instance) -> "PlotPayload":
        return cls(
            name=inst.name,
            coords=np.asarray(inst.coords, dtype=float),
            depot_indices=list(inst.depot_indices),
            customer_indices=list(inst.customer_indices),
            num_nodes=inst.num_nodes,
            num_depots=inst.num_depots,
            capacity=int(inst.capacity),
        )


def _draw_routes(ax, payload: "PlotPayload", routes: list[list[int]], title: str):
    """Draw customers, depots and the per-vehicle routes onto a single Axes."""
    coords = payload.coords
    cmap = colormaps["tab20"]

    # customers (faint, behind routes)
    cust = coords[payload.customer_indices]
    ax.scatter(cust[:, 0], cust[:, 1], s=6, c="0.8", zorder=1,
               linewidths=0, label="customer")

    # one distinct colour per truck/route
    drawn = nonempty_routes(routes)
    for i, route in enumerate(drawn):
        pts = coords[route]
        color = cmap(i % 20)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=0.9, alpha=0.85, zorder=2)
        # mark the customers served by this route in the route's colour
        ax.scatter(pts[1:-1, 0], pts[1:-1, 1], s=10, color=color, zorder=3, linewidths=0)

    # depots on top
    dep = coords[payload.depot_indices]
    ax.scatter(dep[:, 0], dep[:, 1], s=130, c="black", marker="s",
               zorder=5, label="depot", edgecolors="white", linewidths=0.8)
    for k, d in enumerate(payload.depot_indices):
        ax.annotate(f"D{k}", (coords[d, 0], coords[d, 1]),
                    color="white", fontsize=7, ha="center", va="center", zorder=6)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)


def save_route_map(
    payload: "PlotPayload",
    routes: list[list[int]],
    distance: Optional[float],
    vehicles_used: int,
    solved: bool,
    out_path: str,
):
    """Render the optimised vehicle routes as a single network plot and save it.

    Produced next to the raw data files (e.g. ``..._route.png``) so each map
    pairs directly with its ``.vrp`` / ``.yaml``. Thread-safe (OO API only).
    """
    fig = Figure(figsize=(9, 9))
    ax = fig.subplots()

    if solved:
        title = (f"OR-Tools optimised routes   |   dist={distance:,.0f}   |   "
                 f"trucks={vehicles_used}")
    else:
        title = "OR-Tools — no solution found"
        routes = []

    _draw_routes(ax, payload, routes, title)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"{payload.name}   (nodes={payload.num_nodes}, "
        f"depots={payload.num_depots}, capacity={payload.capacity})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)   # OO Figure is GC'd; no plt.close needed


def plot_comparison(
    payload: "PlotPayload",
    greedy_routes: list[list[int]],
    greedy_dist: float,
    ort_routes: list[list[int]],
    ort_dist: Optional[float],
    ort_vehicles_used: int,
    ort_solved: bool,
    out_path: str,
):
    """Save a side-by-side (greedy | OR-Tools) route network plot (thread-safe)."""
    fig = Figure(figsize=(15, 7.5))
    axes = fig.subplots(1, 2)

    _draw_routes(
        axes[0], payload, greedy_routes,
        f"Greedy NN  |  dist={greedy_dist:,.0f}  |  routes={len(nonempty_routes(greedy_routes))}",
    )

    if ort_solved:
        _draw_routes(
            axes[1], payload, ort_routes,
            f"OR-Tools GLS  |  dist={ort_dist:,.0f}  |  routes={ort_vehicles_used}",
        )
    else:
        _draw_routes(axes[1], payload, [], "OR-Tools GLS  |  no solution found")

    fig.suptitle(
        f"{payload.name}   "
        f"(nodes={payload.num_nodes}, depots={payload.num_depots}, capacity={payload.capacity})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)


# --------------------------------------------------------------------------- #
#  Batch driver
# --------------------------------------------------------------------------- #
@dataclass
class BenchRow:
    instance: str
    dataset: str
    nodes: int
    depots: int
    greedy_dist: float
    greedy_veh: int
    greedy_time: float
    ortools_dist: Optional[float]
    ortools_veh: int
    ortools_time: float
    improvement: Optional[float]   # percentage


def _improvement(greedy: float, ortools: Optional[float]) -> Optional[float]:
    if ortools is None or greedy <= 0:
        return None
    return (greedy - ortools) / greedy * 100.0


def run_batch(args) -> list[BenchRow]:
    rows: list[BenchRow] = []
    # Async image generation: each solved instance hands its (lightweight)
    # plot payload to a background thread pool so route maps render while the
    # solver moves on to the next instance.  Figures use the thread-safe
    # matplotlib OO API.  Each future is tagged with a label for error reports.
    plot_jobs: list[tuple[str, Future]] = []
    executor = None if args.no_plots else ThreadPoolExecutor(max_workers=args.plot_workers)

    try:
        for dataset in args.sets:
            directory = str(os.path.join(args.data_root, dataset))
            if not os.path.isdir(directory):
                print(f"!! skipping missing directory: {directory}")
                continue

            pairs = data_loader.discover_instances(directory)
            if args.limit is not None:
                pairs = pairs[: args.limit]

            print(f"\n=== {dataset}: {len(pairs)} instance(s) ===")
            for n, (vrp, yml) in enumerate(pairs, 1):
                try:
                    inst = data_loader.load_instance(vrp, yml)
                except Exception as exc:  # keep the batch alive
                    print(f"  [{n}/{len(pairs)}] {os.path.basename(vrp)}  LOAD ERROR: {exc}")
                    continue

                if args.max_dim is not None and inst.num_nodes > args.max_dim:
                    print(f"  [{n}/{len(pairs)}] {inst.name:<22} skipped (nodes "
                          f"{inst.num_nodes} > max-dim {args.max_dim})")
                    continue

                # greedy
                t0 = time.perf_counter()
                g_routes = greedy_nearest_neighbor(inst)
                g_dist = total_distance(g_routes, inst.distance_matrix)
                g_time = time.perf_counter() - t0

                # or-tools
                t0 = time.perf_counter()
                ort = solve_ortools(
                    inst,
                    time_limit=args.time_limit,
                    vehicle_buffer=args.vehicle_buffer,
                    scale=args.scale,
                    first_solution=args.first_solution,
                    metaheuristic=args.metaheuristic,
                )
                o_time = time.perf_counter() - t0

                imp = _improvement(g_dist, ort.distance)

                # ----- async image generation ----------------------------- #
                if executor is not None:
                    payload = PlotPayload.from_instance(inst)
                    # 1) optimised route map, saved next to the raw data files
                    route_png = os.path.join(directory, f"{inst.name}_route.png")
                    plot_jobs.append((route_png, executor.submit(
                        save_route_map, payload, ort.routes, ort.distance,
                        ort.vehicles_used, ort.solved, route_png)))
                    # 2) greedy-vs-OR-Tools comparison, saved under output_plots/
                    cmp_png = os.path.join(args.plot_dir, dataset, f"{inst.name}.png")
                    plot_jobs.append((cmp_png, executor.submit(
                        plot_comparison, payload, g_routes, g_dist, ort.routes,
                        ort.distance, ort.vehicles_used, ort.solved, cmp_png)))

                rows.append(BenchRow(
                    instance=inst.name,
                    dataset=dataset,
                    nodes=inst.num_nodes,
                    depots=inst.num_depots,
                    greedy_dist=g_dist,
                    greedy_veh=len(nonempty_routes(g_routes)),
                    greedy_time=g_time,
                    ortools_dist=ort.distance,
                    ortools_veh=ort.vehicles_used,
                    ortools_time=o_time,
                    improvement=imp,
                ))

                o_dist_str = f"{ort.distance:,.0f}" if ort.distance is not None else "FAILED"
                imp_str = f"{imp:+.1f}%" if imp is not None else "  n/a"
                print(f"  [{n}/{len(pairs)}] {inst.name:<22} "
                      f"nodes={inst.num_nodes:<4} greedy={g_dist:>12,.0f} "
                      f"or-tools={o_dist_str:>12} improve={imp_str:>7} "
                      f"({o_time:4.1f}s)")
    finally:
        if executor is not None:
            # wait for all queued route maps to finish rendering
            executor.shutdown(wait=True)

    # surface any plotting errors without aborting the benchmark
    if plot_jobs:
        failures = 0
        for path, fut in plot_jobs:
            exc = fut.exception()
            if exc is not None:
                failures += 1
                print(f"  (plot failed: {os.path.basename(path)}: {exc})")
        print(f"\nImage generation: {len(plot_jobs) - failures}/{len(plot_jobs)} "
              f"figures saved "
              f"({len(rows)} route maps in the data dirs + comparison plots).")

    return rows


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def _signed_int(value) -> str:
    """Explicit-sign integer string: +N for positive, 0 for zero, -N for negative."""
    v = int(value)
    if v > 0:
        return f"+{v}"
    if v == 0:
        return "0"
    return str(v)          # negative numbers already carry the '-' sign


def results_dataframe(rows: list[BenchRow]) -> pd.DataFrame:
    """Build a pandas DataFrame of per-instance results, including the derived
    ``veh_saved`` column (= G.Veh - O.Veh, NA when OR-Tools found no solution)."""
    df = pd.DataFrame([{
        "instance": r.instance,
        "dataset": r.dataset.replace("_instance_set", ""),
        "nodes": r.nodes,
        "depots": r.depots,
        "greedy_dist": r.greedy_dist,
        "greedy_veh": r.greedy_veh,
        "greedy_time": r.greedy_time,
        "ortools_dist": r.ortools_dist,
        "ortools_veh": r.ortools_veh,
        "ortools_time": r.ortools_time,
        "improvement": r.improvement,
    } for r in rows])

    # A non-null OR-Tools distance is itself the proof of a valid solution, so
    # we derive a transient mask here instead of persisting a 'solved' column.
    solved_mask = df["ortools_dist"].notna()

    # New column: vehicles saved = Greedy vehicles - OR-Tools vehicles.
    # Use a nullable integer so failed instances (no solution) stay NA rather
    # than reporting a misleading positive saving.
    df["veh_saved"] = (df["greedy_veh"] - df["ortools_veh"]).astype("Int64")
    df.loc[~solved_mask, "veh_saved"] = pd.NA
    return df


def build_summary(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Compute the aggregate benchmark metrics as ordered (Metric, Value) rows."""
    n_total = len(df)
    solved = df[df["ortools_dist"].notna()]
    n_solved = len(solved)

    sum_g = float(solved["greedy_dist"].sum())
    sum_o = float(solved["ortools_dist"].sum())
    overall_rate = (sum_g - sum_o) / sum_g * 100.0 if sum_g > 0 else 0.0
    mean_rate = float(solved["improvement"].mean()) if n_solved else 0.0
    wins = int((solved["improvement"] > 0).sum())

    # --- vehicle-savings metrics (summed over solved instances) ----------- #
    total_veh_saved = int(solved["veh_saved"].sum()) if n_solved else 0
    avg_veh_saved = total_veh_saved / n_total if n_total else 0.0

    return [
        ("Instances benchmarked", str(n_total)),
        ("OR-Tools solved", f"{n_solved}/{n_total}"),
        ("Total Greedy distance", f"{sum_g:,.0f}"),
        ("Total OR-Tools distance", f"{sum_o:,.0f}"),
        ("Aggregate distance saved", f"{sum_g - sum_o:,.0f}"),
        ("OVERALL OPTIMISATION RATE", f"{overall_rate:+.2f}%"),
        ("Mean per-instance improvement", f"{mean_rate:+.2f}%"),
        ("OR-Tools beat baseline on", f"{wins}/{n_solved} instances"),
        ("Total vehicles saved", _signed_int(total_veh_saved)),
        ("Average vehicles saved", f"{avg_veh_saved:+.2f}"),
    ]


def print_summary(rows: list[BenchRow]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Render the main results table and the summary panel.

    Returns ``(results_df, summary_rows)`` so callers can export them.
    """
    if not rows:
        print("\nNo instances were solved.")
        return pd.DataFrame(), []

    df = results_dataframe(rows)

    # ---- main table (rendered with tabulate for the ASCII grid) ---------- #
    table = []
    for r in df.itertuples(index=False):
        is_solved = pd.notna(r.ortools_dist)
        o_dist = f"{r.ortools_dist:,.0f}" if is_solved else "FAILED"
        imp = f"{r.improvement:+.1f}%" if pd.notna(r.improvement) else "n/a"
        veh_saved = _signed_int(r.veh_saved) if is_solved else "n/a"
        table.append([
            r.instance, r.dataset, r.nodes, r.depots,
            f"{r.greedy_dist:,.0f}", r.greedy_veh,
            o_dist, r.ortools_veh, veh_saved,
            f"{r.ortools_time:.1f}s", imp,
        ])

    headers = ["Instance", "Set", "Nodes", "Dep",
               "Greedy Dist", "G.Veh", "OR-Tools Dist", "O.Veh", "Veh. Saved",
               "OR Time", "Improvement"]

    print("\n")
    print("=" * 108)
    print(" MDCVRP BENCHMARK — Greedy Nearest-Neighbour  vs  Google OR-Tools (GLS)")
    print("=" * 108)
    print(tabulate(table, headers=headers, tablefmt="fancy_grid",
                   colalign=("left", "left", "right", "right", "right",
                             "right", "right", "right", "right", "right", "right")))

    # ---- summary panel --------------------------------------------------- #
    summary_rows = build_summary(df)
    print("\n" + tabulate(summary_rows, headers=["Metric", "Value"],
                          tablefmt="rounded_grid"))
    return df, summary_rows


def _safe_to_csv(df: pd.DataFrame, path: str, label: str, **kwargs):
    """Write a DataFrame to CSV, reporting (not raising) when the file is locked."""
    try:
        df.to_csv(path, index=False, **kwargs)
        print(f"{label} written to {path}")
    except PermissionError:
        print(f"!! Could not write {label} to {path} — the file is open "
              f"(close it in Excel/editor and re-run). Skipping.")
    except OSError as exc:
        print(f"!! Could not write {label} to {path}: {exc}. Skipping.")


def write_csv(df: pd.DataFrame, path: str):
    """Export the per-instance results DataFrame to CSV."""
    print()  # blank line before the file-write messages
    _safe_to_csv(df, path, "Results", float_format="%.3f")


def write_summary_csv(summary_rows: list[tuple[str, str]], path: str):
    """Export the standalone summary panel as a clean Metric/Value CSV."""
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    _safe_to_csv(summary_df, path, "Summary")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="MDCVRP benchmark: Greedy baseline vs Google OR-Tools.")
    ap.add_argument("--data-root", default=os.path.join(here, "data"),
                    help="root folder containing the instance set subfolders")
    ap.add_argument("--sets", nargs="+",
                    default=["incom2024_instance_set", "mim2025_instance_set"],
                    help="instance-set subfolders to benchmark")
    ap.add_argument("--time-limit", type=float, default=30.0,
                    help="OR-Tools search time limit per instance (seconds)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max number of instances per set (for quick runs)")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="skip instances with more than this many nodes")
    ap.add_argument("--vehicle-buffer", type=float, default=1.3,
                    help="fleet size = ceil(min_vehicles * buffer)")
    ap.add_argument("--scale", type=int, default=100,
                    help="integer scaling factor for OR-Tools arc costs")
    ap.add_argument("--first-solution", default="PATH_CHEAPEST_ARC",
                    help="OR-Tools first-solution strategy")
    ap.add_argument("--metaheuristic", default="GUIDED_LOCAL_SEARCH",
                    help="OR-Tools local-search metaheuristic")
    ap.add_argument("--plot-dir", default=os.path.join(here, "output_plots"),
                    help="directory for the greedy-vs-OR-Tools comparison plots")
    ap.add_argument("--plot-workers", type=int, default=3,
                    help="background threads for async route-map image generation")
    ap.add_argument("--no-plots", action="store_true", help="disable plotting")
    ap.add_argument("--results-csv", default=os.path.join(here, "benchmark_results.csv"),
                    help="path for the per-instance CSV results file")
    ap.add_argument("--summary-csv", default=os.path.join(here, "benchmark_summary.csv"),
                    help="path for the standalone summary CSV file")
    return ap


def main():
    args = build_argparser().parse_args()
    start = time.perf_counter()
    rows = run_batch(args)
    df, summary_rows = print_summary(rows)
    if rows:
        write_csv(df, args.results_csv)
        write_summary_csv(summary_rows, args.summary_csv)
    print(f"\nTotal wall-clock: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
