import argparse
import csv
import gzip
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Set, Tuple

import networkx as nx

from query_kcore_greedy_anchored import (
    k_core_nodes,
    query_centric_kcore_greedy,
    query_query_only_baseline,
    query_kcore_only_baseline,
)

Node = Hashable


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated integer list such as '5,10,20'."""
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _open_text(path: Path):
    """Open either a plain text edge list or a gzip-compressed edge list."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _convert_node(token: str, node_type: str):
    if node_type == "int":
        return int(token)
    if node_type == "str":
        return token

    # auto: use integer IDs when every token is integer-like; otherwise string.
    try:
        return int(token)
    except ValueError:
        return token


def load_edge_list(
    path: str,
    delimiter: Optional[str] = None,
    node_type: str = "auto",
    comment_prefix: str = "#",
) -> nx.Graph:
    """
    Load an undirected, unweighted graph from a 2-column edge list.

    Accepted examples:
        0 1
        0 2
        1 5

    The input may be plain text (.txt, .edges, ...) or gzip compressed (.gz).
    Blank lines and lines starting with comment_prefix are ignored.
    Extra columns are ignored; only the first two fields are used.

    delimiter=None means arbitrary whitespace, which matches the SNAP
    facebook_combined.txt(.gz) format.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    G = nx.Graph()
    skipped_lines = 0

    with _open_text(file_path) as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or (comment_prefix and line.startswith(comment_prefix)):
                continue

            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) < 2:
                skipped_lines += 1
                continue

            try:
                u = _convert_node(parts[0], node_type)
                v = _convert_node(parts[1], node_type)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse node ID at line {line_no}: {line!r}"
                ) from exc

            if u == v:
                # Self-loops do not affect the simple k-core experiment.
                continue

            G.add_edge(u, v)

    if G.number_of_nodes() == 0:
        raise ValueError("No valid edges were found in the input file.")

    if skipped_lines:
        print(f"[Info] skipped malformed lines: {skipped_lines}")

    # Duplicate edges are automatically merged by nx.Graph().
    return G


def choose_queries_random(
    G: nx.Graph,
    initial_core: Set[Node],
    query_size: int,
    rng: random.Random,
) -> Tuple[Set[Node], Dict]:
    """Uniform baseline: choose queries randomly from V - C_k(G)."""
    pool = list(set(G.nodes()) - initial_core)
    if len(pool) < query_size:
        raise ValueError(
            f"Not enough vertices outside C_k(G): requested {query_size}, "
            f"available {len(pool)}."
        )

    Q = set(rng.sample(pool, query_size))
    return Q, {
        "selection": "random",
        "selection_rule": "uniform-outside-core",
        "seed_node": "",
        "candidate_pool_size": len(pool),
        "local_pool_size": len(pool),
        "avg_deficit": "",
        "avg_shell_neighbors": "",
        "avg_local_score": "",
    }



def choose_queries_boundary_biased(
    G: nx.Graph,
    initial_core: Set[Node],
    lower_core: Set[Node],
    k: int,
    query_size: int,
    rng: random.Random,
    candidate_fraction: float = 0.30,
    min_shell_neighbors: int = 1,
) -> Tuple[Set[Node], Dict]:
    """
    Boundary-biased query sampling without localization.

    This sampler DOES NOT evaluate k-1 components, G_Q, activation edges, or
    the proposed algorithm's ScoreQ. It uses only local structural features in
    the ORIGINAL graph:

        d_core(q)  = |N(q) intersect C_k(G)|
        deficit(q) = max(1, k - d_core(q))
        d_shell(q) = |N(q) intersect (C_{k-1}(G) - C_k(G))|
        local_score(q) = d_shell(q) / deficit(q)

    Sampling procedure:
      1. Prefer nodes with at least min_shell_neighbors neighbors in the k-1 shell.
      2. Keep the top candidate_fraction by local_score. Ties are randomized.
      3. Uniformly sample query_size vertices from that candidate pool.

    There is no random seed node or hop-radius restriction. Therefore selected
    queries may come from different graph regions while still being biased toward
    promising k-core boundaries.
    """
    if not 0 < candidate_fraction <= 1:
        raise ValueError("candidate_fraction must be in (0, 1].")
    if min_shell_neighbors < 0:
        raise ValueError("min_shell_neighbors must be >= 0.")

    outside = set(G.nodes()) - set(initial_core)
    if len(outside) < query_size:
        raise ValueError(
            f"Not enough vertices outside C_k(G): requested {query_size}, "
            f"available {len(outside)}."
        )

    shell = set(lower_core) - set(initial_core)

    # Prefer vertices outside C_{k-1}.
    # This avoids selecting the k-1-shell vertices themselves as queries.
    exterior = set(G.nodes()) - set(lower_core)
    query_universe = exterior if len(exterior) >= query_size else outside

    features: Dict[Node, Dict[str, float]] = {}
    strict_pool: List[Node] = []

    for q in query_universe:
        d_core = 0
        d_shell = 0
        for u in G.neighbors(q):
            if u in initial_core:
                d_core += 1
            elif u in shell:
                d_shell += 1

        deficit = max(1, k - d_core)
        local_score = d_shell / deficit

        features[q] = {
            "core_neighbors": d_core,
            "shell_neighbors": d_shell,
            "deficit": deficit,
            "local_score": local_score,
        }

        if d_shell >= min_shell_neighbors:
            strict_pool.append(q)

    if len(strict_pool) >= query_size:
        eligible = strict_pool
        selection_rule = "shell-neighbor-threshold"
    else:
        # Keep every node satisfying the shell-neighbor threshold, then fill
        # only the shortage uniformly at random from the whole query universe.
        eligible = list(strict_pool)
        remaining = [q for q in query_universe if q not in set(strict_pool)]
        rng.shuffle(remaining)
        needed = query_size - len(eligible)
        eligible.extend(remaining[:needed])
        selection_rule = "shell-neighbor-threshold+random-fill"

    # Shuffle first so ties in local_score do not systematically favor node IDs.
    eligible = list(eligible)
    rng.shuffle(eligible)
    eligible.sort(key=lambda q: features[q]["local_score"], reverse=True)

    top_n = max(query_size, math.ceil(candidate_fraction * len(eligible)))
    top_n = min(top_n, len(eligible))
    candidate_pool = eligible[:top_n]

    if len(candidate_pool) < query_size:
        raise ValueError(
            f"Candidate pool too small: requested {query_size}, "
            f"available {len(candidate_pool)}."
        )

    # Sample globally from the whole candidate pool.
    selected = set(rng.sample(candidate_pool, query_size))

    def avg_feature(name: str) -> float:
        return sum(features[q][name] for q in selected) / len(selected)

    return selected, {
        "selection": "boundary-biased",
        "selection_rule": selection_rule,
        "seed_node": "",
        "candidate_pool_size": len(candidate_pool),
        # For a non-localized sampler, local_pool is simply the candidate pool.
        "local_pool_size": len(candidate_pool),
        "avg_deficit": avg_feature("deficit"),
        "avg_shell_neighbors": avg_feature("shell_neighbors"),
        "avg_local_score": avg_feature("local_score"),
    }

def run_one(
    G: nx.Graph,
    initial_core: Set[Node],
    Q: Set[Node],
    query_info: Dict,
    dataset_name: str,
    algorithm: str,
    k: int,
    query_size: int,
    repeat: int,
    seed: int,
    max_iterations: int,
    verbose: bool,
) -> Dict:
    """Run one algorithm on an already chosen query set Q."""
    row = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "vertices": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "k": k,
        "query_size": query_size,
        "repeat": repeat,
        "seed": seed,
        "initial_core_size": len(initial_core),
        "outside_core_size": G.number_of_nodes() - len(initial_core),
        "queries": " ".join(map(str, sorted(Q, key=str))),
        "query_selection": query_info.get("selection", ""),
        "selection_rule": query_info.get("selection_rule", ""),
        "query_seed_node": query_info.get("seed_node", ""),
        "candidate_pool_size": query_info.get("candidate_pool_size", ""),
        "local_pool_size": query_info.get("local_pool_size", ""),
        "avg_query_deficit": query_info.get("avg_deficit", ""),
        "avg_query_shell_neighbors": query_info.get("avg_shell_neighbors", ""),
        "avg_query_local_score": query_info.get("avg_local_score", ""),
        "success": False,
        "added_edges": "",
        "iterations": "",
        "component_steps": "",
        "query_query_steps": "",
        "query_core_steps": "",
        "fallback_steps": "",
        "final_core_size": "",
        "runtime_sec": "",
        "error": "",
    }

    if not initial_core:
        row["error"] = "C_k(G) is empty."
        return row

    start_time = time.perf_counter()

    try:
        if algorithm == "proposed":
            _, added_edges, history, final_core = query_centric_kcore_greedy(
                G,
                Q=Q,
                k=k,
                max_iterations=max_iterations,
                verbose=verbose,
                initial_core=initial_core,
                return_final_core=True,
            )
        elif algorithm == "query-query-only":
            _, added_edges, history, final_core = query_query_only_baseline(
                G,
                Q=Q,
                k=k,
                max_iterations=max_iterations,
                verbose=verbose,
                initial_core=initial_core,
                return_final_core=True,
            )
        elif algorithm == "query-kcore-only":
            _, added_edges, history, final_core = query_kcore_only_baseline(
                G,
                Q=Q,
                k=k,
                max_iterations=max_iterations,
                verbose=verbose,
                initial_core=initial_core,
                return_final_core=True,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        elapsed = time.perf_counter() - start_time
        success = Q.issubset(final_core)

        component_steps = sum(1 for h in history if h.get("type") == "component")
        query_query_steps = sum(
            1 for h in history
            if h.get("type") in {"query-query fallback", "query-query only"}
        )
        query_core_steps = sum(
            1 for h in history
            if h.get("type") in {"query-core fallback", "query-k-core only"}
        )

        row.update({
            "success": success,
            "added_edges": len(added_edges),
            "iterations": len(history),
            "component_steps": component_steps,
            "query_query_steps": query_query_steps,
            "query_core_steps": query_core_steps,
            "fallback_steps": query_query_steps + query_core_steps,
            "final_core_size": len(final_core),
            "runtime_sec": elapsed,
        })

        if not success:
            row["error"] = "Not all queries entered the ordinary C_k(G_A)."

    except Exception as exc:
        row["runtime_sec"] = time.perf_counter() - start_time
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str, int, int], List[Dict]] = {}
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["k"], row["query_size"])
        groups.setdefault(key, []).append(row)

    summary: List[Dict] = []

    for (dataset, algorithm, k, query_size), group in sorted(groups.items()):
        successes = [r for r in group if r["success"]]

        def numbers(key: str) -> List[float]:
            return [float(r[key]) for r in successes if r[key] != ""]

        edge_values = numbers("added_edges")
        runtime_values = numbers("runtime_sec")
        final_core_values = numbers("final_core_size")

        summary.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "k": k,
            "query_size": query_size,
            "avg_added_edges": statistics.mean(edge_values) if edge_values else "",
            "avg_runtime_sec": statistics.mean(runtime_values) if runtime_values else "",
            "avg_final_core_size": statistics.mean(final_core_values) if final_core_values else "",
        })

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the anchored query-centric k-core greedy experiment on any "
            "undirected 2-column edge-list dataset."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a 2-column edge-list file (.txt or .gz).",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name written to result CSV. Default: input filename.",
    )
    parser.add_argument(
        "--k-values",
        default="10,20",
        help="Comma-separated k values, e.g. 5,10,15,20.",
    )
    parser.add_argument(
        "--query-sizes",
        default="10,20,30",
        help="Comma-separated query-set sizes, e.g. 1,5,10,20.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of random query sets for each (k, |Q|).",
    )
    parser.add_argument(
        "--algorithm",
        choices=["all", "proposed", "query-query-only", "query-kcore-only"],
        default="all",
        help=(
            "Default all: run proposed, query-query-only, and query-kcore-only "
        ),
    )
    parser.add_argument(
        "--query-selection",
        choices=["boundary-biased", "random"],
        default="boundary-biased",
        help=(
            "Query sampling method. boundary-biased samples randomly from the global "
            "boundary candidate pool; random is uniform outside C_k(G)."
        ),
    )
    parser.add_argument(
        "--boundary-candidate-fraction",
        type=float,
        default=0.5,
        help="Top local-score fraction retained as randomized candidates (0.50 = top 50 percent). Default: 0.50.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum greedy iterations per experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_results",
        help=(
            "Base directory for experiment results. A unique subdirectory is "
            "created for every run so previous results are never overwritten."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed greedy iterations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load the dataset and print graph statistics.",
    )
    args = parser.parse_args()
    DELIMITER = None
    NODE_TYPE = "auto"
    MIN_SHELL_NEIGHBORS = 1

    #
    if args.seed is None:
        base_seed = random.SystemRandom().randrange(0, 2**63)
    else:
        base_seed = args.seed

    data_path = Path(args.data)
    dataset_name = args.dataset_name or data_path.name
    #

    print(f"Loading dataset: {data_path}")
    G = load_edge_list(
        str(data_path),
        delimiter=DELIMITER,
        node_type=NODE_TYPE,
    )

    print(f"Dataset = {dataset_name}")
    print(f"|V| = {G.number_of_nodes():,}")
    print(f"|E| = {G.number_of_edges():,}")
    print(f"Connected components = {nx.number_connected_components(G):,}")

    if args.dry_run:
        print("Dry run completed: dataset format is valid.")
        return

    k_values = parse_int_list(args.k_values)
    query_sizes = parse_int_list(args.query_sizes)

    # Create a unique result directory for this program run.
    # Example:
    #   experiment_results/facebook_combined.txt_k-10-20_q-10-30_r-5_frac-0.3_20260908_231500/
    # This prevents a later run from overwriting summary.csv
    # produced by an earlier run.
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    def safe_path_part(value) -> str:
        text = str(value)
        return "".join(
            ch if ch.isalnum() or ch in "._-" else "-"
            for ch in text
        )

    data_label = safe_path_part(data_path.name)
    k_label = safe_path_part(args.k_values.replace(",", "-"))
    q_label = safe_path_part(args.query_sizes.replace(",", "-"))
    fraction_label = safe_path_part(f"{args.boundary_candidate_fraction:g}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_dir_name = (
        f"{data_label}_k-{k_label}_q-{q_label}_"
        f"r-{args.repeats}_frac-{fraction_label}_{timestamp}"
    )
    output_dir = base_output_dir / run_dir_name

    # Two runs with identical parameters can start within the same second.
    # In that rare case, append _2, _3, ... instead of reusing the directory.
    if output_dir.exists():
        suffix = 2
        while True:
            candidate_dir = base_output_dir / f"{run_dir_name}_{suffix}"
            if not candidate_dir.exists():
                output_dir = candidate_dir
                break
            suffix += 1

    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Result directory = {output_dir}")

    rows: List[Dict] = []
    if args.algorithm == "all":
        algorithms = ["proposed", "query-query-only", "query-kcore-only"]
    else:
        algorithms = [args.algorithm]

    for k in k_values:
        initial_core = k_core_nodes(G, k)
        lower_core = k_core_nodes(G, k - 1) if k > 1 else set(G.nodes())
        shell = lower_core - initial_core
        outside = G.number_of_nodes() - len(initial_core)
        print(
            f"\nk={k}: |C_k(G)|={len(initial_core):,}, "
            f"|C_(k-1)-C_k|={len(shell):,}, outside={outside:,}"
        )
        if not initial_core:
            print("  skipped: C_k(G) is empty, so query-to-k-core comparison is undefined.")
            continue

        for query_size in query_sizes:
            for repeat in range(1, args.repeats + 1):
                run_seed = base_seed + k * 1_000_000 + query_size * 10_000 + repeat
                rng = random.Random(run_seed)

                try:
                    if args.query_selection == "random":
                        Q, query_info = choose_queries_random(
                            G, initial_core, query_size, rng
                        )
                    else:
                        Q, query_info = choose_queries_boundary_biased(
                            G=G,
                            initial_core=initial_core,
                            lower_core=lower_core,
                            k=k,
                            query_size=query_size,
                            rng=rng,
                            candidate_fraction=args.boundary_candidate_fraction,
                            min_shell_neighbors=MIN_SHELL_NEIGHBORS,
                        )
                except ValueError as exc:
                    print(
                        f"  failed to choose queries: k={k}, |Q|={query_size}, "
                        f"repeat={repeat}: {exc}"
                    )
                    continue

                print(
                    f"\nComparison: k={k}, |Q|={query_size}, "
                    f"repeat={repeat}/{args.repeats}, seed={run_seed}"
                )
                if query_info.get("selection") == "boundary-biased":
                    print(
                        f"  query-selection=boundary-biased, "
                        f"rule={query_info['selection_rule']}, "
                        f"candidate-pool={query_info['candidate_pool_size']}"
                    )
                    print(
                        f"  query-features: avg-deficit={query_info['avg_deficit']:.3f}, "
                        f"avg-shell-neighbors={query_info['avg_shell_neighbors']:.3f}, "
                        f"avg-local-score={query_info['avg_local_score']:.3f}"
                    )

                comparison_rows: List[Dict] = []

                for algorithm in algorithms:
                    row = run_one(
                        G=G,
                        initial_core=initial_core,
                        Q=Q,
                        query_info=query_info,
                        dataset_name=dataset_name,
                        algorithm=algorithm,
                        k=k,
                        query_size=query_size,
                        repeat=repeat,
                        seed=run_seed,
                        max_iterations=args.max_iterations,
                        verbose=args.verbose,
                    )
                    rows.append(row)
                    comparison_rows.append(row)

                    if row["success"]:
                        print(
                            f"  {algorithm:<18} "
                            f"edges={int(row['added_edges']):>5}  "
                            f"time={float(row['runtime_sec']):>9.6f}s"
                        )
                    else:
                        print(
                            f"  {algorithm:<18} FAILED  "
                            f"time={float(row['runtime_sec']):.6f}s  "
                            f"error={row['error']}"
                        )

                    # Save the summary incrementally so completed results survive interruption.
                    write_csv(output_dir / "summary.csv", summarize(rows))

                if comparison_rows:
                    print("  " + "-" * 49)
                    print("  method               added_edges   runtime_sec")
                    for row in comparison_rows:
                        edge_text = str(row["added_edges"]) if row["success"] else "FAIL"
                        print(
                            f"  {row['algorithm']:<20} "
                            f"{edge_text:>11}   "
                            f"{float(row['runtime_sec']):>11.6f}"
                        )

    print("\nFinished.")
    print(f"Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
