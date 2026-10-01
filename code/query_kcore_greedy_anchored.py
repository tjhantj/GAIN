from collections import deque
from dataclasses import dataclass
from typing import Hashable, Iterable, Set, Dict, List, Tuple, Optional, Any
import networkx as nx

Node = Hashable
Edge = Tuple[Node, Node]


@dataclass
class Candidate:
    component_index: int
    layer_start: int
    nodes: Set[Node]                 # S_i^j = L_j union ... union L_h
    candidate_layer: Set[Node]        # L_j: newly exposed layer of candidate S_i^j
    support_need: int                # N_S(S_i^j): total required support units
    activation_edges: List[Edge]
    query_gain: int
    score_q: float                   # Score_Q = G_Q / N_S(S_i^j)


def k_core_nodes(G: nx.Graph, k: int) -> Set[Node]:
    """Return the ordinary vertex set C_k(G)."""
    if k <= 0:
        return set(G.nodes())

    # The experiment graph is already simple/self-loop-free. Avoid copying the
    # entire graph in the common case; copy only if self-loops actually exist.
    if nx.number_of_selfloops(G) == 0:
        return set(nx.k_core(G, k=k).nodes())

    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    return set(nx.k_core(H, k=k).nodes())


def anchored_core_nodes(
    G: nx.Graph,
    k: int,
    anchors: Iterable[Node],
) -> Set[Node]:
    """
    Return the anchored k-core used during search.

    Query vertices are never peeled. A non-query vertex is peeled whenever its
    degree in the currently remaining graph becomes smaller than k.
    """
    if k <= 0:
        return set(G.nodes())

    anchors = set(anchors)
    missing = anchors - set(G.nodes())
    if missing:
        raise ValueError(f"Anchor vertices not in G: {missing}")

    remaining = set(G.nodes())
    degree = {v: G.degree(v) for v in G.nodes()}

    queue = deque(
        v for v in remaining
        if v not in anchors and degree[v] < k
    )

    while queue:
        v = queue.popleft()
        if v not in remaining or v in anchors:
            continue
        if degree[v] >= k:
            continue

        remaining.remove(v)

        for u in G.neighbors(v):
            if u not in remaining:
                continue
            degree[u] -= 1
            if u not in anchors and degree[u] < k:
                queue.append(u)

    return remaining


def onion_layers_for_component(
    G: nx.Graph,
    S: Set[Node],
    anchored_core: Set[Node],
    k: int,
) -> List[Set[Node]]:
    """
    Construct onion layers L_1, ..., L_h for one k-1 component.

    This is a queue/batch-based peeling implementation. Vertices already in
    ``anchored_core`` are stable support and are never peeled here.  L_1 is the
    set of vertices in S whose support degree in anchored_core union S is below
    k. After removing an entire layer at once, only neighbors of that layer can
    lose support and become members of the next layer.

    Unlike the previous implementation, this routine does not rescan every
    remaining vertex and all of its neighbors for every onion layer. Each
    candidate vertex is removed at most once and its adjacency list is scanned
    at most once during peeling, so the running time is

        O(|S| + sum_{v in S} deg_G(v)),

    which is linear in the component plus its incident-edge volume.
    """
    remaining = set(S)
    if not remaining:
        return []

    anchored_core = set(anchored_core)

    # Current support degree with respect to anchored_core union S.
    # Anchored-core vertices are permanent support and are never decremented.
    support_degree: Dict[Node, int] = {}
    for v in remaining:
        support_degree[v] = sum(
            1
            for u in G.neighbors(v)
            if u in remaining or u in anchored_core
        )

    # Batch 1: vertices already below k before any vertex of S is peeled.
    current_layer: Set[Node] = {
        v for v in remaining if support_degree[v] < k
    }
    layers: List[Set[Node]] = []

    while current_layer:
        layer = set(current_layer)
        layers.append(layer)

        # Remove the whole batch simultaneously. This is important: a vertex
        # that falls below k because of L_j belongs to L_{j+1}, not L_j.
        remaining.difference_update(layer)

        next_layer: Set[Node] = set()

        # Only surviving neighbors can lose support. No global rescan is needed.
        for v in layer:
            for u in G.neighbors(v):
                if u not in remaining:
                    continue

                support_degree[u] -= 1
                if support_degree[u] < k:
                    next_layer.add(u)

        current_layer = next_layer

    # Normally a true (k-1)-shell component is completely peeled. Preserve the
    # previous routine's behavior for a stable remainder, so callers see the
    # same layer convention even in unexpected/local edge cases.
    if remaining:
        layers.append(set(remaining))

    return layers

def initial_query_deficits(
    G: nx.Graph,
    Q: Set[Node],
    k: int,
    anchored_core: Set[Node],
) -> Dict[Node, int]:
    """
    R(q) = max(0, k - |N(q,G_A) intersect anchored-C_k(G_A,Q)|).

    Queries are anchors during search, but an anchored query is not considered
    finished until it actually has k surviving neighbors.
    """
    R: Dict[Node, int] = {}
    anchored_core = set(anchored_core)

    for q in Q:
        support = sum(1 for u in G.neighbors(q) if u in anchored_core)
        R[q] = max(0, k - support)

    return R


def query_contribution_count(
    G: nx.Graph,
    nodes: Iterable[Node],
    Q_in: Set[Node],
) -> Dict[Node, int]:
    """For each candidate vertex, count adjacent unfinished queries."""
    Q_in = set(Q_in)
    result: Dict[Node, int] = {}

    for v in nodes:
        result[v] = sum(1 for u in G.neighbors(v) if u in Q_in)

    return result


def _planned_edge_exists(
    G: nx.Graph,
    planned_edges: Set[frozenset],
    u: Node,
    v: Node,
) -> bool:
    """Check an edge in G plus the small local candidate-edge set."""
    if u == v:
        return True
    return G.has_edge(u, v) or frozenset((u, v)) in planned_edges


def requisite_degrees_for_candidate(
    G: nx.Graph,
    candidate_nodes: Set[Node],
    anchored_core: Set[Node],
    k: int,
) -> Dict[Node, int]:
    """
    Compute the paper's required support for every node in S_i^j.

    With U = S_i^j union Q union C_k(G_A), represented here by the selected
    candidate nodes together with ``anchored_core``, each node v requires

        r(v) = max(0, k - |N(v) intersect U|).

    Summing these values gives N_S(S_i^j), the denominator of Score_Q.
    """
    candidate_nodes = set(candidate_nodes)
    support_region = set(anchored_core) | candidate_nodes
    requisite: Dict[Node, int] = {}

    for v in candidate_nodes:
        inside = sum(1 for u in G.neighbors(v) if u in support_region)
        requisite[v] = max(0, k - inside)

    return requisite


def build_candidate_support_edges(
    G: nx.Graph,
    candidate_layer: Set[Node],
    requisite: Dict[Node, int],
    anchored_core: Set[Node],
    Q: Set[Node],
    stable_non_query_order: Optional[List[Node]] = None,
    query_anchor_order: Optional[List[Node]] = None,
    query_gain_capacity: Optional[Dict[Node, int]] = None,
) -> Optional[List[Edge]]:
    """
    Greedily build an activation-edge set for one activation candidate.

    This is a greedy heuristic for constructing the activation edges. The
    caller and scoring logic are unchanged: the returned edges must supply every
    deficient candidate vertex v with at least ``requisite[v]`` new support units.

    Greedy priority:

      1) connect two still-deficient candidate vertices whenever possible, so one
         added edge supplies one support unit to both endpoints;
      2) for a remaining single-endpoint support need, prefer an unfinished query
         edge that can still reduce query deficit;
      3) otherwise use a stable non-query anchored-core vertex;
      4) finally use any query anchor as stable support;
      5) if anchored-core support is unavailable, use a missing candidate edge to
         an already-satisfied candidate vertex as a feasibility fallback.

    The method is intentionally greedy and does not guarantee a minimum-cardinality
    activation set.  It avoids NumPy/SciPy and any exact optimization.
    """
    if not candidate_layer:
        return []

    layer = set(candidate_layer)
    anchored_core = set(anchored_core)
    Q = set(Q)

    need = {v: max(0, int(requisite.get(v, 0))) for v in layer}
    active = [v for v in layer if need[v] > 0]
    if not active:
        return []

    if query_gain_capacity is None:
        query_gain_capacity = {}
    gain_cap = {
        q: max(0, int(query_gain_capacity.get(q, 0)))
        for q in Q
    }

    if stable_non_query_order is None:
        stable_non_query_order = sorted(
            anchored_core - Q,
            key=lambda x: G.degree(x),
            reverse=True,
        )
    if query_anchor_order is None:
        query_anchor_order = sorted(
            Q,
            key=lambda x: (gain_cap.get(x, 0), G.degree(x)),
            reverse=True,
        )

    added: List[Edge] = []
    planned_edges: Set[frozenset] = set()
    remaining = dict(need)

    def add_edge(u: Node, v: Node) -> None:
        edge = (u, v)
        added.append(edge)
        planned_edges.add(frozenset(edge))

    # Phase 1: greedily exploit 2-for-1 candidate--candidate edges.
    # Process larger deficits first. For each u, use the currently most-deficient
    # compatible partner. No all-pairs edge list or optimization model is built.
    while True:
        deficient = [v for v in active if remaining[v] > 0]
        if len(deficient) < 2:
            break

        deficient.sort(key=lambda v: (remaining[v], G.degree(v)), reverse=True)
        made_pair = False

        for i, u in enumerate(deficient):
            if remaining[u] <= 0:
                continue

            best_v = None
            best_key = None
            for v in deficient[i + 1:]:
                if remaining[v] <= 0:
                    continue
                if _planned_edge_exists(G, planned_edges, u, v):
                    continue

                key = (remaining[v], G.degree(v))
                if best_v is None or key > best_key:
                    best_v = v
                    best_key = key

            if best_v is None:
                # A compatible partner may occur before u in the sorted order.
                for v in deficient[:i]:
                    if remaining[v] <= 0:
                        continue
                    if _planned_edge_exists(G, planned_edges, u, v):
                        continue

                    key = (remaining[v], G.degree(v))
                    if best_v is None or key > best_key:
                        best_v = v
                        best_key = key

            if best_v is not None:
                add_edge(u, best_v)
                remaining[u] -= 1
                remaining[best_v] -= 1
                made_pair = True

        if not made_pair:
            break

    # Phase 2: satisfy any residual single-endpoint support greedily.
    for v in sorted(active, key=lambda x: (remaining[x], G.degree(x)), reverse=True):
        while remaining[v] > 0:
            chosen_support = None

            # Prefer a query edge only while it still gives useful query gain.
            for q in query_anchor_order:
                if gain_cap.get(q, 0) <= 0:
                    continue
                if not _planned_edge_exists(G, planned_edges, v, q):
                    chosen_support = q
                    break

            if chosen_support is not None:
                add_edge(v, chosen_support)
                gain_cap[chosen_support] -= 1
                remaining[v] -= 1
                continue

            # Otherwise use stable non-query core support.
            for u in stable_non_query_order:
                if not _planned_edge_exists(G, planned_edges, v, u):
                    chosen_support = u
                    break

            if chosen_support is not None:
                add_edge(v, chosen_support)
                remaining[v] -= 1
                continue

            # Query anchors are also stable support, even when they yield no
            # additional useful query gain.
            for q in query_anchor_order:
                if not _planned_edge_exists(G, planned_edges, v, q):
                    chosen_support = q
                    break

            if chosen_support is not None:
                add_edge(v, chosen_support)
                remaining[v] -= 1
                continue

            # Last feasibility fallback: a missing edge to any other candidate
            # vertex still gives v one support unit.
            for u in layer:
                if not _planned_edge_exists(G, planned_edges, v, u):
                    chosen_support = u
                    break

            if chosen_support is None:
                return None

            add_edge(v, chosen_support)
            remaining[v] -= 1
            if chosen_support in remaining and remaining[chosen_support] > 0:
                remaining[chosen_support] -= 1

    supplied = {v: 0 for v in active}
    for u, v in added:
        if u in supplied:
            supplied[u] += 1
        if v in supplied:
            supplied[v] += 1

    if any(supplied[v] < need[v] for v in active):
        raise RuntimeError(
            "Greedy activation-edge builder failed to satisfy a requisite degree."
        )

    return added


def _added_query_support_counts(
    activation_edges: List[Edge],
    Q_in: Set[Node],
    suffix: Set[Node],
) -> Dict[Node, int]:
    """Count newly planned query--suffix edges for unfinished queries."""
    counts: Dict[Node, int] = {}

    for u, v in activation_edges:
        if u in Q_in and v in suffix:
            counts[u] = counts.get(u, 0) + 1
        elif v in Q_in and u in suffix:
            counts[v] = counts.get(v, 0) + 1

    return counts


def decrement_query_deficits_by_suffix(
    G_after: nx.Graph,
    Q: Set[Node],
    old_R: Dict[Node, int],
    activated_suffix: Set[Node],
) -> Dict[Node, int]:
    """Decrease R(q) using neighbors in the newly activated suffix."""
    new_R = dict(old_R)
    activated_suffix = set(activated_suffix)

    # Iterate over q's neighbors instead of scanning the whole suffix for q.
    for q in Q:
        if old_R[q] <= 0:
            continue
        gained = sum(1 for v in G_after.neighbors(q) if v in activated_suffix)
        new_R[q] = max(0, old_R[q] - gained)

    return new_R


def best_candidate_of_component(
    G: nx.Graph,
    S: Set[Node],
    component_index: int,
    anchored_core: Set[Node],
    Q_in: Set[Node],
    Q: Set[Node],
    R: Dict[Node, int],
    k: int,
    stable_non_query_order: Optional[List[Node]] = None,
    query_anchor_order: Optional[List[Node]] = None,
    min_score_to_beat: float = 0.0,
) -> Optional[Candidate]:
    """
    Evaluate S_i^j = L_j union ... union L_h for one shell component.

    For the selected onion suffix:
      * Activating L_j activates the whole suffix S_i^j.
      * For v in L_j, r_j(v) = max(0, k - |N(v) intersect
        (anchored_core union S_i^j)|).
      * N_S(S_i^j) is the total requisite support over every v in S_i^j,
        matching the paper definition.

    Paper-aligned selection score:

        Score_Q(S_i^j) = G_Q(S_i^j) / N_S(S_i^j).

    The activation-edge set is still constructed separately to realize those
    support units; its cardinality is not used as the score denominator.

    Speed pruning uses the same support-based score. A direct action has
    support-efficiency 1, so direct actions win ties.
    """
    layers = onion_layers_for_component(G, S, anchored_core, k)
    if not layers:
        return None

    Q_in = set(Q_in)
    suffix: Set[Node] = set()
    suffix_query_count: Dict[Node, int] = {q: 0 for q in Q_in}
    best: Optional[Candidate] = None
    best_key = None

    total_remaining_query_deficit = sum(R[q] for q in Q_in)

    for j in range(len(layers) - 1, -1, -1):
        candidate = set(layers[j])
        if not candidate:
            continue

        suffix.update(candidate)

        for v in candidate:
            for u in G.neighbors(v):
                if u in Q_in:
                    suffix_query_count[u] += 1

        requisite = requisite_degrees_for_candidate(
            G=G,
            candidate_nodes=suffix,
            anchored_core=anchored_core,
            k=k,
        )
        # Paper definition:
        #   N_S(S_i^j) = sum_{v in S_i^j} max(0, k - deg(v, U)).
        # This is the number of REQUIRED SUPPORT UNITS, not the number of
        # activation edges that the greedy edge-placement routine happens to add.
        required_support = sum(requisite.values())
        if required_support <= 0:
            continue

        # Upper bound under the paper score: even if this candidate consumed
        # every remaining query deficit, the denominator stays N_S(S_i^j).
        optimistic_best_score = total_remaining_query_deficit / required_support
        if optimistic_best_score <= min_score_to_beat:
            continue

        # Existing suffix neighbors already consume part of R(q). Only the
        # residual part can still be gained through new candidate--query edges.
        query_gain_capacity = {
            q: max(0, R[q] - suffix_query_count[q])
            for q in Q_in
        }

        activation_edges = build_candidate_support_edges(
            G=G,
            candidate_layer=set(suffix),
            requisite=requisite,
            anchored_core=anchored_core,
            Q=Q,
            stable_non_query_order=stable_non_query_order,
            query_anchor_order=query_anchor_order,
            query_gain_capacity=query_gain_capacity,
        )
        if not activation_edges:
            continue

        extra_query_support = _added_query_support_counts(
            activation_edges=activation_edges,
            Q_in=Q_in,
            suffix=suffix,
        )

        GQ = 0
        for q in Q_in:
            support = suffix_query_count[q] + extra_query_support.get(q, 0)
            GQ += min(R[q], support)

        if GQ <= 0:
            continue

        # edge_cost is retained only as the realized number of added edges.
        # It is NOT the denominator of Score_Q.
        edge_cost = len(activation_edges)
        score_q = GQ / required_support

        # A direct action wins ties.
        if score_q <= min_score_to_beat:
            continue

        key = (
            score_q,
            GQ,
            -required_support,
            -edge_cost,
            -len(suffix),
        )

        if best is None or key > best_key:
            best = Candidate(
                component_index=component_index,
                layer_start=j + 1,
                nodes=set(suffix),
                candidate_layer=candidate,
                support_need=required_support,
                activation_edges=list(activation_edges),
                query_gain=GQ,
                score_q=score_q,
            )
            best_key = key

    return best

def best_query_query_edge(
    G: nx.Graph,
    Q: Set[Node],
    R: Dict[Node, int],
) -> Optional[Edge]:
    """
    Fill remaining deficits only with missing query-query edges.

    Prefer an edge whose two endpoints are both unfinished (gain 2). If only
    one endpoint is unfinished, an edge to a finished query has gain 1.
    """
    q_list = list(Q)
    best: Optional[Edge] = None
    best_key = None

    for i, u in enumerate(q_list):
        for v in q_list[i + 1:]:
            if G.has_edge(u, v):
                continue

            gain = int(R[u] > 0) + int(R[v] > 0)
            if gain == 0:
                continue

            key = (
                gain,
                R[u] + R[v],
                G.degree(u) + G.degree(v),
            )
            if best is None or key > best_key:
                best = (u, v)
                best_key = key

    return best



def query_query_edge_gain(
    edge: Optional[Edge],
    R: Dict[Node, int],
) -> int:
    """Return total query-deficit reduction of one query-query edge."""
    if edge is None:
        return 0
    u, v = edge
    return int(R.get(u, 0) > 0) + int(R.get(v, 0) > 0)

def decrement_query_deficits_by_query_edge(
    R: Dict[Node, int],
    edge: Edge,
) -> Dict[Node, int]:
    """A new query-query edge gives one additional support to each endpoint."""
    u, v = edge
    new_R = dict(R)
    new_R[u] = max(0, new_R[u] - 1)
    new_R[v] = max(0, new_R[v] - 1)
    return new_R


def best_query_core_edge(
    G: nx.Graph,
    Q: Set[Node],
    R: Dict[Node, int],
    ordinary_core: Set[Node],
    core_order: Optional[List[Node]] = None,
) -> Optional[Edge]:
    """
    Return one missing edge from an unfinished query to the ordinary k-core.

    This is used only after no useful query-query edge remains.  Only non-query
    vertices of the ordinary C_k(G_A) are used here, because query-query edges
    are handled by the previous fallback stage.
    """
    unfinished = [q for q in Q if R[q] > 0]
    if not unfinished:
        return None

    unfinished.sort(key=lambda q: (R[q], G.degree(q)), reverse=True)

    if core_order is None:
        core_order = sorted(
            ordinary_core - Q,
            key=lambda u: G.degree(u),
            reverse=True,
        )

    for q in unfinished:
        for u in core_order:
            if u != q and not G.has_edge(q, u):
                return q, u

    return None


def decrement_query_deficits_by_core_edge(
    R: Dict[Node, int],
    edge: Edge,
    Q: Set[Node],
) -> Dict[Node, int]:
    """A query--ordinary-core edge gives one stable support to its query endpoint."""
    u, v = edge
    new_R = dict(R)
    if u in Q and new_R[u] > 0:
        new_R[u] -= 1
    elif v in Q and new_R[v] > 0:
        new_R[v] -= 1
    else:
        raise ValueError("query-core edge has no unfinished query endpoint")
    return new_R


def _final_core_and_valid(
    G: nx.Graph,
    Q: Set[Node],
    k: int,
) -> Tuple[Set[Node], bool]:
    """Compute the final ordinary k-core once and validate every query."""
    final_core = k_core_nodes(G, k)
    return final_core, Q.issubset(final_core)


def query_centric_kcore_greedy(
    G: nx.Graph,
    Q: Iterable[Node],
    k: int,
    max_iterations: int = 10000,
    verbose: bool = False,
    initial_core: Optional[Set[Node]] = None,
    return_final_core: bool = False,
):
    """
    Query-centric greedy k-core activation with anchored queries.

    Full mode evaluates onion candidates with the paper's support-efficiency:

      1) onion candidate S_i^j:
            Score_Q = G_Q / N_S(S_i^j)
      2) direct query-query / query-core actions:
            support-efficiency = 1

    Candidate search is restricted to C_{k-1}(G_A) \\ C_k(G_A). A candidate is
    used only when its Score_Q is strictly greater than the direct
    support-efficiency; direct actions win ties.

    ``initial_core`` may be supplied by the experiment runner. Because edge
    insertion cannot destroy an existing k-core, the initial ordinary C_k(G)
    remains stable direct support throughout the run.

    With ``return_final_core=True`` the final ordinary core is returned as a
    fourth result so the experiment runner does not compute it twice.
    """
    if G.is_directed():
        raise ValueError("G must be an undirected graph.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    GA = G.copy()
    GA.remove_edges_from(nx.selfloop_edges(GA))
    Q = set(Q)

    missing = Q - set(GA.nodes())
    if missing:
        raise ValueError(f"Query vertices not in G: {missing}")

    if initial_core is None:
        stable_ordinary_core = k_core_nodes(GA, k)
    else:
        stable_ordinary_core = set(initial_core)

    anchored_core = anchored_core_nodes(GA, k, Q)
    R = initial_query_deficits(GA, Q, k, anchored_core)

    added_edges: List[Edge] = []
    history: List[Dict[str, Any]] = []
    component_phase = True

    ordinary_core_cache: Set[Node] = set(stable_ordinary_core)
    ordinary_core_order: List[Node] = sorted(
        ordinary_core_cache - Q,
        key=lambda x: GA.degree(x),
        reverse=True,
    )
    ordinary_core_refreshed = False

    for iteration in range(1, max_iterations + 1):
        if all(R[q] == 0 for q in Q):
            final_core, valid = _final_core_and_valid(GA, Q, k)
            if not valid:
                raise RuntimeError(
                    "All anchored query deficits reached zero, but the final "
                    "ordinary k-core validation failed."
                )
            if return_final_core:
                return GA, added_edges, history, final_core
            return GA, added_edges, history

        Q_in = {q for q in Q if R[q] > 0}

        # Best currently available direct action.
        qq_edge = best_query_query_edge(GA, Q, R)
        qq_gain = query_query_edge_gain(qq_edge, R)
        # A direct q-q edge supplies one support unit to each unfinished endpoint.
        # Therefore its query-deficit reduction per supplied support unit is 1.
        qq_score = 1.0 if qq_edge is not None else 0.0

        qcore_edge: Optional[Edge] = None
        qcore_score = 0.0

        # If q-q exists its support score is 1, so query-core can only tie it.
        if qq_edge is None:
            qcore_edge = best_query_core_edge(
                G=GA,
                Q=Q,
                R=R,
                ordinary_core=ordinary_core_cache,
                core_order=ordinary_core_order,
            )
            if qcore_edge is None and not ordinary_core_refreshed:
                ordinary_core_cache = k_core_nodes(GA, k)
                ordinary_core_order = sorted(
                    ordinary_core_cache - Q,
                    key=lambda x: GA.degree(x),
                    reverse=True,
                )
                ordinary_core_refreshed = True
                qcore_edge = best_query_core_edge(
                    G=GA,
                    Q=Q,
                    R=R,
                    ordinary_core=ordinary_core_cache,
                    core_order=ordinary_core_order,
                )
            if qcore_edge is not None:
                qcore_score = 1.0

        direct_score = max(qq_score, qcore_score)

        # Original search range: only C_{k-1}(G_A) \\ C_k(G_A).
        chosen: Optional[Candidate] = None

        if component_phase:
            lower = anchored_core_nodes(GA, k - 1, Q)
            region = lower - anchored_core - Q

            if region:
                components = [
                    set(c)
                    for c in nx.connected_components(GA.subgraph(region))
                ]

                if components:
                    stable_non_query_order = sorted(
                        anchored_core - Q,
                        key=lambda x: GA.degree(x),
                        reverse=True,
                    )
                    query_anchor_order = sorted(
                        Q,
                        key=lambda x: (
                            R.get(x, 0) > 0,
                            R.get(x, 0),
                            GA.degree(x),
                        ),
                        reverse=True,
                    )

                    component_winners: List[Candidate] = []
                    for i, S in enumerate(components):
                        cand = best_candidate_of_component(
                            G=GA,
                            S=S,
                            component_index=i,
                            anchored_core=anchored_core,
                            Q_in=Q_in,
                            Q=Q,
                            R=R,
                            k=k,
                            stable_non_query_order=stable_non_query_order,
                            query_anchor_order=query_anchor_order,
                            min_score_to_beat=direct_score,
                        )
                        if cand is not None:
                            component_winners.append(cand)

                    if component_winners:
                        chosen = max(
                            component_winners,
                            key=lambda c: (
                                c.score_q,
                                c.query_gain,
                                -c.support_need,
                                -len(c.activation_edges),
                                -len(c.nodes),
                            ),
                        )

            if chosen is not None and chosen.score_q > direct_score:
                chosen_contribution = query_contribution_count(
                    GA, chosen.nodes, Q_in
                )

                GA.add_edges_from(chosen.activation_edges)
                added_edges.extend(chosen.activation_edges)

                anchored_core = anchored_core_nodes(GA, k, Q)
                if not chosen.nodes.issubset(anchored_core):
                    raise RuntimeError(
                        "Selected onion suffix did not enter the anchored "
                        "k-core after adding support to its candidate layer."
                    )

                R = decrement_query_deficits_by_suffix(
                    G_after=GA,
                    Q=Q,
                    old_R=R,
                    activated_suffix=chosen.nodes,
                )

                history.append({
                    "iteration": iteration,
                    "type": "component",
                    "shell_depth": 1,
                    "component_index": chosen.component_index,
                    "layer_start": chosen.layer_start,
                    "candidate_layer": set(chosen.candidate_layer),
                    "candidate_size": len(chosen.candidate_layer),
                    "required_support": chosen.support_need,
                    "Sij": set(chosen.nodes),
                    "activation_edges": list(chosen.activation_edges),
                    "edge_cost": len(chosen.activation_edges),
                    "GQ": chosen.query_gain,
                    "ScoreQ": chosen.score_q,
                    "direct_score_to_beat": direct_score,
                    "node_query_contribution": chosen_contribution,
                    "R": dict(R),
                    "finished_queries": {q for q in Q if R[q] == 0},
                    "anchored_core_size": len(anchored_core),
                })

                if verbose:
                    print(
                        f"[{iteration}] k-1 component, "
                        f"S_i^{chosen.layer_start}, "
                        f"NS={chosen.support_need}, "
                        f"edges={len(chosen.activation_edges)}, "
                        f"GQ={chosen.query_gain}, "
                        f"ScoreQ={chosen.score_q:.4f}, "
                        f"direct={direct_score:.4f}, R={R}"
                    )
                continue

            # No candidate through the configured search depth strictly beats a
            # direct support action. Permanently leave component search.
            component_phase = False

        # Direct action. q-q wins ties.
        if qq_edge is not None:
            GA.add_edge(*qq_edge)
            added_edges.append(qq_edge)
            R = decrement_query_deficits_by_query_edge(R, qq_edge)

            history.append({
                "iteration": iteration,
                "type": "query-query fallback",
                "edge": qq_edge,
                "edge_cost": 1,
                "gain": qq_gain,
                "ScoreQ": qq_score,
                "R": dict(R),
                "finished_queries": {q for q in Q if R[q] == 0},
                "anchored_core_size": len(anchored_core),
            })

            if verbose:
                print(
                    f"[{iteration}] query-query={qq_edge}, "
                    f"gain={qq_gain}, ScoreQ={qq_score:.1f}, R={R}"
                )
            continue

        if qcore_edge is None:
            qcore_edge = best_query_core_edge(
                G=GA,
                Q=Q,
                R=R,
                ordinary_core=ordinary_core_cache,
                core_order=ordinary_core_order,
            )

        if qcore_edge is None and not ordinary_core_refreshed:
            ordinary_core_cache = k_core_nodes(GA, k)
            ordinary_core_order = sorted(
                ordinary_core_cache - Q,
                key=lambda x: GA.degree(x),
                reverse=True,
            )
            ordinary_core_refreshed = True
            qcore_edge = best_query_core_edge(
                G=GA,
                Q=Q,
                R=R,
                ordinary_core=ordinary_core_cache,
                core_order=ordinary_core_order,
            )

        if qcore_edge is None:
            unfinished = {q: R[q] for q in Q if R[q] > 0}
            raise RuntimeError(
                "Remaining query deficits cannot be resolved by query-query "
                "edges or direct query-to-ordinary-k-core edges. "
                f"unfinished={unfinished}."
            )

        GA.add_edge(*qcore_edge)
        added_edges.append(qcore_edge)
        R = decrement_query_deficits_by_core_edge(R, qcore_edge, Q)

        history.append({
            "iteration": iteration,
            "type": "query-core fallback",
            "edge": qcore_edge,
            "edge_cost": 1,
            "gain": 1,
            "ScoreQ": 1.0,
            "R": dict(R),
            "finished_queries": {q for q in Q if R[q] == 0},
            "ordinary_core_size": len(ordinary_core_cache),
        })

        if verbose:
            print(f"[{iteration}] query-core={qcore_edge}, ScoreQ=1.0, R={R}")

    raise RuntimeError("Maximum number of iterations exceeded.")



def _baseline_query_deficits(
    G: nx.Graph,
    Q: Set[Node],
    k: int,
    initial_core: Set[Node],
) -> Dict[Node, int]:
    """
    Deficit used by the two simple comparison baselines.

    The initial ordinary k-core is permanent support.  Because every query is
    required to survive in the final solution, already-existing query-query
    edges can also be counted as support.  Therefore, once every deficit below
    reaches zero, C_k(G) union Q contains a subgraph in which every query has
    at least k neighbors, which is sufficient for all queries to belong to the
    final ordinary k-core.
    """
    support_region = set(initial_core) | set(Q)
    return {
        q: max(
            0,
            k - sum(1 for u in G.neighbors(q) if u in support_region),
        )
        for q in Q
    }


def query_query_only_baseline(
    G: nx.Graph,
    Q: Iterable[Node],
    k: int,
    max_iterations: int = 10000,
    verbose: bool = False,
    initial_core: Optional[Set[Node]] = None,
    return_final_core: bool = False,
):
    """
    Baseline that adds ONLY missing query-query edges.

    This bulk implementation preserves the baseline rule but avoids rescanning all
    query pairs after every inserted edge. Missing query-query adjacencies are
    prepared once, then deficits are filled greedily in memory:

      1) pair two unfinished queries whenever possible (gain 2);
      2) if no unfinished--unfinished pair is available, connect an unfinished
         query to any remaining query with a missing edge (gain 1).

    All selected edges are inserted into the graph in one batch before the final
    ordinary k-core validation.
    """
    if G.is_directed():
        raise ValueError("G must be an undirected graph.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    GA = G.copy()
    GA.remove_edges_from(nx.selfloop_edges(GA))
    Q = set(Q)

    missing = Q - set(GA.nodes())
    if missing:
        raise ValueError(f"Query vertices not in G: {missing}")

    stable_core = k_core_nodes(GA, k) if initial_core is None else set(initial_core)
    R = _baseline_query_deficits(GA, Q, k, stable_core)

    added_edges: List[Edge] = []
    history: List[Dict[str, Any]] = []

    if all(R[q] == 0 for q in Q):
        final_core, valid = _final_core_and_valid(GA, Q, k)
        if not valid:
            raise RuntimeError(
                "query-query-only deficits reached zero, but final ordinary "
                "k-core validation failed."
            )
        if return_final_core:
            return GA, added_edges, history, final_core
        return GA, added_edges, history

    # Build the missing query-query adjacency once. Afterwards no NetworkX pair
    # scan is needed while deficits change.
    q_list = list(Q)
    available: Dict[Node, Set[Node]] = {q: set() for q in Q}
    for i, u in enumerate(q_list):
        for v in q_list[i + 1:]:
            if not GA.has_edge(u, v):
                available[u].add(v)
                available[v].add(u)

    degree_hint = {q: GA.degree(q) for q in Q}
    iteration = 0

    def use_edge(u: Node, v: Node, gain: int) -> None:
        nonlocal iteration
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError("Maximum number of iterations exceeded.")

        available[u].discard(v)
        available[v].discard(u)
        added_edges.append((u, v))

        if R[u] > 0:
            R[u] -= 1
        if R[v] > 0:
            R[v] -= 1

        history.append({
            "iteration": iteration,
            "type": "query-query only",
            "edge": (u, v),
            "edge_cost": 1,
            "gain": gain,
            "R": dict(R),
        })

        if verbose:
            print(
                f"[{iteration}] query-query-only={(u, v)}, "
                f"gain={gain}, R={R}"
            )

    # Phase 1: always prefer an edge whose two endpoints are both unfinished.
    # Larger deficits are handled first; this is the same gain-2 priority as the
    # previous one-edge-at-a-time baseline, without rescanning every pair.
    while True:
        unfinished = [q for q in Q if R[q] > 0]
        if len(unfinished) < 2:
            break

        unfinished.sort(key=lambda q: (R[q], degree_hint[q]), reverse=True)
        made_progress = False

        for u in unfinished:
            if R[u] <= 0:
                continue

            candidates = [v for v in available[u] if R[v] > 0]
            if not candidates:
                continue

            v = max(candidates, key=lambda x: (R[x], degree_hint[x]))
            use_edge(u, v, gain=2)
            made_progress = True

        if not made_progress:
            break

    # Phase 2: if some deficit remains, a missing edge to an already-finished
    # query still gives one support unit and is allowed by the Q-Q-only baseline.
    for u in sorted(Q, key=lambda q: (R[q], degree_hint[q]), reverse=True):
        while R[u] > 0:
            if not available[u]:
                unfinished = {q: R[q] for q in Q if R[q] > 0}
                raise RuntimeError(
                    "query-query-only cannot resolve the remaining deficits because "
                    "no useful missing query-query edge remains. "
                    f"unfinished={unfinished}."
                )

            # Prefer a still-unfinished partner if one exists; otherwise any query
            # with a missing edge is enough to reduce u by one.
            candidates = list(available[u])
            v = max(
                candidates,
                key=lambda x: (R[x] > 0, R[x], degree_hint[x]),
            )
            gain = 1 + int(R[v] > 0)
            use_edge(u, v, gain=gain)

    GA.add_edges_from(added_edges)

    final_core, valid = _final_core_and_valid(GA, Q, k)
    if not valid:
        raise RuntimeError(
            "query-query-only deficits reached zero, but final ordinary "
            "k-core validation failed."
        )

    if return_final_core:
        return GA, added_edges, history, final_core
    return GA, added_edges, history


def query_kcore_only_baseline(
    G: nx.Graph,
    Q: Iterable[Node],
    k: int,
    max_iterations: int = 10000,
    verbose: bool = False,
    initial_core: Optional[Set[Node]] = None,
    return_final_core: bool = False,
):
    """
    Baseline that adds ONLY query-to-initial-ordinary-k-core edges.

    For each query q, exactly R(q) missing edges to the initial ordinary k-core
    are selected in one scan of the core order. The previous implementation
    searched the core again after every single inserted edge; this bulk version
    produces the same baseline action type with much less repeated work.
    """
    if G.is_directed():
        raise ValueError("G must be an undirected graph.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    GA = G.copy()
    GA.remove_edges_from(nx.selfloop_edges(GA))
    Q = set(Q)

    missing = Q - set(GA.nodes())
    if missing:
        raise ValueError(f"Query vertices not in G: {missing}")

    stable_core = k_core_nodes(GA, k) if initial_core is None else set(initial_core)
    if not stable_core:
        raise RuntimeError("Initial ordinary k-core is empty.")

    R = _baseline_query_deficits(GA, Q, k, stable_core)
    core_order = sorted(
        stable_core - Q,
        key=lambda u: GA.degree(u),
        reverse=True,
    )

    added_edges: List[Edge] = []
    history: List[Dict[str, Any]] = []
    iteration = 0

    # Each query is independent in this baseline: a q--core edge only reduces
    # that query's deficit. Therefore we can satisfy all R(q) in one pass per q.
    for q in sorted(Q, key=lambda x: (R[x], GA.degree(x)), reverse=True):
        need = R[q]
        if need <= 0:
            continue

        # Membership in the original neighbor set lets us test missing q--core
        # edges in O(1) while scanning the fixed initial-core order once.
        existing_neighbors = set(GA.neighbors(q))
        chosen: List[Node] = []

        for u in core_order:
            if u == q or u in existing_neighbors:
                continue
            chosen.append(u)
            if len(chosen) == need:
                break

        if len(chosen) < need:
            unfinished = {x: R[x] for x in Q if R[x] > 0}
            raise RuntimeError(
                "query-k-core-only cannot resolve the remaining deficits because "
                "there are not enough missing query-to-initial-k-core edges. "
                f"unfinished={unfinished}."
            )

        for u in chosen:
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError("Maximum number of iterations exceeded.")

            edge = (q, u)
            added_edges.append(edge)
            R[q] -= 1

            history.append({
                "iteration": iteration,
                "type": "query-k-core only",
                "edge": edge,
                "edge_cost": 1,
                "gain": 1,
                "R": dict(R),
            })

            if verbose:
                print(f"[{iteration}] query-k-core-only={edge}, R={R}")

    GA.add_edges_from(added_edges)

    final_core, valid = _final_core_and_valid(GA, Q, k)
    if not valid:
        raise RuntimeError(
            "query-k-core-only deficits reached zero, but final ordinary "
            "k-core validation failed."
        )

    if return_final_core:
        return GA, added_edges, history, final_core
    return GA, added_edges, history
