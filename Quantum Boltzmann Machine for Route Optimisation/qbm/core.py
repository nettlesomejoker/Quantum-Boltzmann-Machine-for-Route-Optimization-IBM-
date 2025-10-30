"""
Core utilities for Quantum Boltzmann Machine route optimisation.
This module extracts algorithmic parts from the notebook so they can be used
by a UI or CLI.
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd

# Try to import D-Wave packages if available (non-fatal)
DWAVE_AVAILABLE = False
try:
    from dwave.system import DWaveSampler, EmbeddingComposite
    import dimod
    DWAVE_AVAILABLE = True
except Exception:
    DWAVE_AVAILABLE = False


def build_distance_lookup(df: pd.DataFrame):
    d = {}
    for _, row in df.iterrows():
        a, b = str(row.get("Source")), str(row.get("Destination"))
        # last column assumed to be distance/weight
        w = float(row[df.columns[-1]])
        d[(a, b)] = min(d.get((a, b), w), w)
    for (a, b), w in list(d.items()):
        if (b, a) not in d:
            d[(b, a)] = w
    return d


def build_distance_matrix(cities, dist_lookup):
    N = len(cities)
    D = np.zeros((N, N))
    for i, a in enumerate(cities):
        for j, b in enumerate(cities):
            D[i, j] = dist_lookup.get((a, b), 1e9) if a != b else 0.0
    return D


def var_index(i, p, N):
    return i * N + p


def build_tsp_qubo(D: np.ndarray, A=1000.0, B=1.0):
    N = D.shape[0]
    Q = defaultdict(float)
    for p in range(N):
        for i in range(N):
            idx_i = var_index(i, p, N)
            Q[(idx_i, idx_i)] += -A
            for k in range(i + 1, N):
                idx_k = var_index(k, p, N)
                key = (min(idx_i, idx_k), max(idx_i, idx_k))
                Q[key] += 2.0 * A

    for i in range(N):
        for p in range(N):
            idx_i_p = var_index(i, p, N)
            Q[(idx_i_p, idx_i_p)] += -A
            for q in range(p + 1, N):
                idx_i_q = var_index(i, q, N)
                key = (min(idx_i_p, idx_i_q), max(idx_i_p, idx_i_q))
                Q[key] += 2.0 * A

    for p in range(N):
        p_next = (p + 1) % N
        for i in range(N):
            for j in range(N):
                idx_i_p = var_index(i, p, N)
                idx_j_next = var_index(j, p_next, N)
                key = (min(idx_i_p, idx_j_next), max(idx_i_p, idx_j_next))
                Q[key] += B * float(D[i, j])

    return Q


def onehot_from_route(route, N):
    x = np.zeros(N * N, dtype=int)
    for p, i in enumerate(route):
        x[var_index(i, p, N)] = 1
    return x


def decode_onehot_to_route(x, N):
    x_mat = x.reshape(N, N)
    chosen = np.argmax(x_mat, axis=0).tolist()
    seen, fixed = set(), []
    for c in chosen:
        if c not in seen:
            fixed.append(int(c)); seen.add(c)
        else:
            candidate = next(i for i in range(N) if i not in seen)
            fixed.append(int(candidate)); seen.add(candidate)
    return fixed


def qubo_energy(Q, x):
    e = 0.0
    for (i, j), w in Q.items():
        e += w * float(x[i]) * float(x[j])
    return e


def nearest_neighbour_from_matrix(D, start=0):
    N = D.shape[0]
    remaining = set(range(N))
    route = [start]
    remaining.remove(start)
    current = start
    while remaining:
        nxt = min(remaining, key=lambda j: D[current, j])
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route


def simulated_annealing_local(Q, N, steps=20000, init_onehot=None):
    """Simulated annealing that preserves feasibility by operating on routes.

    - If init_onehot is provided (length N*N array), it is used as the starting solution.
    - Otherwise a greedy nearest-neighbour route is used (from an approximate D built from Q).
    Returns a tuple (best_x, best_e) where best_x is a one-hot flat numpy array.
    """
    # helper to convert route -> onehot and energy
    def route_to_onehot(route):
        return onehot_from_route(route, N)

    # initialize route
    if init_onehot is not None:
        try:
            init_x = np.asarray(init_onehot, dtype=int).flatten()
            route = decode_onehot_to_route(init_x, N)
        except Exception:
            route = nearest_neighbour_from_matrix(build_distance_matrix_from_qubo(Q, N), 0)
    else:
        route = nearest_neighbour_from_matrix(build_distance_matrix_from_qubo(Q, N), 0)

    x = route_to_onehot(route)
    try:
        e = qubo_energy(Q, x); best_route, best_e = route.copy(), e
    except Exception:
        # fallback to zero vector (should not usually happen)
        x = np.zeros(N * N, dtype=int)
        e = qubo_energy(Q, x)
        best_route, best_e = decode_onehot_to_route(x, N), e

    T = 1000.0
    for t in range(steps):
        # propose a swap between two positions to keep solution feasible
        p1, p2 = np.random.choice(range(N), size=2, replace=False)
        new_route = route.copy()
        new_route[p1], new_route[p2] = new_route[p2], new_route[p1]
        x_new = route_to_onehot(new_route)
        e_new = qubo_energy(Q, x_new)
        if e_new < e or np.random.rand() < math.exp((e - e_new) / max(T, 1e-12)):
            route, x, e = new_route, x_new, e_new
            if e < best_e:
                best_route, best_e = route.copy(), e
        T *= 0.99995

    best_x = route_to_onehot(best_route)
    return best_x, best_e


def build_distance_matrix_from_qubo(Q, N):
    """Approximate distance matrix from Q for heuristic initialization.
    If exact D is not available, create a trivial matrix to allow route generation.
    """
    # Create small positive matrix so greedy has something to work with
    D = np.ones((N, N)) * 100.0
    np.fill_diagonal(D, 0.0)
    return D


def quantum_annealing_solve(Q, N, steps=20000, restarts=1, num_reads=100):
    """Try D-Wave hardware; if not available, fallback to local SA.
    Parameters:
      Q: QUBO dict
      N: number of cities
      steps: SA steps per restart (used for fallback)
      restarts: number of SA restarts (fallback)
      num_reads: number of reads for D-Wave sampler
    Returns a tuple (best_x, best_energy).
    """
    if DWAVE_AVAILABLE:
        try:
            sampler = EmbeddingComposite(DWaveSampler())
            Q_dwave = {(i, j): float(w) for (i, j), w in Q.items()}
            response = sampler.sample_qubo(Q_dwave, num_reads=num_reads)
            best_sample = response.first.sample
            x = np.array([best_sample.get(i, 0) for i in range(N * N)])
            return x, float(response.first.energy)
        except Exception:
            # fall through to local SA
            pass

    # fallback: run simulated annealing initialized from greedy route for feasibility
    best_x, best_e = None, float('inf')
    init_route = nearest_neighbour_from_matrix(build_distance_matrix_from_qubo(Q, N), 0)
    init_onehot = onehot_from_route(init_route, N)
    for r in range(max(1, restarts)):
        x_r, e_r = simulated_annealing_local(Q, N, steps=steps, init_onehot=init_onehot)
        if e_r < best_e:
            best_x, best_e = x_r, e_r
    return best_x, best_e


def run_problem(distance_csv: str, orders_csv: str):
    """High level runner: loads CSVs, builds problem, returns D, Q, cities, N."""
    dist_df = pd.read_csv(distance_csv)
    orders = pd.read_csv(orders_csv)
    dist_lookup = build_distance_lookup(dist_df)

    # pick a batch
    if "Order_ID" not in orders.columns:
        batch_id = orders.iloc[0, 0]
        batch = orders[orders.iloc[:, 0] == batch_id]
    else:
        batch_id = orders["Order_ID"].iloc[0]
        batch = orders[orders["Order_ID"] == batch_id]

    source_city = str(batch["Source"].mode().iloc[0]) if "Source" in batch.columns else str(batch.iloc[0, 1])
    destinations = sorted(set(batch["Destination"].astype(str))) if "Destination" in batch.columns else sorted(set(batch.iloc[:, 2].astype(str)))
    cities = [source_city] + [c for c in destinations if c != source_city]
    N = len(cities)
    D = build_distance_matrix(cities, dist_lookup)
    Q = build_tsp_qubo(D)

    # try to extract coordinates (if present) from distance or orders files
    coords = {}
    coord_pairs = [
        ("x", "y"), ("X", "Y"), ("lon", "lat"), ("longitude", "latitude"),
        ("Longitude", "Latitude"), ("LAT", "LON"), ("latitude", "longitude"), ("lat", "lon")
    ]

    def extract_from_df(df):
        """Return (found_coords_dict, (xcol, ycol)) or ({}, None) if not found."""
        possible_city_cols = [c for c in ("Source", "Destination", "City", "Location", "Name") if c in df.columns]
        if not possible_city_cols:
            return {}, None
        for xcol, ycol in coord_pairs:
            if xcol in df.columns and ycol in df.columns:
                found = {}
                for _, row in df.iterrows():
                    city = None
                    for cc in possible_city_cols:
                        if pd.notna(row.get(cc)):
                            city = str(row.get(cc))
                            break
                    if city is None:
                        continue
                    try:
                        xv = float(row[xcol])
                        yv = float(row[ycol])
                        found[city] = (xv, yv)
                    except Exception:
                        continue
                if found:
                    return found, (xcol, ycol)
        return {}, None

    # prefer orders CSV (may have destination coordinates), then distance CSV
    coords, coord_cols = extract_from_df(orders)
    if not coords:
        coords, coord_cols = extract_from_df(dist_df)

    if not coords:
        coords = None
        coord_cols = None

    return {
        "D": D,
        "Q": Q,
        "cities": cities,
        "N": N,
        "coords": coords,
        "coord_cols": coord_cols,
    }
