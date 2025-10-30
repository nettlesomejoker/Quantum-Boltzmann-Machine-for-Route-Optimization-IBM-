"""Debug helper: run the solver on provided CSVs and print diagnostics.

Usage:
  & '.venv/Scripts/python.exe' -m qbm.debug_solution --dist path/to/dist.csv --orders path/to/orders.csv

This script prints:
 - number of 1-bits in the returned sample
 - indices of set bits (first 40 shown)
 - QUBO energy (from qubo_energy)
 - decoded route and travel length
 - counts and components to help debug why energy is negative
"""
import argparse
from qbm import core
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dist', required=True)
    p.add_argument('--orders', required=True)
    args = p.parse_args()

    prob = core.run_problem(args.dist, args.orders)
    D, Q, cities, N = prob['D'], prob['Q'], prob['cities'], prob['N']
    print(f"Loaded problem with N={N} cities")

    x, energy = core.quantum_annealing_solve(Q, N)
    x = np.asarray(x, dtype=int)
    ones = int(x.sum())
    print(f"Solver returned {ones} ones (bits set)")
    print(f"First 80 bits: {x.tolist()[:80]}")

    # energy reported by solver may differ; compute directly
    e_calc = core.qubo_energy(Q, x)
    print(f"Energy reported: {energy}")
    print(f"Energy (computed): {e_calc}")

    route = core.decode_onehot_to_route(x, N)
    length = sum(D[route[i], route[(i+1) % N]] for i in range(N))
    print(f"Decoded route (indices): {route}")
    print(f"Route length (sum of D): {length}")

    # show some Q diagonal info for a few variables
    diag_vals = []
    for i in range(min(len(x), 20)):
        diag_vals.append((i, Q.get((i, i), 0.0)))
    print("First 20 diagonal Q values:")
    for idx, v in diag_vals:
        print(f"  idx={idx} Q_ii={v}")

    # Suggest next steps
    print('\nHints:')
    print(' - If number of ones != N then the sample is infeasible; solver may be exploiting negative linear terms.')
    print(' - Try increasing penalty A in build_tsp_qubo or increase SA steps/num_reads.')
    print(' - You can also run the greedy solver for a baseline:')
    greedy = core.nearest_neighbour_from_matrix(D, 0)
    greedy_len = sum(D[greedy[i], greedy[(i+1) % N]] for i in range(N))
    print(f'   greedy route length = {greedy_len} (route {greedy})')


if __name__ == '__main__':
    main()
