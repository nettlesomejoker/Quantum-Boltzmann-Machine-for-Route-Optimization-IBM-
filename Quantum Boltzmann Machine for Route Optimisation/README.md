# Quantum Boltzmann Machine for Route Optimisation

This repository provides a compact prototype that encodes a routing/TSP-like problem as a QUBO (Quadratic Unconstrained Binary Optimization) and demonstrates solver workflows using both classical and quantum-style samplers. It is intended for experimentation, diagnostics and learning rather than production routing.

Key artifacts
- `qbm/core.py` — core algorithms and helpers (QUBO builder, encoders/decoders, greedy solver, simulated-annealing variants, runner)
- `qbm/streamlit_app.py` — Streamlit web UI (file upload, sample loader, unit conversion, solver controls, diagnostics & plots)
- `qbm/ui.py` — simple Tkinter desktop GUI (legacy/alternate interface)
- `qbm/debug_solution.py` — command-line debug tool that prints solver outputs and diagnostics
- `data/` — multiple sample CSVs included for quick testing (now includes 4-city, 8-city, coords and asymmetric examples)

This README documents the updated UI features, solver flows, diagnostics and practical tuning tips.

## What's new (recent updates)
- Units selector and display: the Streamlit UI now shows route lengths with a units selector (native, km, miles). The UI assumes CSV distances are in metres when converting to km/miles; choose "distance units (native)" if your CSV uses other units.
- Multiple sample datasets and a quick loader are included under `data/` and selectable from the sidebar.
- Raw vs Feasible workflow: the UI exposes a button "Run Raw (infeasible) then Feasible SA" which runs two flows in sequence and displays them side-by-side:
  1. Raw sampler (attempts D‑Wave hardware if available; otherwise runs a simulated bit-flip SA that may produce infeasible bit-vectors) — useful to study raw sampler behaviour and embedding/penalty sensitivity.
  2. Feasible simulated annealing (route-preserving swap-based SA) initialized from the greedy nearest-neighbour route — guarantees feasible one-hot solutions and produces realistic route lengths.
- Diagnostics: the UI now computes and displays bit counts, row/column sums, a feasibility flag, Hamming distance between raw and feasible bit-vectors, QUBO energies and decoded route lengths. These help explain why raw samples may show very negative energies yet not correspond to valid routes.
- Solver improvements: the default local SA fallback (`simulated_annealing_local`) operates on permutations using swaps (keeps feasibility) instead of naive bit-flips; simulated raw sampler uses a bit-flip Metropolis loop to mimic hardware outputs.

## How it works (brief)
- Encoding: one-hot encoding x_{i,p} where i is a city index and p is a position index. Flattened into an N*N-length binary vector.
- QUBO construction: Q aggregates two parts:
  - Constraint penalties A to enforce one-hot behaviour (one city per position and one position per city).
  - Travel cost terms scaled by B that couple position p to p+1 with D[i,j].
- Solvers:
  - Greedy nearest-neighbour baseline (`nearest_neighbour_from_matrix`) — O(N^2) heuristic.
  - Raw sampler: D‑Wave sampler if available (requires Ocean SDK & credentials); otherwise a simulated bit-flip SA that may produce infeasible x.
  - Feasible SA (`simulated_annealing_local`): swap-based simulated annealing that preserves feasibility by swapping two positions in the permutation.
- Decoding: `decode_onehot_to_route` picks argmax per position and repairs duplicates by assigning missing cities in first-available order. Note: different bit-vectors can decode to the same route.

## Algorithms included

This section lists the core algorithms implemented in the project, with brief descriptions, formulas and complexity notes.

1) One-hot encoding
 - Variables: x_{i,p} ∈ {0,1} for city i ∈ {0..N-1} at position p ∈ {0..N-1}.
 - Flattening: var_index(i,p,N) = i * N + p. The binary vector x has length N*N and can be reshaped to an (N, N) matrix where rows index cities and columns index positions.

2) QUBO construction (build_tsp_qubo)
 - Goal: minimize Energy(x) = A * (position-constraints + city-constraints) + B * travel-costs.
 - Constraints (one-hot): for each position p, enforce (sum_i x_{i,p} - 1)^2; for each city i, enforce (sum_p x_{i,p} - 1)^2.
 - Travel cost: for each adjacent position p and p+1, add B * D[i,j] * x_{i,p} * x_{j,p+1} for all i,j.
 - Conceptual energy formula:
   Energy(x) = A * ( sum_p (sum_i x_{i,p} - 1)^2 + sum_i (sum_p x_{i,p} - 1)^2 )
               + B * sum_p sum_{i,j} D[i,j] x_{i,p} x_{j,p+1}.
 - Implementation: `build_tsp_qubo(D, A, B)` expands the squared penalties into diagonal and pairwise QUBO coefficients and adds travel couplings; it returns a sparse dict Q mapping (i,j)->weight.
 - Complexity: naive assembly is O(N^3) due to loops across positions and city pairs.

3) Greedy nearest-neighbour baseline
 - Algorithm: start at a chosen start city (0 by default), repeatedly pick the nearest unvisited city until all visited.
 - Implementation: `nearest_neighbour_from_matrix(D, start)`.
 - Complexity: O(N^2) with a naive search over remaining cities per step.

4) Simulated annealing — infeasible (bit-flip) (simulated raw sampler)
 - Purpose: mimic raw sampler behaviour (hardware samplers or naive bit-space search) which do not enforce one-hot constraints.
 - Moves: flip a single bit x_k (0→1 or 1→0) chosen uniformly at random.
 - Acceptance: Metropolis criterion using temperature schedule T(t) (implemented multiplicative cooling e.g. T *= 0.99995).
 - Tracking: track the best x seen (may be infeasible) by QUBO energy.
 - Implementation: in `qbm/streamlit_app.py` the helper `simulate_infeasible` runs this loop; used when D-Wave hardware is not available or when simulating raw sampler behaviour.
 - Complexity: energy evaluation per move is O(nnz(Q)) where nnz(Q) is number of stored Q interactions; total cost ~ steps * O(nnz(Q)).

5) Simulated annealing — feasible (swap-based) (local permutation SA)
 - Purpose: search only within the feasible permutation subspace (routes) to get realistic route solutions.
 - State representation: a permutation (route) of length N. Convert to one-hot for energy evaluations.
 - Moves: swap two positions p1 and p2 in the route (this preserves one-hot feasibility: exactly one city per position and city appears once).
 - Acceptance: Metropolis criterion as above; energy computed by converting route → one-hot and using `qubo_energy(Q, x)`.
 - Implementation: `simulated_annealing_local(Q, N, steps, init_onehot=None)` — can initialize from a provided one-hot or from a greedy heuristic.
 - Complexity: each proposed swap requires computing new energy O(nnz(Q)) unless incremental evaluation is implemented; total cost ~ steps * O(nnz(Q)). The search space is permutations (N!), but local SA explores neighbors defined by swaps.

6) Quantum annealing flow (quantum_annealing_solve)
 - Attempt to use D‑Wave hardware via EmbeddingComposite(DWaveSampler()) and `sample_qubo(Q, num_reads)` when `DWAVE_AVAILABLE`.
 - If hardware is not available or sampling fails, fallback to the feasible simulated annealing restarts (calls `simulated_annealing_local` and returns best feasible x across restarts).
 - The Streamlit UI additionally runs a raw sampler attempt first (hardware or simulated bit-flip) and then runs feasible SA for comparison.

7) Decoding and repair
 - `decode_onehot_to_route(x, N)` reshapes x to (N, N), takes argmax across each position (column) to choose city for that position, and repairs duplicates by filling in missing cities in first-available order.
 - Note: this repair is deterministic and greedy; many different infeasible x may decode to the same repaired permutation, so always inspect bit-counts (ones) and row/col sums to detect infeasibility.

8) Diagnostics computed by tools / UI
 - QUBO energy: qubo_energy(Q,x) = sum_{(i,j) in Q} Q[(i,j)] * x[i] * x[j].
 - Bit-count ones: sum(x) should equal N for feasible solutions.
 - Row sums / column sums: each should be all-ones for a valid one-hot matrix.
 - Hamming distance: count differing bits between two x vectors (e.g., raw vs feasible).
 - Decoded route length: sum of D[route[i], route[(i+1)%N]] over i.

Complexity and practical limits
 - The variable count is N^2. Naive QUBO assembly and energy evaluation in this code are O(N^3) / O(nnz(Q)) respectively. For practical use, N beyond a few tens will become computationally expensive in the naive dense QUBO representation.
 - Feasible-SA restricts search to permutations and avoids interpreting infeasible raw x as valid routes; this is cheaper to reason about but still requires many steps for good optimization on larger N.

If you'd like, I can add a short worked example (4-city) with the exact Q matrix shown and small pseudocode blocks inserted in the README for copy/paste. Otherwise this Algorithms section should make the project internals and available algorithms explicit.

## Quick setup (Windows PowerShell)
Use the project virtual environment if present. From the repository root (PowerShell):

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install numpy pandas matplotlib streamlit
```

Optional (only if you plan to use D‑Wave hardware):

```powershell
& '.\.venv\Scripts\python.exe' -m pip install dwave-ocean-sdk dimod
```

If you want to run the exploratory notebook:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install jupyterlab
& '.\.venv\Scripts\python.exe' -m jupyter lab qbm_route_optimization.ipynb
```

## Run the Streamlit UI (recommended)

From the project root:

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run qbm/streamlit_app.py
```

Notes:
- If port 8501 is occupied you can pass `--server.port` (e.g. `--server.port 8502`).
- Use the sidebar to upload your Distance and Orders CSVs, or use the sample selector and click **Load selected sample**.
- Choose units in the sidebar to control length display (native / km / miles).

### Using the UI features
- Greedy: runs nearest-neighbour and shows decoded route, QUBO energy and a plot.
- Run Raw (infeasible) then Feasible SA: performs two runs and displays them side-by-side with diagnostics (energies, lengths, ones/row/col sums, Hamming distance, decoded routes and plots).

## CSV input format
- Distance CSV: should have columns Source, Destination and a numeric distance column (last column is used if unnamed). The code will try to read multiple rows and create a distance lookup (symmetric when only one direction provided).
- Orders CSV: should contain a batch of stops; `Source` and `Destination` columns are used. If `Order_ID` exists the first batch is selected. If coordinate columns exist (e.g. x/y, lon/lat) and a city column (Source/Destination/City) is present, the UI will plot using those coordinates.

## Diagnostics & common failure modes
- Infeasible raw samples: if penalty `A` is too small the sampler may produce bit-vectors with ones != N to exploit negative diagonal terms, producing very negative energy that does not represent a valid route. Check the UI diagnostics (ones, row/col sums) and increase `A` until violations disappear if you require feasible hardware samples.
- Decoder collisions: several different x may decode to the same route because decoding uses argmax + greedy repair. The UI shows Hamming distance between raw and feasible to reveal bit-level differences even when decoded routes look identical.

## Tuning tips
- Penalty A: increase until constraint violations are rare (try 1e3–1e5 depending on scale).
- Travel scale B: set relative to A so that travel cost is meaningful but doesn't encourage constraint violations.
- SA steps / restarts: increase for better exploration; feasible SA is permutation-constrained and generally more meaningful for route quality.

## Developer notes (where to look)
- `qbm/core.py`: QUBO builder (`build_tsp_qubo`), energy (`qubo_energy`), greedy (`nearest_neighbour_from_matrix`), feasible SA (`simulated_annealing_local`) and `quantum_annealing_solve` (tries hardware then falls back to SA).
- `qbm/streamlit_app.py`: UI wiring, unit conversion, sample loader, simulated raw sampler (bit-flip SA), diagnostics display and plotting.
- `qbm/debug_solution.py`: CLI debug helper showing bit-vector, energies, and decoded route length.

## Suggested improvements
- Randomize simulated raw starts and add restarts (the UI currently starts simulated raw from the greedy one-hot). Random starts will expose sampler variety more clearly.
- Overlay/diff plot: show raw and feasible routes on a single plot highlighting differing edges.
- Seed control: expose a random seed in the UI for reproducible runs.

## References
- D-Wave QUBO / QPU documentation: https://docs.dwavesys.com/
- QUBO and TSP formulations: standard literature on mapping TSP to QUBO / Ising models.
