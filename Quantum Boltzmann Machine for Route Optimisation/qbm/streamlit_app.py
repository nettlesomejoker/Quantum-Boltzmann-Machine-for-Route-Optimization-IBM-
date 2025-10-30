"""
Streamlit web UI for the Quantum Boltzmann Machine route optimisation.

Run with:
    streamlit run qbm/streamlit_app.py

This app uses the core helpers in `qbm.core` so it supports the same
features as the Tkinter UI: greedy solver, quantum (D-Wave) solver with
local simulated annealing fallback, and plotting using coordinates if
they exist in the CSV files.
"""
import tempfile
import os
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from qbm import core


def save_uploaded(uploaded) -> str:
    # Save uploaded file to a temporary path and return path
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tf.write(uploaded.getbuffer())
    tf.flush(); tf.close()
    return tf.name


def plot_route_fig(route, cities, D, coords=None, coord_labels=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    N = len(cities)
    if coords and all(c in coords for c in cities):
        # use provided coordinates
        xs = [coords[c][0] for c in cities]
        ys = [coords[c][1] for c in cities]
        # map route indices to coordinates
        rx = [xs[i] for i in route] + [xs[route[0]]]
        ry = [ys[i] for i in route] + [ys[route[0]]]
        ax.scatter(xs, ys, c='blue')
        for i, label in enumerate(cities):
            ax.text(xs[i], ys[i], str(label), fontsize=8)
        ax.plot(rx, ry, '-o', color='red')
        ax.set_title('Route (coordinates from CSV)')
        # use provided semantic labels if available
        if coord_labels and len(coord_labels) == 2 and all(isinstance(s, str) for s in coord_labels):
            ax.set_xlabel(str(coord_labels[0]))
            ax.set_ylabel(str(coord_labels[1]))
        else:
            ax.set_xlabel('X (coord)')
            ax.set_ylabel('Y (coord)')
        ax.grid(True, linestyle='--', alpha=0.5)
    else:
        # circular layout fallback
        angles = [2 * np.pi * i / max(N, 1) for i in range(N)]
        xs = [np.cos(a) for a in angles]
        ys = [np.sin(a) for a in angles]
        rx = [xs[i] for i in route] + [xs[route[0]]]
        ry = [ys[i] for i in route] + [ys[route[0]]]
        ax.scatter(xs, ys, c='blue')
        for i, label in enumerate(cities):
            ax.text(xs[i] + 0.02, ys[i] + 0.02, str(label), fontsize=8)
        ax.plot(rx, ry, '-o', color='red')
        ax.set_title('Route (circular layout)')
        ax.set_xlabel('X (relative)')
        ax.set_ylabel('Y (relative)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
    return fig


def main():
    st.set_page_config(page_title="QBM Route Optimisation", layout="wide")
    st.title("Quantum Boltzmann Machine — Route Optimisation")

    with st.sidebar:
        st.header("Inputs")
        uploaded_dist = st.file_uploader("Distance CSV", type=["csv"])
        uploaded_orders = st.file_uploader("Orders CSV", type=["csv"])
        dist_path = st.text_input("Or provide distance CSV path", value="")
        orders_path = st.text_input("Or provide orders CSV path", value="")

        # Units selector (will convert displayed lengths assuming CSV distances are in meters)
        unit_options = ["distance units (native)", "km", "miles"]
        default_unit = st.session_state.get('units', "distance units")
        try:
            default_index = unit_options.index(default_unit)
        except Exception:
            default_index = 0
        unit = st.selectbox("Units", unit_options, index=default_index,
                            help="Assumes distance values in CSV are in meters; choose km or miles to convert displayed lengths.")
        st.session_state['units'] = unit

        # Define conversion scales (from meters)
        unit_scales = {
            'distance units (native)': 1.0,
            'km': 0.001,
            'miles': 0.000621371,
        }
        scale = unit_scales.get(unit, 1.0)

        # QUBO and SA parameters
        A_val = st.number_input('Constraint penalty A', value=1000.0, step=100.0, format="%f")
        B_val = st.number_input('Travel cost scale B', value=1.0, step=0.1, format="%f")
        sa_steps = st.number_input('SA steps', value=20000, step=1000)
        sa_restarts = st.number_input('SA restarts', value=1, min_value=1, step=1)
        num_reads = st.number_input('D-Wave num_reads (if available)', value=100, min_value=1, step=1)

        if st.button("Load & Prepare"):
            try:
                # prefer uploaded files
                if uploaded_dist and uploaded_orders:
                    p1 = save_uploaded(uploaded_dist)
                    p2 = save_uploaded(uploaded_orders)
                    info = core.run_problem(p1, p2)
                    # cleanup temp files later
                    st.session_state['_tmp_files'] = st.session_state.get('_tmp_files', []) + [p1, p2]
                else:
                    if not dist_path or not orders_path:
                        st.error("Please supply either uploaded files or file paths for both CSVs.")
                        st.stop()
                    info = core.run_problem(dist_path, orders_path)
                st.session_state['problem'] = info
                st.success(f"Loaded problem with {info['N']} cities")
                if info.get('coords'):
                    st.info(f"Coordinates detected for {len(info['coords'])} locations")
                # show semantic coordinate column names if provided by core.run_problem
                if info.get('coord_cols'):
                    xcol, ycol = info.get('coord_cols')
                    if xcol and ycol:
                        st.info(f"Detected coordinate columns: {xcol}, {ycol}")
                else:
                    st.info("No coordinates detected; using circular layout")
            except Exception as e:
                st.exception(e)

        # quick sample loader
        # quick sample loader (single-click) and selector for multiple included samples
        sample_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        samples = {
            'Default sample': ('sample_distance.csv', 'sample_orders.csv'),
            '4-city sample': ('sample_distance_4.csv', 'sample_orders_4.csv'),
            'Coords sample': ('sample_distance_coords.csv', 'sample_orders_coords.csv'),
            '8-city sample': ('sample_distance_8.csv', 'sample_orders_8.csv'),
            'Asymmetric sample': ('sample_distance_asymmetric.csv', 'sample_orders_asymmetric.csv'),
        }
        sample_names = list(samples.keys())
        chosen = st.selectbox('Choose a sample dataset', sample_names)
        if st.button('Load selected sample'):
            try:
                dist_fname, orders_fname = samples.get(chosen, samples['Default sample'])
                p1 = os.path.join(sample_dir, dist_fname)
                p2 = os.path.join(sample_dir, orders_fname)
                if not (os.path.exists(p1) and os.path.exists(p2)):
                    st.error(f"Sample files not found in {sample_dir}: {dist_fname} / {orders_fname}")
                else:
                    info = core.run_problem(p1, p2)
                    st.session_state['problem'] = info
                    st.session_state['_last_loaded_sample'] = chosen
                    st.success(f"Loaded '{chosen}' with {info['N']} cities")
                    if info.get('coords'):
                        st.info(f"Coordinates detected for {len(info['coords'])} locations")
                    if info.get('coord_cols'):
                        xcol, ycol = info.get('coord_cols')
                        if xcol and ycol:
                            st.info(f"Detected coordinate columns: {xcol}, {ycol}")
            except Exception as e:
                st.exception(e)

    # main area
    st.header("Solvers & Results")
    prob = st.session_state.get('problem')
    col1, col2 = st.columns([1, 2])
    if prob:
        with col1:
            st.write(f"Cities: {prob['N']}")
            if st.button("Run Greedy"):
                route = core.nearest_neighbour_from_matrix(prob['D'], 0)
                length = sum(prob['D'][route[i], route[(i+1)%len(route)]] for i in range(len(route)))
                unit_label = st.session_state.get('units', 'distance units (native)')
                # apply unit conversion scale
                length_display = length * scale
                # build Q with user-selected A/B for diagnostics
                Q_current = core.build_tsp_qubo(prob['D'], A=A_val, B=B_val)
                greedy_onehot = core.onehot_from_route(route, prob['N'])
                greedy_energy = core.qubo_energy(Q_current, greedy_onehot)
                st.write(f"Greedy length: {length_display:.2f} ({unit_label})")
                st.write(f"Greedy QUBO energy: {greedy_energy:.2f}")
                # show decoded route (city names) and bit-vector counts
                route_cities = [prob['cities'][i] for i in route]
                st.markdown("**Decoded route (cities):**")
                st.write(route_cities)
                onehot = core.onehot_from_route(route, prob['N'])
                ones = int(onehot.sum())
                st.write(f"Bit-vector ones (should equal N): {ones} / {prob['N']}")
                fig = plot_route_fig(route, prob['cities'], prob['D'], coords=prob.get('coords'), coord_labels=prob.get('coord_cols'))
                st.pyplot(fig)

            # Run both a raw (potentially infeasible) sampler and a feasible SA (initialized from greedy)
            if st.button("Run Raw (infeasible) then Feasible SA"):
                with st.spinner("Running raw sampler (or simulated raw) and then feasible SA..."):
                    # rebuild Q with chosen A/B
                    Q_current = core.build_tsp_qubo(prob['D'], A=A_val, B=B_val)

                    # greedy initialization (used for feasible SA and as a sensible start for simulated raw)
                    greedy_route = core.nearest_neighbour_from_matrix(prob['D'], 0)
                    greedy_onehot = core.onehot_from_route(greedy_route, prob['N'])

                    # --- Raw sampler (attempt hardware; otherwise simulate an infeasible bit-flip SA) ---
                    raw_x = None
                    raw_energy = None
                    # If D-Wave is available, try to get a raw sample directly
                    if getattr(core, 'DWAVE_AVAILABLE', False):
                        try:
                            from dwave.system import DWaveSampler, EmbeddingComposite
                            Q_dwave = {(i, j): float(w) for (i, j), w in Q_current.items()}
                            sampler = EmbeddingComposite(DWaveSampler())
                            response = sampler.sample_qubo(Q_dwave, num_reads=int(num_reads))
                            best = response.first
                            sample = best.sample
                            raw_x = np.array([int(sample.get(i, 0)) for i in range(prob['N'] * prob['N'])], dtype=int)
                            raw_energy = float(best.energy)
                        except Exception as e:
                            st.warning(f"D-Wave sampling failed; falling back to simulated raw sampler: {e}")

                    # If no raw_x from hardware, simulate a raw/infeasible sampler via bit-flip SA
                    if raw_x is None:
                        def simulate_infeasible(Q, N, steps):
                            # start from greedy one-hot (sensible starting point) but allow bit-flips
                            x = greedy_onehot.copy()
                            best_x = x.copy()
                            best_e = core.qubo_energy(Q, x)
                            T = 1000.0
                            for t in range(steps):
                                i = np.random.randint(0, N * N)
                                x_new = x.copy()
                                x_new[i] = 1 - x_new[i]
                                e_new = core.qubo_energy(Q, x_new)
                                # track best seen (may be infeasible)
                                if e_new < best_e:
                                    best_x, best_e = x_new.copy(), e_new
                                # Metropolis acceptance
                                e_curr = core.qubo_energy(Q, x)
                                if e_new < e_curr or np.random.rand() < np.exp((e_curr - e_new) / max(T, 1e-12)):
                                    x = x_new
                                T *= 0.99995
                            return best_x, best_e

                        raw_x, raw_energy = simulate_infeasible(Q_current, prob['N'], int(sa_steps))

                    # --- Feasible SA (swap-based), initialized from greedy ---
                    feasible_x, feasible_e = core.simulated_annealing_local(Q_current, prob['N'], steps=int(sa_steps), init_onehot=greedy_onehot)

                    # decode routes and compute lengths
                    raw_route = core.decode_onehot_to_route(raw_x, prob['N'])
                    feasible_route = core.decode_onehot_to_route(feasible_x, prob['N'])
                    raw_length = sum(prob['D'][raw_route[i], raw_route[(i + 1) % len(raw_route)]] for i in range(len(raw_route)))
                    feasible_length = sum(prob['D'][feasible_route[i], feasible_route[(i + 1) % len(feasible_route)]] for i in range(len(feasible_route)))
                    unit_label = st.session_state.get('units', 'distance units (native)')
                    # apply unit conversion
                    raw_length_display = raw_length * scale
                    feasible_length_display = feasible_length * scale

                    # Diagnostics: feasibility, row/col sums and Hamming distance
                    def summarize_x(x, N):
                        xa = np.asarray(x, dtype=int).flatten()
                        mat = xa.reshape(N, N)
                        row_sums = mat.sum(axis=1).tolist()
                        col_sums = mat.sum(axis=0).tolist()
                        ones = int(xa.sum())
                        feasible = (ones == N) and all(int(v) == 1 for v in row_sums) and all(int(v) == 1 for v in col_sums)
                        return {
                            'ones': ones,
                            'row_sums': row_sums,
                            'col_sums': col_sums,
                            'feasible': feasible,
                        }

                    raw_summary = summarize_x(raw_x, prob['N'])
                    feas_summary = summarize_x(feasible_x, prob['N'])
                    hamming = int((np.asarray(raw_x, dtype=int).flatten() != np.asarray(feasible_x, dtype=int).flatten()).sum())

                    # Display side-by-side
                    st.markdown(f"**Diagnostics:** Hamming(raw,feasible) = {hamming} — Raw feasible? {raw_summary['feasible']} — Feasible-SA feasible? {feas_summary['feasible']}")
                    left, right = st.columns(2)
                    with left:
                        st.subheader("Raw (infeasible) sampler")
                        st.write(f"Energy (raw): {raw_energy:.2f}")
                        # show diagnostic summary for raw
                        st.write(f"Feasible: {raw_summary['feasible']}")
                        st.write(f"Ones: {raw_summary['ones']} — row sums: {raw_summary['row_sums']} — col sums: {raw_summary['col_sums']}")
                        st.write(f"Route length (raw): {raw_length_display:.2f} ({unit_label})")
                        st.markdown("**Decoded route (raw)**")
                        st.write([prob['cities'][i] for i in raw_route])
                        ones_raw = int(raw_x.sum())
                        st.write(f"Bit-vector ones (raw): {ones_raw} / {prob['N']}")
                        bits_preview = ','.join(map(str, raw_x.tolist()[:min(80, raw_x.size)]))
                        st.write(f"Bits (first {min(80, raw_x.size)}): {bits_preview}")
                        fig_raw = plot_route_fig(raw_route, prob['cities'], prob['D'], coords=prob.get('coords'), coord_labels=prob.get('coord_cols'))
                        st.pyplot(fig_raw)

                    with right:
                        st.subheader("Feasible SA (initialized from greedy)")
                        st.write(f"Energy (feasible): {feasible_e:.2f}")
                        # show diagnostic summary for feasible
                        st.write(f"Feasible: {feas_summary['feasible']}")
                        st.write(f"Ones: {feas_summary['ones']} — row sums: {feas_summary['row_sums']} — col sums: {feas_summary['col_sums']}")
                        st.write(f"Route length (feasible): {feasible_length_display:.2f} ({unit_label})")
                        st.markdown("**Decoded route (feasible)**")
                        st.write([prob['cities'][i] for i in feasible_route])
                        ones_feas = int(feasible_x.sum())
                        st.write(f"Bit-vector ones (feasible): {ones_feas} / {prob['N']}")
                        fig_feas = plot_route_fig(feasible_route, prob['cities'], prob['D'], coords=prob.get('coords'), coord_labels=prob.get('coord_cols'))
                        st.pyplot(fig_feas)

                    # Add a short explanation tooltip / expander clarifying differences and A/B tuning advice
                    with st.expander("Why raw vs repaired differ — quick notes and A/B tuning tips"):
                        st.markdown(
                            """
                        - Raw sampler outputs may be infeasible because physical samplers or naive bit-flip search don't enforce the one-hot constraints; this can produce very negative linear terms and misleading energies.
                        - The feasible SA preserves the one-hot constraints by swapping city positions (route-preserving moves) so lengths and energies reflect valid routes.
                        - Recommended A/B tuning workflow:
                          1. Increase penalty A until raw samples rarely violate one-hot constraints (large A penalises constraint violations).
                          2. Then tune B (travel-cost scale) to improve route quality without allowing constraint violations to dominate.
                          3. Use the feasible SA as a baseline to compare realistic route lengths; use the raw sampler only to study sampler behaviour or embedding effects.
                        """
                        )

        with col2:
            st.subheader("Last Output / Log")
            st.write("Use the buttons on the left to run solvers. Results and plots appear here.")
    else:
        st.info("Load a problem (Distance + Orders CSV) in the sidebar to start.")


if __name__ == '__main__':
    main()
