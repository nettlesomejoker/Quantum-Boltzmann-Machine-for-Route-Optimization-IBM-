"""
Simple Tkinter UI for the Quantum Boltzmann Machine route optimisation.
This UI lets the user choose the distance and orders CSVs, then run
Greedy, Simulated Annealing (local), or Quantum (D-Wave if available,
otherwise local SA) solvers and view results.

Run with: python -m qbm.ui
"""
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import time

from qbm import core

# Matplotlib for embedded plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class QBMApp:
    def __init__(self, root):
        self.root = root
        root.title("QBM Route Optimisation")

        frm = tk.Frame(root)
        frm.pack(padx=10, pady=10)

        tk.Label(frm, text="Distance CSV:").grid(row=0, column=0, sticky="w")
        self.dist_entry = tk.Entry(frm, width=60)
        self.dist_entry.grid(row=0, column=1, padx=5)
        tk.Button(frm, text="Browse", command=self.browse_dist).grid(row=0, column=2)

        tk.Label(frm, text="Orders CSV:").grid(row=1, column=0, sticky="w")
        self.orders_entry = tk.Entry(frm, width=60)
        self.orders_entry.grid(row=1, column=1, padx=5)
        tk.Button(frm, text="Browse", command=self.browse_orders).grid(row=1, column=2)

        tk.Button(frm, text="Load & Prepare", command=self.load_prepare).grid(row=2, column=0, pady=8)
        tk.Button(frm, text="Run Greedy", command=self.run_greedy).grid(row=2, column=1)
        tk.Button(frm, text="Run Quantum/Fallback", command=self.run_quantum).grid(row=2, column=2)

        self.output = scrolledtext.ScrolledText(root, width=90, height=20)
        self.output.pack(padx=10, pady=(0,10))

        # Figure for plotting routes
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(padx=10, pady=(0,10))
        # add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.state = {}

    def browse_dist(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"), ("All files","*")])
        if path:
            self.dist_entry.delete(0, tk.END); self.dist_entry.insert(0, path)

    def browse_orders(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"), ("All files","*")])
        if path:
            self.orders_entry.delete(0, tk.END); self.orders_entry.insert(0, path)

    def log(self, *args):
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        self.output.insert(tk.END, f"[{t}] " + " ".join(str(a) for a in args) + "\n")
        self.output.see(tk.END)

    def load_prepare(self):
        dist = self.dist_entry.get().strip()
        orders = self.orders_entry.get().strip()
        if not dist or not orders:
            messagebox.showwarning("Missing files", "Please select both CSV files before loading.")
            return

        def job():
            try:
                self.log("Loading files...")
                info = core.run_problem(dist, orders)
                self.state.update(info)
                self.log(f"Loaded problem with {info['N']} cities")
            except Exception as e:
                self.log("Error preparing problem:", e)
                messagebox.showerror("Error", str(e))

        threading.Thread(target=job, daemon=True).start()

    def run_greedy(self):
        if 'D' not in self.state:
            messagebox.showwarning("Not loaded", "Please load and prepare data first.")
            return

        def job():
            D = self.state['D']
            cities = self.state['cities']
            N = self.state['N']
            self.log("Running greedy...")
            route = core.nearest_neighbour_from_matrix(D, 0)
            length = sum(D[route[i], route[(i+1)%len(route)]] for i in range(len(route)))
            readable = ' -> '.join(str(cities[i]) for i in route)
            self.log(f"Greedy length: {length}")
            self.log(f"Route: {readable}")
            # plot route
            self.plot_route(route, cities, D)

        threading.Thread(target=job, daemon=True).start()

    def run_quantum(self):
        if 'Q' not in self.state:
            messagebox.showwarning("Not loaded", "Please load and prepare data first.")
            return

        def job():
            Q = self.state['Q']
            N = self.state['N']
            D = self.state['D']
            cities = self.state['cities']
            self.log("Running quantum/fallback solver...")
            x, energy = core.quantum_annealing_solve(Q, N)
            route = core.decode_onehot_to_route(x, N)
            length = sum(D[route[i], route[(i+1)%len(route)]] for i in range(len(route)))
            readable = ' -> '.join(str(cities[i]) for i in route)
            self.log(f"Quantum/fallback energy: {energy}")
            self.log(f"Length: {length}")
            self.log(f"Route: {readable}")
            # plot route
            self.plot_route(route, cities, D)

        threading.Thread(target=job, daemon=True).start()
    def plot_route(self, route, cities, D):
        """Plot the given route on the embedded matplotlib canvas.
        The plot uses a simple 2D layout where cities are placed on a circle
        (since we don't have coordinates in the CSV)."""
        # create circular layout
        N = len(cities)
        if N == 0:
            return
        angles = [2 * 3.141592653589793 * i / N for i in range(N)]
        xs = [np.cos(a) for a in angles]
        ys = [np.sin(a) for a in angles]
        self.ax.clear()
        # draw nodes
        self.ax.scatter(xs, ys, c='blue')
        for i, label in enumerate(cities):
            # small offset for labels
            self.ax.text(xs[i] + 0.02, ys[i] + 0.02, str(label), fontsize=8, ha='left')
        # draw route edges
        route_pts_x = [xs[i] for i in route] + [xs[route[0]]]
        route_pts_y = [ys[i] for i in route] + [ys[route[0]]]
        self.ax.plot(route_pts_x, route_pts_y, '-o', color='red')
        self.ax.set_title('Route (circular layout)')
        self.ax.axis('equal')
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = QBMApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
