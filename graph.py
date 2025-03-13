import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class DiffusionGraph:
    """
    A graph modeling heat conduction in a network of capacitive bodies at uniform temperatures
    """
    def __init__(self, n: int, C: np.ndarray, G: np.ndarray, pos: np.ndarray, fig: Figure=None, ax: Axes=None):
        """
        :param n: number of vertices
        :param C: capacitance of each vertex, n positive floats
        :param G: conductance of each edge, n * n positive floats, symmetric
        :param pos: visual position of each vertex, n * 2 floats
        :param fig: matplotlib Figure on which to draw, will be created automatically if unspecified
        :param ax: matplotlib Axes on which to draw, will be created automatically if unspecified
        """
        # Intrinsic properties
        self.n = n
        self.C = C
        self.G = G
        # Matrix for the ODE system dT/dt = A * T
        self.A = np.diag(1 / C) * (G - np.diag(np.sum(G, axis=1)))

        # Simulation parameters and data
        self.t_start = None     # Simulation time start
        self.t_end = None       # Simulation time end
        self.dt = None          # Simulation time step
        self.initial = None     # Initial temperatures
        self.boundary_conditions: list[callable] = [None] * n

        self.t = 0              # Time
        self.T = np.zeros(n)    # Temperatures of the nodes

        # Plotting information
        self.pos = pos
        self.fig, self.ax = fig, ax # Matplotlib figure and axes
        self.anim_artists = []

    def mean_T(self) -> float:
        """
        Weighted mean temperature of the graph
        Constant over time if the system is isolated
        """
        return np.sum(self.C * self.T) / np.sum(self.C)

    def msd_T(self) -> float:
        """
        Weighted mean square deviation of the temperatures
        Monotonically decreasing over time if the system is isolated
        """
        return np.sum(self.C * (self.T - self.mean_T()) ** 2) / np.sum(self.C)

    def draw_vertices(self) -> Artist:
        """
        Plot the vertices of the graph to this object's Axes
        :return: Artist representing the plotted points
        """
        scatter = self.ax.scatter(
            x=self.pos[:, 0],
            y=self.pos[:, 1],
            s=self.C / np.max(self.C) * 50,
            c=self.T,
            marker="o",
            cmap="plasma",
            vmin=np.min(self.initial),
            vmax=np.max(self.initial),
            edgecolors=["w" if bc is None else "g" for bc in self.boundary_conditions],
        )
        return scatter

    def draw_edges(self) -> list[Artist]:
        pass

    def animate(self, t_start: float=None, t_end: float=None, dt: float=None, initial: np.ndarray[float]=None):
        self.t_start = t_start or self.t_start
        self.t_end = t_end or self.t_end
        self.dt = dt or self.dt
        self.initial = initial or self.initial

        if self.t_start is None or self.t_end is None or self.dt is None or self.initial is None:
            raise RuntimeError("Missing simulation parameters")

        self.t = self.t_start
        self.T = self.initial
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig.tight_layout()
            self.fig.set_facecolor("0.2")
            self.ax.set_facecolor("0.2")
            self.ax.set_axis_off()
        self.anim_artists = []

        while self.t < self.t_end:
            # Draw the current state of the graph and add it to the animation
            self.anim_artists.append(self.draw_edges())
            self.anim_artists[-1].append(self.draw_vertices())

            # Explicit Euler integration of the ODE
            new_T = self.T + self.dt * self.A * self.T

            # Compute boundary conditions if any
            for i in range(self.n):
                if self.boundary_conditions[i] is not None:
                    new_T[i] = self.boundary_conditions[i](self, i)

            # Apply changes
            self.t += self.dt
            self.T = new_T
