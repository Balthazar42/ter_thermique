import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class DiffusionGraph:
    """
    A graph modeling heat conduction in a network of capacitive bodies at uniform temperatures
    """
    def __init__(self, n: int, C: np.ndarray, G: np.ndarray, pos: np.ndarray):
        """
        :param n: number of vertices
        :param C: capacitance of each vertex, n positive floats
        :param G: conductance of each edge, n * n positive floats, symmetric
        :param pos: visual position of each vertex, n * 2 floats
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
        self.fig, self.ax = None, None # Matplotlib figure and axes
        self.anim = []

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

    def start_simulation(self):
        """
        Sets/resets the graph to its initial state
        Create a new figure and clear the animation
        """
        if self.t_start is None or self.t_end is None or self.dt is None or self.initial is None:
            raise RuntimeError("Missing simulation parameters")
        self.t = self.t_start
        self.T = self.initial
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()
        self.anim = []

    def simulation_step(self):
        """
        Compute the next time step of the simulation
        """
        # Explicit Euler integration of the ODE
        new_T = self.T + self.dt * self.A * self.T

        # Compute and apply boundary conditions
        for i in range(self.n):
            if self.boundary_conditions[i] is not None:
                new_T[i] = self.boundary_conditions[i](self, i)

        # Apply changes
        self.t += self.dt
        self.T = new_T

