import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.animation import ArtistAnimation
from matplotlib.patheffects import TickedStroke, Normal
from matplotlib import pyplot as plt


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
        self.A = np.diag(1 / C) @ (G - np.diag(np.sum(G, axis=1)))

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

    def mean_temp(self) -> float:
        """
        Weighted mean temperature of the graph
        Constant over time if the system is isolated
        """
        return np.sum(self.C * self.T) / np.sum(self.C)

    def msd_temp(self) -> float:
        """
        Weighted mean square deviation of the temperatures
        Monotonically decreasing over time if the system is isolated
        """
        return np.sum(self.C * (self.T - self.mean_temp()) ** 2) / np.sum(self.C)

    def draw_vertices(self) -> Artist:
        """
        Plot the vertices of the graph to this object's Axes
        :return: Artist representing the plotted points
        """
        scatter = self.ax.scatter(
            x=self.pos[:, 0],
            y=self.pos[:, 1],
            s=self.C / np.max(self.C) * 500,
            c=self.T,
            marker="o",
            cmap="plasma",
            vmin=np.min(self.initial),
            vmax=np.max(self.initial),
            edgecolors=["w" if bc is None else "g" for bc in self.boundary_conditions],
            linewidths=3,
            zorder=np.inf,
        )
        return scatter

    def draw_edges(self) -> list[Artist]:
        """
        Plot the edges of the graph to this object's Axes
        :return: list of Artists representing the plotted lines
        """
        edges = []

        for i in range(self.n):
            for j in range(i):
                if self.G[i, j] != 0:
                    angle = np.pi / 2 + np.atan(3 * self.G[i, j] * (self.T[i] - self.T[j]) / np.max(self.G) / (np.max(self.initial) - np.min(self.initial)))
                    edges.extend(self.ax.plot(
                        (self.pos[i, 0], self.pos[j, 0]), # x
                        (self.pos[i, 1], self.pos[j, 1]), # y
                        color="w",
                        linewidth=self.G[i, j] / np.max(self.G) * 2,
                        path_effects=[
                            TickedStroke(spacing=30, angle= angle * 180 / np.pi, length=0.2),
                            TickedStroke(spacing=30, angle=-angle * 180 / np.pi, length=0.2),
                            Normal(),
                        ],
                    ))

        return edges

    def animate(self, t_start: float=None, t_end: float=None, dt: float=None, initial: np.ndarray=None) -> ArtistAnimation:
        """
        Simulate the behaviour of the graph and render it to the figure as an animation

        :param t_start: Simulation time start
        :param t_end: Simulation time end
        :param dt: Simulation time step
        :param initial: Initial temperature conditions
        :return: matplotlib ArtistAnimation, must be saved to a variable until show is called
        """
        self.t_start = t_start if t_start is not None else self.t_start
        self.t_end = t_end if t_end is not None else self.t_end
        self.dt = dt if dt is not None else self.dt
        self.initial = initial if initial is not None else self.initial

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

        anim_artists = []
        while self.t < self.t_end:
            # Draw the current state of the graph and add it to the animation
            anim_artists.append(self.draw_edges())
            anim_artists[-1].append(self.draw_vertices())

            # Explicit Euler integration of the ODE
            new_T = self.T + self.dt * self.A @ self.T

            # Compute boundary conditions if any
            for i in range(self.n):
                if self.boundary_conditions[i] is not None:
                    new_T[i] = self.boundary_conditions[i](self, i)

            # Apply changes
            self.t += self.dt
            self.T = new_T

        return ArtistAnimation(fig=self.fig, artists=anim_artists, interval=50)

    def simulate(self, t_start: float=None, t_end: float=None, dt: float=None, initial: np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate the behaviour of the graph and output the computed temperatures
        Returns a tuple (t, T) where
        t is a 1D array of the times for which the solution was computed
        T is a 2D array of the temperatures of the vertices over time
        T has the following shape : (number of vertices, number of time steps)

        :param t_start: Simulation time start
        :param t_end: Simulation time end
        :param dt: Simulation time step
        :param initial: Initial temperature conditions
        :return: vector of times, matrix of temperatures
        """
        self.t_start = t_start if t_start is not None else self.t_start
        self.t_end = t_end if t_end is not None else self.t_end
        self.dt = dt if dt is not None else self.dt
        self.initial = initial if initial is not None else self.initial

        if self.t_start is None or self.t_end is None or self.dt is None or self.initial is None:
            raise RuntimeError("Missing simulation parameters")

        t = np.arange(self.t_start, self.t_end, self.dt)
        T = np.empty((self.n, len(t)))
        T[:, 0] = self.initial

        self.t = self.t_start
        self.T = self.initial
        for j in range(1, len(t)):
            # Explicit Euler integration of the ODE
            T[:, j] = self.T + self.dt * self.A @ self.T

            # Compute boundary conditions if any
            for i in range(self.n):
                if self.boundary_conditions[i] is not None:
                    T[i, j] = self.boundary_conditions[i](self, i)

            # Apply changes to the graph
            self.t = t[j]
            self.T = T[:, j]

        return t, T





triangle = DiffusionGraph(
    n=3,
    C=np.array([1, 2, 1]),
    G=0.1 * np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]),
    pos=np.array([[-1, 0], [0, 1], [1, 0]]),
)

gigraphe = DiffusionGraph(
    n=8,
    C=np.array([5, 5, 1, 1, 1, 1, 3, 2]),
    G=0.2 * np.array([
        [0, 3, 2, 2, 0, 0, 0, 0],
        [3, 0, 0, 0, 2, 2, 1, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2, 0],
    ]),
    pos=np.array([
        [-2, 0],
        [0, 0],
        [-2.5, -1],
        [-1.5, -1],
        [-0.5, -1],
        [0.5, -1],
        [0, 2.5],
        [1, 2.5],
    ]),
)


def line(n: int, period: float=None) -> DiffusionGraph:
    """
    Create a line graph with uniform capacitance and conductance
    Optionally, force the first vertex's temperature to follow
    the function of time 10*sin(2pi/T)

    :param n: Number of vertices
    :param period: Optional period of the first vertex's temperature oscillation
    :return: Line graph
    """
    G = np.zeros((n, n))
    for i in range(n-1):
        G[i, i+1] = 1
        G[i+1, i] = 1
    g = DiffusionGraph(
        n=n,
        C=np.ones(n),
        G=G,
        pos=np.stack((np.arange(n), np.zeros(n)), axis=1)
    )

    if period is not None:
        def wave(graph: DiffusionGraph, i: int):
            return 10 * np.sin(2 * np.pi / period * graph.t)
        g.boundary_conditions[0] = wave

    return g


def plane(p: int, q: int) -> DiffusionGraph:
    """
    Create a square grid graph with uniform capacitance and conductance


    :param p: Number of lines
    :param q: Number of columns
    :return: Plane graph
    """

    G = np.zeros((p*q, p*q))
    for i in range(p):
        for j in range(q):
            if i != 0:
                G[i + j * p, i - 1 + j * p] = 1
            if i != p - 1:
                G[i + j * p, i + 1 + j * p] = 1
            if j != 0:
                G[i + j * p, i + (j - 1) * p] = 1
            if j != q - 1:
                G[i + j * p, i + (j + 1) * p] = 1

    x, y = np.meshgrid(np.arange(p), np.arange(q))
    pos = np.stack((x.flatten(), y.flatten()), axis=1)
    g = DiffusionGraph(
        n=p*q,
        C=np.ones(p*q),
        G=G*0.1,
        pos=pos,
    )
    return g


if __name__ == "__main__":
    p, q = 5, 5
    g = plane(p, q)
    # g.boundary_conditions[12] = lambda gr, i: 10 * np.sin(2*np.pi/10*gr.t)
    initial = np.random.uniform(-10, 10, (p, q))
    t, T = g.simulate(
        t_start=0,
        t_end=15,
        dt=0.1,
        initial=initial.flatten(),
    )
    print(t.shape, T.shape)
    print(t[0], T[:, 0])
    print(t[-1], T[:, -1])