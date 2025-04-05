from collections.abc import Callable
from graph import DiffusionGraph, line, gigraphe
import numpy as np
from matplotlib import pyplot as plt


def heater(power: float | Callable[[float], float]) -> Callable[[DiffusionGraph, int], float]:
    """
    Create a heater boundary condition for a graph vertex
    The heater's power output may be specified as a constant or as a function of time

    In addition to exchanging heat with adjacent vertices normally,
    the vertex will receive power(t) * dt extra energy every time step
    Positive power values translate to energy added to the vertex

    :param power: The heater's power output
    :return: Heater boundary condition function
    """
    # If power is a constant, make it a constant function
    if not callable(power):
        power = lambda t: power

    def heater_bc(graph: DiffusionGraph, i: int) -> float:
        new_T = graph.T[i]
        new_T += graph.dt * np.sum(graph.A[i] * graph.T)
        new_T += graph.dt * power(graph.t) / graph.C[i]
        return new_T

    return heater_bc



if __name__ == "__main__":
    n = 8
    g = gigraphe
    t, T = g.simulate(
        t_start=0,
        t_end=20,
        dt=0.1,
        initial=np.linspace(0, 10, n)
    )
    mean_T = g.C @ T / np.sum(g.C)
    msd_T = g.C @ (T - mean_T) ** 2 / np.sum(g.C)
    plt.plot(mean_T)
    plt.figure()
    plt.plot(msd_T)
    plt.show()