from collections.abc import Callable
from graph import DiffusionGraph
import numpy as np


def fixed_temp(temp: float | Callable[[float], float]) -> Callable[[DiffusionGraph, int], float]:
    """
    Create a fixed temperature boundary condition for a graph vertex\n
    The temperature may be specified as a constant or as a function of time\n

    Every time step, the temperature of the vertex will be\n
    set to the constant (or the current value of the function)\n
    regardless of the state of the rest of the graph\n

    :param temp: The vertex's temperature
    :return: Fixed temperature boundary condition function
    """
    # If temp is a constant, make it a constant function
    if not callable(temp):
        c = temp
        temp = lambda t: c

    def temp_bc(graph: DiffusionGraph, i: int) -> float:
        return temp(graph.t)

    return temp_bc


def heater(power: float | Callable[[float], float]) -> Callable[[DiffusionGraph, int], float]:
    """
    Create a heater boundary condition for a graph vertex\n
    The heater's power output may be specified as a constant or as a function of time\n
    Positive values for power cause heating, while negative values cause cooling\n

    In addition to exchanging heat with adjacent vertices normally,
    the vertex will receive power(t) * dt extra energy every time step\n

    :param power: The heater's power output
    :return: Heater boundary condition function
    """
    # If power is a constant, make it a constant function
    if not callable(power):
        c = power
        power = lambda t: c

    def heater_bc(graph: DiffusionGraph, i: int) -> float:
        new_U = graph.U[i]
        new_U += graph.dt * np.sum(graph.A[i] * graph.U)
        new_U += graph.dt * (power(graph.t)) / graph.C[i]
        return new_U

    return heater_bc


def heater_thermostat(power: float, threshold: float) -> Callable[[DiffusionGraph, int], float]:
    """
    Create a thermostatic heater boundary condition for a graph vertex\n
    Positive values for power cause heating, while negative values cause cooling\n


    In addition to exchanging heat with adjacent vertices normally,
    the vertex will receive power(t) * dt extra energy every time step\n
    if and only if the current temperature of the vertex exceeds the threshold\n

    :param power: The heater's power output
    :param threshold: The heater's activation threshold
    :return: Thermostatic heater boundary condition function
    """

    def tms_heater_bc(graph: DiffusionGraph, i: int) -> float:
        new_U = graph.U[i]
        new_U += graph.dt * np.sum(graph.A[i] * graph.U)
        if (power > 0 and graph.U[i] < threshold) or (power < 0 and graph.U[i] > threshold):
            new_U += graph.dt * power / graph.C[i]
        return new_U

    return tms_heater_bc