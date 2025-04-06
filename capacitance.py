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


def harmonic_response(graph: DiffusionGraph, freq: float) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, float]:
    power = lambda t: np.sin(2*np.pi*freq*t)
    graph.boundary_conditions[0] = heater(power)
    t, T = graph.simulate(
        t_start=0,
        t_end= 4 / freq,
        dt=1 / (freq * 1000),
        initial=np.zeros(graph.n),
    )

    m = 1/1000 * np.sum(T[0][-1000:] * np.exp(-2j * np.pi * freq * t[-1000:]))
    return (t, power(t), T[0]), 2 * np.abs(m), np.angle(m) + np.pi / 2


def bode_diagram(graph: DiffusionGraph, low: float, high: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.geomspace(low, high, 64)
    amplitudes = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)

    for i in range(len(freqs)):
        _, a, phi = harmonic_response(graph, freqs[i])
        amplitudes[i] = a
        phases[i] = phi

    return freqs, amplitudes, phases



if __name__ == "__main__":
    n = 5
    g = line(n)

    (t, input, output), a, phi = harmonic_response(g, 2e-1)
    print(f"Amplitude = {a}, Phase = {phi}")
    plt.plot(t, input, label="Input power")
    plt.plot(t, output, label="First node temperature")
    plt.legend()


    freqs, amplitudes, phases = bode_diagram(g, 0.01, 100)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_ylabel("Amplitude")
    # ax1.set_xlabel("Frequency")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(freqs, amplitudes)

    ax2.set_ylabel("Phase")
    ax2.set_xlabel("Frequency")
    ax2.set_xscale("log")
    ax2.plot(freqs, phases)
    plt.show()