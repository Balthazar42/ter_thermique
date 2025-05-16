from collections.abc import Callable
from graph import DiffusionGraph, line, gigraphe
from boundary_cond import heater
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps


def harmonic_response(graph: DiffusionGraph, freq: float, input: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.array]:
    power = lambda t: np.sin(2*np.pi*freq*t)
    graph.boundary_conditions[input] = heater(power)
    t, U = graph.simulate(
        t_start=0,
        t_end= 4 / freq,
        dt=1 / (freq * 1000),
        initial=np.zeros(graph.n),
    )

    c_ampl = 1 / 1000 * np.sum(U[:, -1000:] * np.exp(-2j * np.pi * freq * t[-1000:]), axis=1)
    c_ampl = 2j * c_ampl
    return t, power(t), U, c_ampl


def bode_diagram(graph: DiffusionGraph, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.geomspace(low, high, 64)
    amplitudes = np.zeros((freqs.size, graph.n), dtype=complex)

    for i in range(len(freqs)):
        _, _, _, c_ampl = harmonic_response(graph, freqs[i])
        amplitudes[i] = c_ampl

    return freqs, amplitudes



if __name__ == "__main__":
    g = line(10, 1, 1)

    t, input, U, m = harmonic_response(g, 1)
    print(np.abs(m), np.angle(m)/np.pi, sep="\n")

    eigvals = np.linalg.eigvalsh(g.A)
    print(eigvals)
    freqs, amplitudes = bode_diagram(g, 0.0025, 10)
    X, Y = np.real(amplitudes), np.imag(amplitudes)
    print(X.shape)
    colours = colormaps["viridis"](np.linspace(0, 1, len(X.T)))
    for x, y, col in zip(X.T, Y.T, colours):
        plt.plot(x, y, color=col)
    plt.axis("equal")
    plt.show()

    # freqs, amplitudes, phases = bode_diagram(g, 1e-1, 100)
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.set_ylabel("Amplitude")
    # # ax1.set_xlabel("Frequency")
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    # ax1.plot(freqs, amplitudes)
    #
    # ax2.set_ylabel("Phase")
    # ax2.set_xlabel("Frequency")
    # ax2.set_xscale("log")
    # ax2.plot(freqs, phases)
    #
    # plt.figure()
    # derlog = np.log(amplitudes)
    # derlog = derlog[1:] - derlog[:-1]
    # # print(derlog)
    # plt.xlabel("Frequency")
    # plt.semilogx()
    # plt.plot(freqs[:-1], derlog)
    # plt.show()