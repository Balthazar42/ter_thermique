from graph import DiffusionGraph, line
from boundary_cond import fixed_temp
import numpy as np
from matplotlib import pyplot as plt


def transient_response(graph: DiffusionGraph, input: int=0):
    temp = 10
    graph.boundary_conditions[input] = fixed_temp(temp)
    initial = np.zeros(graph.n)
    initial[input] = temp

    t, U = graph.simulate(
        t_start=0,
        t_end=100,
        dt=0.1,
        initial=initial,
    )

    energy = graph.C @ U - temp * graph.C[input]
    final_energy = (sum(graph.C) - graph.C[input]) * temp
    a, b = np.polyfit(t, np.log(1-energy/final_energy), deg=1, w=np.sqrt(1 - energy/final_energy))
    tau = - 1 / a
    energy_fit = final_energy * (1 - np.exp(-t / tau))

    return t, U, energy, energy_fit, tau



if __name__ == "__main__":
    g = line(5, 1, 1)
    t, U, energy, energy_fit, tau = transient_response(g)
    print(f"{tau=}")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, U.T)
    ax1.set_ylabel("Température")

    ax2.plot(t, energy, label="Energie calculée")
    ax2.plot(t, energy_fit, label="Régression")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel("Energie")
    ax2.legend()

    plt.show()
