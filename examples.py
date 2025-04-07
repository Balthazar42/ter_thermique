from graph import DiffusionGraph, gigraphe, line, plane
from capacitance import heater, harmonic_response, bode_diagram
import numpy as np
import matplotlib.pyplot as plt


# Diffusion pure, graphe quelconque
# g = gigraphe
# anim = g.animate(
#     t_start=0,
#     t_end=20,
#     dt=0.1,
#     initial=np.random.uniform(0, 10, g.n),
# )
# # anim.save("gigraphe_thermique.gif")
# plt.show()

# Diffusion pure, graphe ligne
# n=5
# g = line(n, cap=1, cond=0.25)
# anim = g.animate(
#     t_start=0,
#     t_end=20,
#     dt=0.1,
#     initial=np.random.uniform(0, 10, g.n),
# )
# g.fig.set_size_inches(8, 4)
# # anim.save("line.gif")
# plt.show()


# Diffusion avec forçage sur le premier sommet, graphe ligne
# n=5
# g = line(n, period=10)
# anim = g.animate(
#     t_start=0,
#     t_end=30,
#     dt=0.1,
#     initial=np.linspace(-10, 10, n),
# )
# g.fig.set_size_inches(8, 4)
# # anim.save("line.gif")
# plt.show()


# Diffusion pure, graphe plan
# p, q = 5, 5
# g = plane(p, q, cap=1, cond=0.1)
# initial = np.random.uniform(-10, 10, (p, q))
# anim = g.animate(
#     t_start=0,
#     t_end=15,
#     dt=0.1,
#     initial=initial.flatten(),
# )
# g.fig.set_size_inches(8, 8)
# # anim.save("plane.gif")
# plt.show()


# Diffusion avec forçage sur le sommet central, graphe plan
# p, q = 5, 5
# g = plane(p, q)
# g.boundary_conditions[p//2*5+q//2] = lambda gr, i: 10 * np.sin(2*np.pi/10*gr.t)
# initial = np.random.uniform(-10, 10, (p, q))
# anim = g.animate(
#     t_start=0,
#     t_end=15,
#     dt=0.1,
#     initial=initial.flatten(),
# )
# g.fig.set_size_inches(8, 8)
# # anim.save("plane.gif")
# plt.show()


# Tests en régime harmonique (work in progress)
# g = line(10, 1, 1)
#
# (t, input, output), a, phi = harmonic_response(g, 1e-1)
# print(f"Amplitude = {a}, Phase = {phi}")
# plt.plot(t, input, label="Input power")
# plt.plot(t, output, label="First node temperature")
# plt.xlabel("Time")
# plt.legend()
#
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